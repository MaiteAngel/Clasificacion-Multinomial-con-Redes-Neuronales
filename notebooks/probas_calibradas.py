import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from scipy.optimize import minimize 
from sklearn.metrics import log_loss

def softmax(x):
    """
    Compute softmax values for each sets of scores in x.
    
    Parameters:
        x (numpy.ndarray): array containing m samples with n-dimensions (m,n)
    Returns:
        x_softmax (numpy.ndarray) softmaxed values for initial (m,n) array
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0, keepdims=1)

def loss_fun(x,probs, true):
    # Calculates the loss using log-loss (cross-entropy loss)
    #scaled_probs = .predict(probs, x)    
    loss = log_loss(y_true=true, y_pred=softmax(probs/x))
    return loss

def probas_calibradas(modelo,x_valid,x_test,y_valid,input_shape=(48,48,1),numero_clases=7): 	
	# Hold weights:
	weights = modelo.get_weights()

	# Model - softmax
	modelo_no_softmax = keras.Sequential([
	    keras.layers.InputLayer(input_shape=input_shape),
	    keras.models.Sequential(modelo.layers[:-1]),
	    keras.layers.Dense(numero_clases)
	])

	# 4. Pass the imagenet weights onto the second resnet
	modelo_no_softmax.set_weights(modelo.get_weights())

	y_logits_val=modelo_no_softmax.predict(x_valid)
	y_logits_test=modelo_no_softmax.predict(x_test)

	y_probs_val=modelo.predict(x_valid)
	y_probs_test=modelo.predict(x_test)

	true = y_valid.flatten() # Flatten y_val

	opt = minimize(loss_fun, x0 = 1, args=(y_logits_val, true))
	T= opt.x[0]

	y_probs_ts=[softmax(probs/T) for probs in y_logits_test]
	
	return y_probs_ts
