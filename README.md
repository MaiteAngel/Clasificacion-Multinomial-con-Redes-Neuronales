# Tesis: Un enfoque probabilístico de las redes neuronales aplicadas a la clasificación multinomial de imágenes.
## Departamento de Matemática, FCEN, UBA.


### Autor: Maite Angel

##### Notebooks creados en el marco de la tesis por la Licenciatura de Ciencias Matemáticas. 
##### Estan confeccionados para ser ejecutados en Google Colaboratory.


* _**Ejemplos_de_juguete_fc_vs_cnn.ipynb**_: Se estudia el salto de la discriminación logística a redes FC con una capa oculta levantando el supuesto de linealidad de los datos. Mostramos en un ejemplo práctico como la necesidad de un proceso de Feature engineering, donde se deben conocer las particularidades de los datos para manipularlos correctamente, se levanta con las redes neuronales FC donde este proceso se vuelve implícito a la hora del entrenamiento.

* _**Ejemplos_de_juguete_logistico_vs_fc.ipynb**_: Se estudia el desempeño de las redes convolucionales, sobre el dataset MINST de reconocimiento de dígitos. Se entrenan dos arquitecturas, una red FC y una CNN donde ambas se desempeñaron razonablemente bien en términos de predicción. Luego corrompemos el conjunto de datos de testeo, permitiendo pequeñas traslaciones de los objetos que nos dejaron mostrar un escenario donde las FC perdían todo su poder empeorando mucho en términos de, por ejemplo exactitud (0.3), en tanto las CNN mostraron un desempeño mucho más robusto mostrándose mucho más invariantes a estos cambios (0.7).

Además se estudia el problema de reconocimiento de expresiones faciales sobre el dataset FER2013 de la competencia de Kaggle _Challenges in Representation Learning: Facial Expression Recognition Challenge_. Se confeccionan los siguientes códigos:

* _**Expresiones_faciales_entrenamiento.ipynb**_: Se entrenan 4 modelos relevantes al marco teorico global de la tesis. Estos son: _regresión logística, redes fully connected, redes convolucionales_ y _redes preentrenadas_.

* _**Expresiones_faciales_testeo.ipynb**_: Se observan las métricas que se obtienen con los modelos previamente entrenados. Se exploran los resultados obtenidos en términos de calibración y mapas de calor entre otros.

