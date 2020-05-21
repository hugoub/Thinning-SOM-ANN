# Operational efficiency prediction in thinning and clear-cutting using artificial neural networks

The forestry sector is fundamental to Chile's economic development. Clear-cutting and thinning are two paramount processes in the forestry industry. In this article, artificial neural networks are used to estimate crop yields and analyze the thinning process. A multi-layer perceptron neural network is used to obtain clear-cutting forecasts, and a self-organized map (SOM) network to examine the thinning process. Both models are executed using GPU training in TensorFlow. For forecasting, 768 neural network configurations are implemented. The one with the best performance is selected based on the mean squared error criterion. The results are compared with those obtained in linear regression, showing a MSE and r^2 of 2484.56 and 0.78 respectively, versus 3979.21 and 0.64 obtained in the best linear regression model. Furthermore, three SOM structures are tested with three different number of nodes, a 2-dimensional, a toroidal and a 3-dimensional lattice. The best network corresponds to a 3-dimensional structure of 9x9x9 nodes. Additionally, the algorithms of k-means and silhouette coefficient are used to generate clusters. The best structure is selected by quantization error. Three clusters are obtained, and the analysis show that the attribute of slope of the land has greater influence in the thinning process. The results of both procedures ensure that both artificial neural networks are powerful tools for estimation and data analysis.

# Archivos e Instrucciones (Español)

Se muestra una breve explicación de como funciona cada uno de los archivos.

## linear_regression.py
Utiliza los datos del archivo **datos_corregidos_cosecha.csv** para generar las regresiones lineales. Es posible modificar el learning rate, la cantidad de epochs, el tamaño del mini-batch, y el k del cross validation.

- En la carpeta "Grafos", se almacena los grafos de las regresiones lineales generados por tensorflow.
- En la carpeta "Resultados", se almacenam los el R cuadrado, el MSE, el MAE y el tiempo de ejecución para cada k, además del promedio y la desviación estándar para cada uno.

El nombre de los archivos generados se constituye de la siguiente forma.

```
Nombre del archivo lr: lr-“learning rate”-“epochs”-“mini batch”-“k”
```
 Para los grafos generados, anteponer la palabra “grafo” al nombre del archivo lr.


 ## neural_network.py 
 Genera una Red Neuronal que utiliza los datos de **datos_corregidos_cosecha.csv**. Se puede modificar la función de activación, el número de capas, el número de nodos por capa, los epochs, el tamaño del mini-batch, el k del cross validation y el learning rate.

- En la carpeta "Grafos", se almacena el grafo generado por tensorflow para cada k con el nombre grafo-“Nombre del archivo nn”
- En la carpeta "Resultados", se almacena el R cuadrado, el MSE, el MAE y el tiempo de ejecución para cada k, además del promedio y la desviación estándar para cada uno. 
- En la carpeta "Entrenamiento", se almacena el entrenamiento de la red neuronal con el nombre, mod-“Nombre del archivo nn”.
- En la carpeta "y_finales", se almacena el valor de la solución real junto con el valor estimado de la red neuronal.

El nombre de los archivos generados se constituye de la siguiente forma.

```
Nombre del archivo nn: "número de capas"-"nodos por capa”-"función de activación"-"learning rate"-“mini batch”-“epochs”-“k del cross validation”
```

## SOM_2dimensions.py

Se carga el archivo **datos_corregidos_raleo.csv** en un dataframe. Se debe definir las dimensiones del SOM m y n, el número de iteraciones y el learning rate.

- En la carpeta "Resultados_SOM", se almacenan los centroides, el vector de ubicación, la dimensión del SOM en m x n, el quantization error y el topographic error. Se almacena con el nombre resultadosSOM-“Nombre del archivo SOM”.

```
Nombre del archivo SOM: “learning rate”-“número de iteraciones”-“dimensión m”-“dimensión n”.
```

## Graficador_SOM_2dimensions.py
Lee los archivos de resultados entregados por **SOM_2dimensions.py**.

Permite hacer:
- SOM con forma de gráfico hexagonal (panal de abeja) por cada uno de los atributos.
- Gráfico de número de cluster vs SSE, para utilizar el elbow method.
- Gráfico de clusters con el estilo de panal de abejas.


## SOM_3dimensions.py

Se carga el archivo **datos_corregidos_raleo.csv**. Se debe definir la dimensiones m, n y l, el número de iteraciones y el learning rate de cada SOM.

- En la carpeta "Resultados_SOM", se almacenan los centroides, el vector de ubicación, las dimensiones m, n y l, el quantization error y el topographic error. Se almacenan con el nombre resultadosSOM3D-“Nombre del archivo SOM3D”.
```
Nombre del archivo SOM3D: “learning rate”-“número de iteraciones”-“dimensión m”-“dimensión n”-“dimensión l”
```

## Graficador_SOM_3dimensions.py
Lee los archivos de resultados entregados por **SOM_3dimensions.py**.

Permite hacer: 
- SOM con forma de cubo por cada uno de los atributos.
- Gráfico de número de cluster vs SSE, para utilizar el elbow method.
- Gráfico de cubo mostrando los clusters.
- Gráficos de clusters en 2 dimensiones

## SOM_toroidal.py

Se carga el archivo **datos_corregidos_raleo.csv**. Se define las dimensiones m y n, el número de iteraciones y el learning rate de cada SOM toroidal.

- En la carpeta "Resultados_SOM", se almacenan los centroides, el vector de ubicación, las dimensiones m y n, el quantization error y el topographic error. Se almacena con el nombre resultadosSOMToroidal-“Nombre del archivo SOM toroidal”.

```
Nombre del archivo SOM toroidal: “learning rate”-“número de iteraciones”-“dimensión m”-“dimensión n”
```

## Graficador_SOM_toroidal.py
Lee los archivos de resultados entregados por **SOM_toroidal.py**.

Permite hacer:
- SOM de toroide y de panal de abejas por cada uno de los atributos.
- Gráfico de número de cluster vs SSE, para utilizar el elbow method.
- Gráfico de toroide mostrando los clusters.
- Gráficos de clusters en 2 dimensiones 


# Requerimientos

Se utlizó las siguientes librerias ejecutadas en Python 3.7.5.

```
scikit-learn 0.19.2
tensorflow 1.14.0
pandas 1.0.3
matplotlib 3.2.1
numpy 1.16.4
scipy 1.4.1
``` 

## **Hugo Ubilla**