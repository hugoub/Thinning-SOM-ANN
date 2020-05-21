#!/usr/bin/env 
# Hugo Ubilla
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 14:13:59 2018

@author: hugoub
"""
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import time
 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import Imputer # Para asignar valores a distintos datos
from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors, cm
from matplotlib.collections import RegularPolyCollection

class SOM(object):
    """
    2-D Self-Organizing Map with Gaussian Neighbourhood function
    and linearly decreasing learning rate.
    """
 
    #To check if the SOM has been trained
    _trained = False
 
    def __init__(self, m, n, dim, n_iterations=100, alpha=None, sigma=None):
        """
        Initializes all necessary components of the TensorFlow
        Graph.
 
        m X n are the dimensions of the SOM. 'n_iterations' should
        should be an integer denoting the number of iterations undergone
        while training.
        'dim' is the dimensionality of the training inputs.
        'alpha' is a number denoting the initial time(iteration no)-based
        learning rate. Default value is 0.3
        'sigma' is the the initial neighbourhood value, denoting
        the radius of influence of the BMU while training. By default, its
        taken to be half of max(m, n).
        """
 
        #Assign required variables first
        self._m = m
        self._n = n
        if alpha is None:
            alpha = 0.3
        else:
            alpha = float(alpha)
        if sigma is None:
            sigma = max(m, n) / 2.0
        else:
            sigma = float(sigma)
        self._n_iterations = abs(int(n_iterations))
 
        ##INITIALIZE GRAPH
        self._graph = tf.Graph()
 
        ##POPULATE GRAPH WITH NECESSARY COMPONENTS
        with self._graph.as_default():
 
            ##VARIABLES AND CONSTANT OPS FOR DATA STORAGE
 
            #Randomly initialized weightage vectors for all neurons,
            #stored together as a matrix Variable of size [m*n, dim]
            
            ## Vector de pesos
            self._weightage_vects = tf.Variable(tf.random_normal([m*n, dim]))
 
            #Matrix of size [m*n, 2] for SOM grid locations
            #of neurons
            
            ## Vector de localizacion de nodos
            self._location_vects = tf.constant(np.array(
                list(self._neuron_locations(m, n))))
 
            ##PLACEHOLDERS FOR TRAINING INPUTS
            #We need to assign them as attributes to self, since they
            #will be fed in during training
 
            #The training vector
            
            ## Vector de input
            self._vect_input = tf.placeholder("float", [dim])
            #Iteration number
 
           
            ## Iteracion 
            self._iter_input = tf.placeholder("float")
 
            matriz = np.array(list(self._neuron_locations(m,n)))
            
            matrices = []
            
            matrices = np.stack((matriz,
                                 matriz + np.array([m,n]), 
                                 matriz - np.array([m,n]), 
                                 matriz + np.array([m,0]),
                                 matriz - np.array([m,0]),
                                 matriz + np.array([0,n]),
                                 matriz - np.array([0,n]),
                                 matriz + np.array([m,-n]),
                                 matriz + np.array([-m,n])
                                 ))

            distancias_matriz = []
            
            for i in range(n*m):
                distancias_matriz.append([])
                for j in range(m*n):
                    distancias_matriz[i].append(np.min(np.sum(np.power(np.subtract(matriz[i], matrices[:,j]),2), axis = 1)))
            
            distancias_matriz = tf.constant(np.array(distancias_matriz))

            
            ##CONSTRUCT TRAINING OP PIECE BY PIECE
            #Only the final, 'root' training op needs to be assigned as
            #an attribute to self, since all the rest will be executed
            #automatically during training
 
            #To compute the Best Matching Unit given a vector
            #Basically calculates the Euclidean distance between every
            #neuron's weightage vector and the input, and returns the
            #index of the neuron which gives the least value
            
            ## Indice del BMU (no es necesario crear arreglo de vectores)
            bmu_index = tf.argmin(tf.sqrt(tf.reduce_sum(
                tf.pow(tf.subtract(self._weightage_vects, self._vect_input), 2), 1)), 0)
   
            #This will extract the location of the BMU based on the BMU's
            #index
            
            slice_input = tf.pad(tf.reshape(bmu_index, [1]),
                                 np.array([[0, 1]]))
            
            ## Entrega la localizacion en x e y
            bmu_loc = tf.reshape(tf.slice(self._location_vects, slice_input,
                                          tf.constant(np.array([1, 2]),"int64")),
                                 [2])

            #To compute the alpha and sigma values based on iteration
            #number
            learning_rate_op = tf.subtract(1.0, tf.div(self._iter_input,
                                                  self._n_iterations))
            
            # Creado por mi            
            #lear_rate = tf.exp(tf.negative(tf.div(self._iter_input, self._n_iterations)))
            
            _alpha_op = tf.multiply(alpha, learning_rate_op)
            _sigma_op = tf.multiply(sigma, learning_rate_op)
            
            #Construct the op that will generate a vector with learning
            #rates for all neurons, based on iteration number and location
            #wrt BMU.          
            
            #bmu_distance_squares = tf.reduce_sum(tf.pow(tf.subtract(
            #        self._location_vects, bmu_loc), 2), 1)
            
            bmu_distance_squares = distancias_matriz[bmu_index]
                                
            # Crea un arreglo de m*n para cada funcion dependiendo de la dist al BMU            
            neighbourhood_func = tf.exp(tf.negative(tf.div(tf.cast(
                bmu_distance_squares, "float32"), tf.multiply(tf.pow(_sigma_op, 2.), 2.))))

            print(neighbourhood_func)
            
            learning_rate_op = tf.multiply(_alpha_op, neighbourhood_func)
            
            print(learning_rate_op)
            
            #Finally, the op that will use learning_rate_op to update
            #the weightage vectors of all neurons based on a particular
            #input
            
#            learning_rate_multiplier = [tf.tile(tf.slice(
#                learning_rate_op, np.array([i]), np.array([1])), [dim]) for i in range(m*n)]
#            
            learning_rate_op_2 = tf.reshape(learning_rate_op, (m*n,1))

            #print(learning_rate_multiplier)
            
            weightage_delta = tf.multiply(
                learning_rate_op_2,
                tf.subtract(self._vect_input,
                       self._weightage_vects))                                         
            
            new_weightages_op = tf.add(self._weightage_vects,
                                       weightage_delta)
            self._training_op = tf.assign(self._weightage_vects,
                                          new_weightages_op)                                       
 
            ##INITIALIZE SESSION
            self._sess = tf.Session()
 
            ##INITIALIZE VARIABLES
            init_op = tf.global_variables_initializer()
            self._sess.run(init_op)
 
    def _neuron_locations(self, m, n):
        """
        Yields one by one the 2-D locations of the individual neurons
        in the SOM.
        """
        #Nested iterations over both dimensions
        #to generate all 2-D locations in the map
        for i in range(m):
            for j in range(n):
                yield np.array([i, j])
 
    def train(self, input_vects):
        """
        Trains the SOM.
        'input_vects' should be an iterable of 1-D NumPy arrays with
        dimensionality as provided during initialization of this SOM.
        Current weightage vectors for all neurons(initially random) are
        taken as starting conditions for training.
        """
 
        #Training iterations
        for iter_no in range(self._n_iterations):
            #Train with each vector one by one
            if iter_no % 20 == 0:
                print(iter_no)
            for input_vect in input_vects:
                self._sess.run(self._training_op,
                               feed_dict={self._vect_input: input_vect,
                                          self._iter_input: iter_no})
 
        #Store a centroid grid for easy retrieval later on
        centroid_grid = [[] for i in range(self._m)]
        self._weightages = list(self._sess.run(self._weightage_vects))
        self._locations = list(self._sess.run(self._location_vects))
        for i, loc in enumerate(self._locations):
            centroid_grid[loc[0]].append(self._weightages[i])
        self._centroid_grid = centroid_grid
 
        self._trained = True
 
    def get_centroids(self):
        """
        Returns a list of 'm' lists, with each inner list containing
        the 'n' corresponding centroid locations as 1-D NumPy arrays.
        """
        if not self._trained:
            raise ValueError("SOM not trained yet")
        return self._centroid_grid
 
    def map_vects(self, input_vects):
        """
        Maps each input vector to the relevant neuron in the SOM
        grid.
        'input_vects' should be an iterable of 1-D NumPy arrays with
        dimensionality as provided during initialization of this SOM.
        Returns a list of 1-D NumPy arrays containing (row, column)
        info for each input vector(in the same order), corresponding
        to mapped neuron.
        """
 
        if not self._trained:
            raise ValueError("SOM not trained yet")
 
        to_return = []
        
        distances = []
        
        
        contador_adyacentes = 0
        
        matriz = np.array(list(self._neuron_locations(self._m, self._n)))
        
        m = self._m
        
        n = self._n
        
        matrices = []
        
        matrices = np.stack((matriz,
                             matriz + np.array([m,n]), 
                             matriz - np.array([m,n]), 
                             matriz + np.array([m,0]),
                             matriz - np.array([m,0]),
                             matriz + np.array([0,n]),
                             matriz - np.array([0,n]),
                             matriz + np.array([m,-n]),
                             matriz + np.array([-m,n])
                             ))
                
        distancias_matriz = []

        for i in range(n*m):
            distancias_matriz.append([])
            for j in range(m*n):
                distancias_matriz[i].append(np.min(np.sum(np.power(np.subtract(matriz[i], matrices[:,j]),2), axis = 1)))
        
        distancias_matriz = np.array(distancias_matriz)
        
            
        for vect in input_vects:

            # min_index is the index of the BMU
            
            lista_indices = [i for i in range(len(self._weightages))]
            
            min_index = min(lista_indices,
                            key=lambda x: np.linalg.norm(vect - self._weightages[x]))

            # min_index_2 is the index of the 2nd BMU
            
            lista_indices.pop(min_index) # El indice es el mismo que el valor
            
            min_index_2 = min(lista_indices,
                            key=lambda x: np.linalg.norm(vect - self._weightages[x]))      
            
            r2 = np.sqrt(2)

            if np.sqrt(distancias_matriz[min_index][min_index_2]) > r2:    
#                print('loc 1')
#                print(locaciones[min_index])
#                print('loc 2')
#                print(locaciones[min_index_2])
                contador_adyacentes += 1


            distance = np.linalg.norm(vect - self._weightages[min_index])
            
            distances.append(distance)
            
            to_return.append(self._locations[min_index])            
        
        # Quantization Error qe (the mean of all distances to the BMU)!
        self.distances = distances  
        
        # Topographic error te
        self.proporcion = contador_adyacentes / len(input_vects)
        
        self.prom_dist = np.mean(self.distances)
        
        return to_return


'''
#For plotting the images
from matplotlib import pyplot as plt
 
#Training inputs for RGBcolors
colors = np.array(
     [[0., 0., 0.],
      [0., 0., 1.],
      [0., 0., 0.5],
      [0.125, 0.529, 1.0],
      [0.33, 0.4, 0.67],
      [0.6, 0.5, 1.0],
      [0., 1., 0.],
      [1., 0., 0.],
      [0., 1., 1.],
      [1., 0., 1.],
      [1., 1., 0.],
      [1., 1., 1.],
      [.33, .33, .33],
      [.5, .5, .5],
      [.66, .66, .66]])
color_names = \
    ['black', 'blue', 'darkblue', 'skyblue',
     'greyblue', 'lilac', 'green', 'red',
     'cyan', 'violet', 'yellow', 'white',
     'darkgrey', 'mediumgrey', 'lightgrey']
 
#Train a 20x30 SOM with 400 iterations
som = SOM(50, 50, 3, 400)

start = time.time()
som.train(colors)
dur = time.time() - start 

print(dur)

#Get output grid
image_grid = som.get_centroids()
 
#Map colours to their closest neurons
mapped = som.map_vects(colors)
 
#Plot
plt.imshow(image_grid)
plt.title('Color SOM')
for i, m in enumerate(mapped):
    plt.text(m[1], m[0], color_names[i], ha='center', va='center',
             bbox=dict(facecolor='white', alpha=0.5, lw=0))
plt.show()
'''

### CARGAR DATOS DF

RUTA_DATASET = "datasets/madera" # Ruta del Workspace

def cargar_datos(path=RUTA_DATASET):
    csv_path = os.path.join(path, "datos_corregidos_raleo.csv")
    return pd.read_csv(csv_path, sep = ",") # o simplemente poner la ruta completa en esta funcion

cosecha_dataframe = cargar_datos()

s_set = 1

# Aleatorizar datos mediante una permutación aleatoria

np.random.seed(10) # Seed
cosecha_dataframe = cosecha_dataframe.reindex(
        np.random.permutation(cosecha_dataframe.index))

### TRANSFORMACIONES

# Funciones de Transformaciones

class DataFrameSelector(BaseEstimator, TransformerMixin):
    '''
    Entrada: Recibe todo el nombre de los atributos de dataframe que se
    utilizaran.
    Fit: Todo el DataFrame.
    Transform: Retorna un DF solo con las columnas seleccionadas.
    '''
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names] #Lo dejamos como DataFrame


class LabelBinarizerForMultiplesFeatures(BaseEstimator, TransformerMixin):
    '''
    Fit: DF con columnas de tipo categoricas
    Transform: Matriz con la transformacion de cada columna de entrada en 
    columnas de 1 y 0. El numero de columnas de salida por columna de entrada
    sera (numero de categorias por columna de entrada - 1)
    '''
    def __init__(self, sparse_output=False):
        self.sparse_output = sparse_output
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        enc = LabelBinarizer(sparse_output=self.sparse_output)
        transformacion = np.empty(shape=(X.shape[0],0))
        for column in X.columns:
            resto = enc.fit_transform(X[column])
            if resto.shape[1] == 1:
                transformacion = np.append(transformacion, resto, axis=1)
            else:
                transformacion = np.append(transformacion, resto[:,:-1], axis=1)
        return transformacion 
    
class LabelEncoderForMultiplesFeatures(BaseEstimator, TransformerMixin):
    '''
    Fit: Recibe DF de columnas categoricas
    Transform: Retorna una columna de datos numericos con numeros enteros 
    '''    
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        enc = LabelEncoder()
        transformacion = np.empty(shape=(X.shape[0],0))
        for column in X.columns:
            resto = enc.fit_transform(X[column])
            resto = resto.reshape(resto.shape[0],1)
            transformacion = np.append(transformacion, resto, axis=1)
        return transformacion
    

# SELECCION DE ATRIBUTOS DEL MODELO
       
# Si se hace un list de un dataframe, el resultado sera sus columnas
    
'''
num_attribs = ["ano","Mes","Cod_Sip","cod_emsefor","Dens","M3_Arb","M3_HA",
               "T_EFEC","T_PERD","dias_trab","N__Trab","H_H_efectivas",
               "Volumen_H_H","Superficie_Total","Coor_X","Coor_Y"]

cat_attribs = ["EMPRESA","Nombre_Especie","estacion","Unidad","Tipo_eq",
               "Sigla","Fecha","Predio","Emsefor"]
'''


num_attribs = ["Dens","M3_Arb","M3_HA","N__Trab","T_EFEC","Coor_X","Coor_Y",
               "inclinacion","Vol_NOC"]  # Debe llevar VOL_NOC

cat_attribs_bin= ["Nombre_Especie","Tipo_eq","season","Emsefor"]

num_pipeline = Pipeline([
        ('selector',DataFrameSelector(num_attribs)),
        ('imputer', Imputer(strategy="median")),
        #('minmaxscaler', MinMaxScaler(feature_range=(-1, 1))),
        ('scaler', StandardScaler()),
        ])

cat_pipeline_bin = Pipeline([
        ('selector', DataFrameSelector(cat_attribs_bin)),
        ('label_binarizer', LabelBinarizerForMultiplesFeatures()),
        ])

full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline_bin", cat_pipeline_bin),
        ])


### FEATURES
cosecha_prepared = full_pipeline.fit_transform(cosecha_dataframe)

cosecha_datos = cosecha_dataframe[num_attribs].to_numpy()
cosecha_nombres = list(cosecha_dataframe[num_attribs].columns.values)

#
#RUTA_DATASET = "~/ml_paper/datasets/raleo_profe" # Ruta del Workspace
#
#def cargar_datos(path=RUTA_DATASET):
#    csv_path = os.path.join(path, "cosecha_r.csv")
#    return pd.read_csv(csv_path, sep = ",") # o simplemente poner la ruta completa en esta funcion
#
#cosecha_dataframe2 = cargar_datos()
#
#cosecha_prepared2 = cosecha_dataframe2[['V1','V2','V3','V4','V5']].values
#
#cosecha_prepared = cosecha_prepared2

#cosecha_prepared = num_pipeline.fit_transform(cosecha_dataframe)
#
#m1 = 9 # filas
#n1 = 9 # 10 nodos por fila
#l_rate = 0.5
#n_iter = 30

#m1 = 20 # filas
#n1 = 20 # 10 nodos por fila
#n_iter = 500

dimensiones = [[8,8]]
num_iter = [100]
l_rate = 0.05

for dim_m, dim_n in dimensiones:
    
    for iteracion in num_iter:
                
        som = SOM(dim_m, dim_n, cosecha_prepared.shape[1], n_iterations = iteracion, alpha = l_rate)

        start = time.time()
        
        som.train(cosecha_prepared)
        #num_attribs = ["Dens","M3_Arb","M3_HA","N__Trab","T_EFEC","Coor_X","Coor_Y","inclinacion","Vol_NOC"]  # Debe llevar VOL_NOC

        duracion = time.time() - start

        #Obtener centroides
        centroides = som.get_centroids()
         
        #obtener vectores
        mapped = som.map_vects(cosecha_prepared)
        
        promedio_distancias = np.mean(som.distances)
        
        if s_set == 2:
            datos = open('./resultadosSOM-'+str(l_rate)+'-'+str(+iteracion)+'-'+str(dim_m)+'x'+str(dim_n)+'Set2',"w")
        else:
            datos = open('./Resultados_SOM/resultadosSOMToroidal-'+str(l_rate)[2:]+'-'+str(+iteracion)+'-'+str(dim_m)+'x'+str(dim_n),"w")
        
        
        datos.write('centroides = '+str(centroides)+'\n')
        datos.write('mapped = '+str(mapped)+'\n')
        datos.write('m1 = '+str(dim_m)+'\n')
        datos.write('n1 = '+str(dim_n)+'\n')
        datos.write('qe = '+str(som.prom_dist)+'\n')
        datos.write('te = '+str(som.proporcion)+'\n')
        datos.write('dur = '+str(duracion)+'\n')
        datos.close()
