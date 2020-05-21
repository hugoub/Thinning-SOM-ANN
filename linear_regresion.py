#!/usr/bin/env 
# by Hugo Ubilla
# -*- coding: utf-8 -*-

### IMPORT

# Principales
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import time

# Para Graficar
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Utilidades de Sklearn
# Para el Pipeline
from sklearn.pipeline import Pipeline, FeatureUnion

# Para las Transformaciones
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.preprocessing import Imputer # Para asignar valores a distintos datos

# Para el cross validation
from sklearn.model_selection import ShuffleSplit, KFold

# Para minibatch
from sklearn.utils import shuffle

# Para evaluar
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Opciones de Config Pandas y TensorFlow
#tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 30                    # 10 filas maximo
pd.options.display.max_columns = 30                 # 10 columnas maximo
pd.options.display.float_format = '{:.1f}'.format   # Solo un decimal en float

### CARGAR DATOS DF

RUTA_DATASET = "datasets/madera" # Ruta del Workspace

def cargar_datos(path=RUTA_DATASET):
    csv_path = os.path.join(path, "datos_corregidos_cosecha.csv")
    return pd.read_csv(csv_path, sep = ",") # o simplemente poner la ruta completa en esta funcion

cosecha_dataframe = cargar_datos()
cosecha_dataframe2 = cargar_datos()

# Aleatorizar datos mediante una permutación aleatoria

np.random.seed(10) # Seed
cosecha_dataframe = cosecha_dataframe.reindex(
        np.random.permutation(cosecha_dataframe.index))

### Seleccion de atributos ###
cosecha_label =  cosecha_dataframe[["Vol_NOC"]].copy()

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
               "inclinacion"]

cat_attribs_bin= ["EMPRESA","Nombre_Especie","Tipo_eq","season","Emsefor"]


num_pipeline = Pipeline([
        ('selector',DataFrameSelector(num_attribs)),
        ('imputer', Imputer(strategy="median")),
        ('std_scaler', StandardScaler()),
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

### TARGET
cosecha_label = cosecha_label.values
          
# Datos para entrenamiento 
train_features = cosecha_prepared
train_target = cosecha_label

#lear = [0.1, 0.001, 0.0001, 0.00001]
#epoc = [800,1600,3200,6400]
#mini_size = [2048,4096]
#k_cross_val = [5,10]

lear = [0.1]
epoc = [100]
mini_size = [10000]
k_cross_val = [5]

for lea in lear:
    for epo in epoc:
        for mini in mini_size:
            for k_fo in k_cross_val:
                
                # Cross Validation
                kf = KFold(n_splits=k_fo, shuffle = False, random_state=34) 
            
                # Split
                k = 0
                
                # Almacenar valores MSE
                val_mse = []
                val_mae = []
                val_r2 = []
                val_dur_k = []            

                start = time.time()
            
                for train_indices, val_indices in kf.split(train_features):
                    
                    tf.reset_default_graph()   
    
                    X_train, y_train = shuffle(train_features[train_indices],train_target[train_indices], random_state= 2)
                               
                    # tf Graph Input
                    with tf.name_scope('placeholder'):
                        X = tf.placeholder("float64", shape=(None,train_features.shape[1]), name = 'X')
                        y = tf.placeholder("float64", shape=(None,1), name = 'y')
                    
                    # Definicion de tensores, inicializacion (13 columnas y un objetivo)
                    with tf.name_scope('weights'):
                        w = tf.Variable(tf.truncated_normal([1, train_features.shape[1]], mean=0.0, stddev=1.0, dtype=tf.float64), name = 'weight')
                        b = tf.Variable(tf.zeros(1, dtype = tf.float64), name = 'bias')
                    
                    with tf.name_scope("prediction"):
                        y_pred = tf.add(b, tf.matmul(X,tf.transpose(w)), name = 'prediction')
                    
                    # Root mean squared error
                    with tf.name_scope("loss"):
                        #costo = tf.losses.mean_squared_error(y, y_pred)
                        costo = tf.reduce_mean(tf.square(y - y_pred), name = 'loss')
                        #costo = tf.reduce_sum(tf.square(Y - pred, name='loss'))
                    
                    with tf.name_scope("summaries"):
                        tf.summary.histogram("costitos", costo)
                        res_costo_train = tf.summary.scalar("loss", costo)
                        res_costo_valid = tf.summary.scalar("loss_val_set", costo)
                        #merged = tf.summary.merge_all()
                    
                    # Hiperparametros
                    with tf.name_scope('train') as scope:
                        training_epochs = epo
                        learning_rate = lea
                        minibatch_size = mini
                      
                    nombre = 'lr-'+str(lea)+'-'+str(epo)+'-'+str(mini)+'-'+str(k_fo)  
                        
                    with tf.name_scope("optim"):
                        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(costo)
                        #optimizer = tf.train.AdamOptimizer(learning_rate).minimize(costo)
                    saver = tf.train.Saver()
                    
                    # Inicializador de variables
                    init = tf.global_variables_initializer()
                    
                    saver = tf.train.Saver()
                    
                    start_k = time.time()
            
                    # Comenzar entrenamiento
                    with tf.Session() as sess:
                        
                        # Inicializar variables
                        sess.run(init)   
                               
                        #Guardar grafo
                        writer = tf.summary.FileWriter( './Grafos/grafo-'+nombre, sess.graph) # Guardar Grafo
                        
                        # Union de todos los summary
                        #merge = tf.summary.merge_all()
                        merge = tf.summary.merge([res_costo_train])
                        contador = 0

                        for epoch in range(training_epochs):
                            
                            for i in range(0, len(train_indices), minibatch_size):
                                
                                contador += 1 
                                _, summary = sess.run([optimizer, merge], feed_dict={X: X_train[i:i+minibatch_size], y: y_train[i:i+minibatch_size]})
                                #_, loss, summary = sess.run([optimizer, costo, merge], feed_dict={X: X_train[i:i+minibatch_size], y: y_train[i:i+minibatch_size]})
                                writer.add_summary(summary, contador)
                        
                                #costo_valid = sess.run(res_costo_valid, feed_dict={X: train_features[val_indices], y: train_target[val_indices]})
                                #writer.add_summary(costo_valid, contador)
                                
                            if (epoch > 0 and epoch % 20 == 0):
                                #print('%',w.eval()[0][0])train_features
                                #print(' cambio ')
                                costo1 = costo.eval(feed_dict={X: X_train[i:i+minibatch_size], y: y_train[i:i+minibatch_size]})
                                print('division: %i, epoch: %i, costo: %i' %(k, epoch+1, costo1))
                           
                        
                        dur_k = time.time() - start_k
                        
                        k += 1

                        y_T = train_target[val_indices]
                        y_P = sess.run(y_pred, feed_dict={X: train_features[val_indices]})
                            
                        r2 = r2_score(y_T, y_P)
                        mse = mean_squared_error(y_T,y_P) #mean square error
                        mae = mean_absolute_error(y_T,y_P)
                          
                        val_mse.append(np.round(mean_squared_error(y_T,y_P),2))              
                        val_mae.append(np.round(mean_absolute_error(y_T,y_P),2))
                        val_r2.append(np.round(r2_score(y_T,y_P),2))
                        val_dur_k.append(dur_k)
                    
                        print("Aca va el r2: ", r2)
                        print("Aca va el mse: ", mse)
                        print("Aca va el mae: ", mae)
                        print("##################################")
                        
                dur = time.time() - start
                
                datos = open('./Resultados_lr/'+nombre,"w")
                
                datos.write('val_mse = '+str(val_mse)+'\n')
                datos.write('mean_mse = '+str(np.round(np.mean(val_mse),3))+'\n')
                datos.write('std_mse = '+str(np.round(np.std(val_mse),3))+'\n')
    
                datos.write('val_mae = '+str(val_mae)+'\n')
                datos.write('mean_mae = '+str(np.round(np.mean(val_mae),3))+'\n')
                datos.write('std_mae = '+str(np.round(np.std(val_mae),3))+'\n')            
    
                datos.write('val_r2 = '+str(val_r2)+'\n')
                datos.write('mean_r2 = '+str(np.round(np.mean(val_r2),3))+'\n')
                datos.write('std_r2 = '+str(np.round(np.std(val_r2),3))+'\n')                      
                
                datos.write('dur_sep = '+str(val_dur_k)+'\n')
                datos.write('dur_total = '+str(np.round(dur,2))+'\n')
                datos.write('dur_total2 = '+str(np.round(sum(val_dur_k),3))+'\n')
                
                datos.close()
                
                print('Validation mse ',val_mse)
                print('Validation mae ',val_mae)
                print('Validation r2 ',val_r2)

