#!/usr/bin/env 
# by Hugo Ubilla
# -*- coding: utf-8 -*

# Utility imports
import numpy as np
import os
# Data Managing imports
import pandas as pd
import time

# Para Graficar
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Machine Learning imports
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import StandardScaler, OneHotEncoder
# INGRESAR DATA FRAME

tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
#tf.logging.set_verbosity(tf.logging.ERROR)

from sklearn.utils import shuffle

#Para dividir
from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

### Para el Pipeline ###
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion

### Para las transformaciones ###
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.preprocessing import Imputer # Para asignar valores a distintos datos

# Opciones para Pandas
pd.options.display.max_rows = 30                    # 10 filas maximo
pd.options.display.max_columns = 30                 # 10 columnas maximo
pd.options.display.float_format = '{:.1f}'.format   # Solo un decimal en float

### Cargar conjunto de Datos como DataFrame ###

# Inicializacion de los pesos
def _weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

# Inicializacion del bias por capa
def _bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

# Funcion de activacion (relu, tanh, sigmoid)
def _activation(z_val, act_name ='ReLU'):
    if act_name == "ReLU":
        return tf.nn.relu(z_val, name='act_fun_'+act_name)
    elif act_name == "Tanh":
        return tf.nn.tanh(z_val, name='act_fun_'+act_name)
    elif act_name == "LReLU":
        return tf.nn.leaky_relu(z_val, name='act_fun_'+act_name)
    else:
        return tf.nn.sigmoid(z_val, name='act_fun_'+act_name)

# Funcion de variables
def _variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)              # Promedio de variable
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))     # Desv estandar
        tf.summary.scalar('stddev', stddev)                             # max
        tf.summary.scalar('max', tf.reduce_max(var))                    # min
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


# Clase para crear read neuronal
class Custom_Nnet(object): 
    def __init__(self, layer_sizes, activations, learning_rate=0.0001, dropout=1, epochs=100, mini_batch=1024, kf = 10, k = 0):
        self.layer_sizes = layer_sizes              # Arreglo de [10,10], neuronas por capa
        self.activations = activations              # Funcion de activacion por capa, lista ['Relu', 'Relu']
        self.num_hidden = len(layer_sizes)          # Numero de hidden layer 2
        self.learning_rate = learning_rate          # Paso
        self.dropout = dropout                      # Desconetar porcentaje de neuronas
        self.epochs = epochs                        # Repeticiones
        self.mini_batch  = mini_batch               # Tamano del minibatch
        self.num_nodos = layer_sizes[0]             # Numero de nodos por capa
        self.act_func = activations[0]
        self.kf = kf
        self.k = k
        print('num_layer: %i, num_nodos: %i, act fn: %s, learning_rates %f, batch_size: %i, epochs: %i, kf = %i, cross val = %i'%(self.num_hidden,self.num_nodos,self.act_func,self.learning_rate,self.mini_batch,self.epochs,self.kf,self.k))
        self.nombre = str(self.num_hidden)+'-'+str(self.num_nodos)+'-'+str(self.act_func)+'-'+str(self.learning_rate)+'-'+str(self.mini_batch)+'-'+str(self.epochs)+'-'+str(self.kf)+'-'+str(self.k)
        print(self.nombre)
        
    def _nn_layer(self, in_vals, out_size, layer_name, activation):
        # output: es toda la X, 
        # size: es el tamaño por capa dentro del for
        # layer_name: nombre de capa
        # funcion de activacion de la capa
        
        with tf.name_scope('Layer_'+layer_name):
            in_size = in_vals.shape.as_list()[1]                            # in_size: Numero de features
            w_fc = _weight_variable([in_size, out_size], 'weight_input')    #  fn [num_features, numero de neuronas por capa] [10,10]
            _variable_summaries(w_fc)                                       # llamada a fn, Se entrega a la funcion de arriba
            b_fc = _bias_variable([out_size], 'bias_input')  # biases       # fn [bias por capa]
            if layer_name == 'Output':                                      # si la capa es de salida
                self.Y_hat = tf.matmul(in_vals, w_fc)+b_fc                  # se retorno solo la prediccion 
                return self.Y_hat
            elif layer_name == 'Hidden_1':                                  # Si la capa es la primera oculta
                z = tf.matmul(in_vals, w_fc) + b_fc                         # se calcula z
                h_fc1 = _activation(z, act_name = activation)               # se calcula la fn de activacion
                tf.summary.histogram('activations_input', h_fc1)            # se agrega al resumen
                return h_fc1
            else:
                z = tf.matmul(in_vals, w_fc) + b_fc                         # Si la capa es ocula pero no la primera
                h_fc1 = _activation(z, act_name = activation)               # se calcula la fn de activation
                h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob, name='Dropout_'+ layer_name)
                return h_fc1_drop

    def _build_network(self,x_width):       # PASO 2, se realiza por cantidad de features
        ### Placeholders ###
        with tf.name_scope('placeholder'):   # Creacion de placeholder (x, y y keep_prob)
            self.y = tf.placeholder(tf.float32, [None, 1], name='y')
            self.X = tf.placeholder(tf.float32, [None, x_width], name='X') # Placeholder values
            self.keep_prob = tf.placeholder("float", name='keep_prob') # Placeholder values

        output = self.X
        output_size = 1  # Solo es un valor el output
        
        for i, size in enumerate(self.layer_sizes + [output_size]):     # Enumera cada una de las capas 
            if i == self.num_hidden:                                    # ultima capa es el output
                layer_name = 'Output'                                   
            else:
                layer_name = 'Hidden_'+str(i+1)                         # darle un nombre a la capa
                activation = self.activations[i]                        # funcion de activacion de la capa
            output = self._nn_layer(output , size, layer_name, activation)  # El output de la primera capa es el nuevo output
            # output: es toda la X, 
            # size: es el tamaño por capa dentro del for
            # layer_name: nombre de capa
            # funcion de activacion de la capa
            

    def _loss(self):
        with tf.name_scope("cost_function"):
            loss = tf.reduce_mean(tf.square(self.y-self.Y_hat, name='loss'))
            l1_regularizer = tf.contrib.layers.l1_regularizer(scale=0.005, scope=None)
            weights = tf.trainable_variables()
            regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, weights)
            regularized_loss = loss + regularization_penalty
            #tf.summary.scalar('cost_function_summary', loss)
            return regularized_loss
    
    def _optimizer(self):
        with tf.name_scope("train"):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self._loss())
    
    def build_graph(self,x_with):
        with tf.Graph().as_default() as g:  
            self._build_network(x_width)
            self._optimizer()
            ### Write Summary Out ###
            self.merged = tf.summary.merge_all()
            self.g = g
            self.saver = tf.train.Saver()

    
    def train(self,X,y,X_val=[],y_val=[]):
        with tf.Session(graph=self.g) as sess:   
            
            # Tamano del minibatch
            minibatch_size = self.mini_batch      
            
            # Para ir calculando perdidas
            y_fake = np.ones((X.shape[0],1))            
            
            # Inicializador
            sess.run(tf.global_variables_initializer())           
            
            # Guardar Grafo
            summary_writer = tf.summary.FileWriter('./Grafos/grafo-'+self.nombre, graph=sess.graph)          
            
            # Arreglo para hacer graficos
            mse_train = []
            mse_valid = []
            
            for epoch in range(self.epochs):
                                                  
                for i in range(0, X_train.shape[0], minibatch_size):       
                        
                    # Si se grafica perdidas, se debe quitar el # a la siguiente sentencia y comentar la subsiguiente.
                    #_, loss = sess.run([self.optimizer, self._loss()], feed_dict={self.X: X_train[i:i+minibatch_size], self.y: y_train[i:i+minibatch_size], self.keep_prob: self.dropout}) # run GD
                    sess.run(self.optimizer, feed_dict={self.X: X_train[i:i+minibatch_size], self.y: y_train[i:i+minibatch_size], self.keep_prob: self.dropout}) # run GD

                if epoch % 10 == 0:
                    print('K del XV: %i Epoch: %i' %(k, epoch+1))

                    # Si se grafica perdidas, descomentar las siguientes 2 sentencias.
                    #mse_train.append(loss)
                    #mse_valid.append(sess.run(self._loss(), feed_dict={self.X: X_val, self.y: y_val, self.keep_prob: self.dropout}))
                
                    # Si se quiere guardar resumen, descomentar.
                    #summary_str = sess.run(self.merged, feed_dict={self.X: X_train, self.y:y_train, self.keep_prob: 1})
                    #summary_writer.add_summary(summary_str, epoch)
                                   
                # Run the code below in the console
                #tensorboard --logdir=C:/Users/rodel/Documents/Machine_Learning/graphs/n_net --host=127.0.0.1
                #tensorboard --inspect --logdir C:/Users/rodel/Documents/Machine_Learning/graphs/n_net
            
             # Si se grafica perdidas, descomentar
#            fig, ax = plt.subplots(figsize=(6,4))
#            plt.ylabel('MSE')
#            plt.xlabel('Epochs')
#            plt.title("Mean Squared Error vs. Epochs")
#            ax.plot(mse_train, label='Error en Training Set' , linewidth=1)
#            ax.plot(mse_valid, label='Error en Validation Set', linewidth=1)
#            plt.legend()
#            fig.savefig('./Graficos/'+self.nombre+'.png')
                 
            self.saver.save(sess, './Entrenamiento/mod-'+self.nombre)  # Guardar entrenamiento         
            summary_writer.close()

            print("Optimization Finished!")
             
    def predict(self,X):
        y_fake = np.ones((X.shape[0],1))
        with tf.Session(graph=self.g) as sess:
            self.saver.restore(sess, './Entrenamiento/mod-'+self.nombre)            
            # Evaluate the predicted values
            yP = sess.run(self.Y_hat, feed_dict={self.X: X, self.y: y_fake, self.keep_prob:1})
            return yP
                
        
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
#cosecha_prepared = num_pipeline.fit_transform(cosecha_dataframe)

### TARGET
cosecha_label = cosecha_label.values
          
# Datos para entrenamiento 
train_features = cosecha_prepared
train_target = cosecha_label

#### GRID
'''
act_function = ['ReLU','LReLU','Tanh','Sigmoid']
capas = [1,2,4,6]
nodos = [25,50,100,200]
epochs =[1000,2000,4000]
mini_batch = [2056,4096]
k_cross_val = [2]
learning_rates = [0.001, 0.0001, 0.00001, 0.000001]
'''

# Datos de instancia a correr

act_function = ['Sigmoid']
capas = [2]
nodos = [50]
epochs =[100]
mini_batch = [4096]
k_cross_val = [5]
learning_rates = [0.001]

for cap in capas:
    for nod in nodos:
        for f_ac in act_function:
            for epo in epochs:
                for bat in mini_batch:
                    for lea in learning_rates:
                        for k_fo in k_cross_val:
                            
                            # Cross Validation
                            kf = KFold(n_splits=k_fo, shuffle = False, random_state=34) 
                            
                            # Split
                            k = 0
                            
                            # Almacenar valores MSE
                            val_mse = []
                            val_mae = []
                            val_r2 = []
                            val_dur = []
                    
                            start = time.time()
                            
                            nombre = str(cap)+'-'+str(nod)+'-'+str(f_ac)+'-'+str(lea)+'-'+str(bat)+'-'+str(epo)+'-'+str(k_fo)
                    
                            for train_indices, val_indices in kf.split(train_features):
                                
                                tf.reset_default_graph()   
                
                                X_train, y_train = shuffle(train_features[train_indices],train_target[train_indices], random_state= 2)
                            
                                hidden = [nod]*cap # Num Hidden
                                
                                act = [f_ac for _ in range(len(hidden))] #act fction
                                
                                tf.reset_default_graph()                     
                                
                                nn=Custom_Nnet(hidden, act, learning_rate = lea, epochs = epo, mini_batch = bat, kf = k_fo, k = k)
                                
                                x_width = train_features.shape[1]
                                
                                nn.build_graph(x_width)
                                                               
                                start1 = time.time()
                                
                                nn.train(X_train,y_train)
                                #nn.train(X_train,y_train,train_features[val_indices],train_target[val_indices])
                                
                                dur1 = time.time() - start1

                                ### Desempeno
                                
                                y2 = train_target[val_indices]
                                
                                Y_hat=nn.predict(train_features[val_indices])
                            
                                mse = mean_squared_error(y2,Y_hat)
                                mae = mean_absolute_error(y2,Y_hat)
                                r2 = r2_score(y2,Y_hat)                           
                                
                                val_mse.append(mse)
                                val_mae.append(mae)
                                val_r2.append(r2)
                                val_dur.append(dur1)
                                
                                print("Aca va el MSE: ", mse)
                                print("Aca va el MAE: ", mae)
                                print("Aca va el r2: ", r2)
                                
                                y_final = nn.predict(cosecha_prepared)
                                datos = open('./y_finales/'+nombre+'--'+str(k),"w")
                                datos.write('y_real = '+str(list(cosecha_label.reshape(68005)))+'\n')
                                datos.write('y_final = '+str(list(y_final.reshape(68005)))+'\n')
                                datos.close()                        
                                
                                k += 1

                            dur = time.time() - start
                            
                            datos = open('./Resultados/'+nombre,"w")
                            
                            datos.write('val_mse = '+str(val_mse)+'\n')
                            datos.write('mean_mse = '+str(np.round(np.mean(val_mse),2))+'\n')
                            datos.write('std_mse = '+str(np.round(np.std(val_mse),2))+'\n')
                
                            datos.write('val_mae = '+str(val_mae)+'\n')
                            datos.write('mean_mae = '+str(np.round(np.mean(val_mae),2))+'\n')
                            datos.write('std_mae = '+str(np.round(np.std(val_mae),2))+'\n')            
                
                            datos.write('val_r2 = '+str(val_r2)+'\n')
                            datos.write('mean_r2 = '+str(np.round(np.mean(val_r2),2))+'\n')
                            datos.write('std_r2 = '+str(np.round(np.std(val_r2),2))+'\n')                      
                            
                            datos.write('dur_sep = '+str(np.round(val_dur,2))+'\n')
                            datos.write('dur_total = '+str(np.round(dur,2))+'\n')
                            datos.write('dur_total2 = '+str(np.round(sum(val_dur),2))+'\n')
                            
                            datos.close()
                            
                            print('Validation mse ',val_mse)
                            print('Validation mae ',val_mae)
                            print('Validation r2 ',val_r2)
                            
