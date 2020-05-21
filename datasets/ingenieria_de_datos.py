#!/usr/bin/env 
# by Hugo Ubilla
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import tensorflow as tf
import os
from IPython import display

#Para dividir
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_squared_error, r2_score

### Para el Pipeline ###
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion

### Para las transformaciones ###
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import Imputer # Para asignar valores a distintos datos
from sklearn.preprocessing import LabelEncoder

from sklearn.decomposition import PCA

# Opciones para Pandas
pd.options.display.max_rows = 30                    # 10 filas maximo
pd.options.display.max_columns = 30                 # 10 columnas maximo
pd.options.display.float_format = '{:.1f}'.format   # Solo un decimal en float

from pandas.tools.plotting import scatter_matrix

### Cargar conjunto de Datos como DataFrame ###

RUTA_DATASET = "~/ml/datasets/madera" # Ruta del Workspace

def cargar_datos(path=RUTA_DATASET):
    csv_path = os.path.join(path, "Vnoc.csv")
    return pd.read_csv(csv_path, sep = ",") # o simplemente poner la ruta completa en esta funcion

cosecha_dataframe = cargar_datos()

### Aleatorizar datoa mediante una permutación aleatoria ###

### Cambiar nombre a una columna ###

#cosecha_dataframe.info()
new_columns = cosecha_dataframe.columns.values; 
new_columns[23] = 'Superficie_Total'; 
cosecha_dataframe.columns = new_columns 

#ordenado1 = cosecha_dataframe.sort_values(by='ano', ascending=False)
#ordenado2 = cosecha_dataframe.sort_values(by='Mes', ascending=True)
#ordenado3 = cosecha_dataframe.sort_values(by='Fecha', ascending=True)


# Dar Formato a fecha y ordenar datos por fecha
cosecha_dataframe['Fecha'] = pd.to_datetime(cosecha_dataframe['Fecha'])
cosecha_dataframe['dia'] = cosecha_dataframe['Fecha'].apply(lambda x: x.day)

# Estaciones (dia y mes son int)

# otono
cosecha_dataframe['otono']  = (((cosecha_dataframe['dia'] >= 21) & (cosecha_dataframe['Mes'] == 3)) | ((cosecha_dataframe['Mes'] > 3) & (cosecha_dataframe['Mes'] < 6)) | ((cosecha_dataframe['dia'] <= 20) & (cosecha_dataframe['Mes'] == 6)) ).apply(lambda x: 1 if x else 0 )
# invierno
cosecha_dataframe['invierno']  = (((cosecha_dataframe['dia'] >= 21) & (cosecha_dataframe['Mes'] == 6)) | ((cosecha_dataframe['Mes'] > 6) & (cosecha_dataframe['Mes'] < 9)) | ((cosecha_dataframe['dia'] <= 22) & (cosecha_dataframe['Mes'] == 9)) ).apply(lambda x: 1 if x else 0)
# primavera
cosecha_dataframe['primavera']  = (((cosecha_dataframe['dia'] >= 23) & (cosecha_dataframe['Mes'] == 9)) | ((cosecha_dataframe['Mes'] > 9) & (cosecha_dataframe['Mes'] < 12)) | ((cosecha_dataframe['dia'] <= 21) & (cosecha_dataframe['Mes'] == 12)) ).apply(lambda x: 1 if x else 0)
# verano
cosecha_dataframe['verano']  = (((cosecha_dataframe['dia'] >= 22) & (cosecha_dataframe['Mes'] == 12)) | (cosecha_dataframe['Mes'] < 3) | ((cosecha_dataframe['dia'] <= 20) & (cosecha_dataframe['Mes'] == 3)) ).apply(lambda x: 1 if x else 0)

season = []
for i in cosecha_dataframe.index:
    if cosecha_dataframe['otono'].iloc[i] == 1:     
        season.append('otono')
    if cosecha_dataframe['invierno'].iloc[i] == 1:
        season.append('invierno')     
    if cosecha_dataframe['primavera'].iloc[i] == 1:
        season.append('primavera')     
    if cosecha_dataframe['verano'].iloc[i] == 1:
        season.append('verano')


season = np.array(season)

cosecha_dataframe['season'] = pd.Series(season)

#cosecha_dataframe[['Mes','dia','season']]

# Pts Atipicos
pts_atipicos = [32227, 32354, 32355, 48794, 48795, 20133, 20136, 20137, 20134, 
                20135, 20138, 20139, 20140, 20141, 20142, 20143, 24861, 24862, 
                24863, 24864, 24865, 32228, 32229, 32230, 32231, 32233, 32356, 
                32358, 32359, 32360, 7643, 7645, 7646, 7647, 7648, 33523, 7474,
                7475]

# Eliminar ptos atipicos
cosecha_dataframe = cosecha_dataframe.drop(pts_atipicos)

cosecha_dataframe.to_csv('./datasets/madera/datos_corregidos.csv', sep=',', index=False)

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

num_attribs = ["ano","Mes","Dens","M3_Arb","M3_HA","T_EFEC",
           "dias_trab","N__Trab","Coor_X","Coor_Y"]

cat_attribs_bin= ["Nombre_Especie","Unidad","Tipo_eq","season"]

cat_attribs_int = ["EMPRESA","Emsefor"]

num_pipeline = Pipeline([
        ('selector',DataFrameSelector(num_attribs)),
        ('imputer', Imputer(strategy="median")),
        ('std_scaler', StandardScaler()),
        ])

cat_pipeline_bin = Pipeline([
        ('selector', DataFrameSelector(cat_attribs_bin)),
        ('label_binarizer', LabelBinarizerForMultiplesFeatures()),
        ])

cat_pipeline_int = Pipeline([
        ('selector', DataFrameSelector(cat_attribs_int)),
        ('label_encoder', LabelEncoderForMultiplesFeatures()),
        ('std_scaler', StandardScaler()),
        ])

full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline_bin", cat_pipeline_bin),
        ])

full_pipeline_int = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline_bin", cat_pipeline_bin),
        ("cat_pipeline_int", cat_pipeline_int),        
        ])


### FEATURES

cosecha_prepared = full_pipeline_int.fit_transform(cosecha_dataframe)
#cosecha_prepared = num_pipeline.fit_transform(cosecha_dataframe)

### TARGET
cosecha_label = cosecha_label.values
