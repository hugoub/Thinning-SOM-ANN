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

# Opciones para Pandas
pd.options.display.max_rows = 30                    # 10 filas maximo
pd.options.display.max_columns = 30                 # 10 columnas maximo
pd.options.display.float_format = '{:.1f}'.format   # Solo un decimal en float

from pandas.plotting import scatter_matrix

### Cargar conjunto de Datos como DataFrame ###

RUTA_DATASET = "datasets/madera" # Ruta del Workspace

def cargar_datos(path=RUTA_DATASET):
    csv_path = os.path.join(path, "Vnoc.csv")
    return pd.read_csv(csv_path, sep = ",") # o simplemente poner la ruta completa en esta funcion

cosecha_dataframe = cargar_datos()

### Cambiar nombre a una columna ###

#cosecha_dataframe.info()
new_columns = cosecha_dataframe.columns.values; 
new_columns[23] = 'Superficie_Total'; 
cosecha_dataframe.columns = new_columns 

##################################################################
###################### BORRADO DE PUNTOS #########################
##################################################################

ptos_atipicos =  [28391, 28518, 28519, 44455, 44456]
cosecha_dataframe = cosecha_dataframe.drop(ptos_atipicos)

### BORRAR COSECHA
indices_raleo = []
for i in cosecha_dataframe['Unidad'].index:
    if cosecha_dataframe['Unidad'].loc[i] == 'Cosecha':
        indices_raleo.append(i)

cosecha_dataframe = cosecha_dataframe.drop(indices_raleo)
### fin BORRAR COSECHA

### BORRAR Coor_X vacios (tb se borran las Coor_Y)
indices_vacios_coor_x = []
for i in cosecha_dataframe['Coor_X'].index:
    if np.isnan(cosecha_dataframe['Coor_X'].loc[i]):
        indices_vacios_coor_x.append(i)

cosecha_dataframe = cosecha_dataframe.drop(indices_vacios_coor_x)
### fin Coor_X y Coor_Y

##################################################################
###################### FIN BORRADO DE PUNTOS #####################
##################################################################

# Si se borran datos se debe reindexar
cosecha_dataframe.index = range(len(cosecha_dataframe.index))


#### TRANSFORMACION DE DATOS SEGUN HISTOGRAMA

cosecha_dataframe["Dens"] = np.log(cosecha_dataframe["Dens"])
cosecha_dataframe["M3_Arb"] = np.sqrt(cosecha_dataframe["M3_Arb"])
cosecha_dataframe["M3_HA"] = np.sqrt(cosecha_dataframe["M3_HA"])
cosecha_dataframe["H_H_efectivas"] = np.sqrt(cosecha_dataframe["H_H_efectivas"])

##########################


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
    if cosecha_dataframe['otono'].loc[i] == 1:     
        season.append('otono')
    if cosecha_dataframe['invierno'].loc[i] == 1:
        season.append('invierno')     
    if cosecha_dataframe['primavera'].loc[i] == 1:
        season.append('primavera')     
    if cosecha_dataframe['verano'].loc[i] == 1:
        season.append('verano')

season = np.array(season)
cosecha_dataframe['season'] = pd.Series(season)


# Agregar columna de inclinación donde vale 1 si hay torre y 0 en otro caso
inclinacion = []
for i in cosecha_dataframe.index:
    if "Torre" in cosecha_dataframe['Tipo_eq'].loc[i]:
        inclinacion.append(1)
    else:
        inclinacion.append(0)
        
inclinacion = np.array(inclinacion)
cosecha_dataframe['inclinacion'] = pd.Series(inclinacion)

# Exportan datos corregidos Raleo

cosecha_dataframe.to_csv('./datasets/madera/datos_corregidos_raleo.csv', sep=',', index=False)






"""
### CORRELACIONES

atributos = ["Dens","M3_Arb","M3_HA","N__Trab","T_EFEC","Coor_X","Coor_Y",
               "inclinacion","Vol_NOC","Nombre_Especie","Tipo_eq","season","Emsefor"]

cosecha_dataframe[atributos].corr().to_csv('hjdaso.csv')



pd.concat(list_of_dataframes).to_csv(...)
"""