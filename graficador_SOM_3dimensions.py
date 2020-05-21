#!/usr/bin/env 
# Hugo Ubilla
# -*- coding: utf-8 -*-

"""
Created on Mon Aug 13 04:07:13 2018

@author: hugoub
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import time
 
from numpy import array,float32

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion

import matplotlib.pyplot as plt
#matplotlib.use('Agg')
from matplotlib import colors, cm
from matplotlib.collections import RegularPolyCollection

exec(open('./Resultados_SOM/resultadosSOM3D-05-10-5x5x5','r').read())

### CARGAR DATOS DF

RUTA_DATASET = "datasets/madera" # Ruta del Workspace

def cargar_datos(path=RUTA_DATASET):
    csv_path = os.path.join(path, "datos_corregidos_raleo.csv")
    return pd.read_csv(csv_path, sep = ",") # o simplemente poner la ruta completa en esta funcion

cosecha_dataframe = cargar_datos()


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

nombre_datos = []
nombre_datos2 = []

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
        global nombre_datos
        enc = LabelBinarizer(sparse_output=self.sparse_output)
        transformacion = np.empty(shape=(X.shape[0],0))
        for column in X.columns:
            resto = enc.fit_transform(X[column])
            nombre_datos.append(enc.classes_)
            nombre_datos2.append(enc.classes_[:-1])
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

cosecha_datos = cosecha_dataframe[num_attribs].to_numpy()
cosecha_nombres = list(cosecha_dataframe[num_attribs].columns.values)

nombres = []
nombres2 = []

for i in range(len(num_attribs)):
    nombres.append(num_attribs[i])
    nombres2.append(num_attribs[i])
    
for i in range(len(cat_attribs_bin)):
    for j in range(len(nombre_datos[i])):
        nombres.append(str(cat_attribs_bin[i]) +str('\n') + str(nombre_datos[i][j]))
    for j in range(len(nombre_datos2[i])):
        nombres2.append(str(cat_attribs_bin[i]) +str('\n') + str(nombre_datos2[i][j]))

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as cm
import numpy as np
from itertools import product, combinations

nombres2[16] = 'EMSEFOR\nAgricola y Forestal\ndel Sur LTDA.'
nombres2[17] = 'EMSEFOR\nEmpresa Marquez\ny Figueroa LTDA.'
nombres2[18] = 'EMSEFOR\nForestal Mantoverde\nLTDA.'
nombres2[19] = 'EMSEFOR\nForestal Reloncavi\nLTDA.'
nombres2[20] = 'EMSEFOR\nImport. y Comer.\nMetsakone LTDA.'
nombres2[21] = 'EMSEFOR\nServicios Forestales\nJ. Gonzalez LTDA.'
nombres2[22] = 'EMSEFOR\nSinergia Forestal\nAriel Guzmán Eirl'

nombres3 = nombres2.copy()
nombres3[0] = 'dens'
nombres3[1] = 'm3_tree'
nombres3[2] = 'm3_ha'
nombres3[3] = 'n_workers'
nombres3[4] = 'effective_t'
nombres3[5] = 'coor_x'
nombres3[6] = 'coor_y'
nombres3[7] = "slope"
nombres3[8] = 'vol_NOC'
nombres3[9] = 'specie\nEuca Globolus'
nombres3[10] = 'specie\nEuca Nitens'
nombres3[11] = 'type_eq\nGrapple skidder'
nombres3[12] = 'type_eq\nCable skidder'
nombres3[13] = 'season\nWinter'
nombres3[14] = 'season\nAutumn'
nombres3[15] = 'season\nSpring'

fig = plt.figure(figsize=(4,10))

for num in range(len(nombres2)):
#for num in range(2):
    # Seleccionar peso de atributo a graficar
    atributo = num
    matriz_peso = np.zeros(shape = (m1,n1,l1))
    ubicacion = np.zeros(shape = (m1*n1,2))
    matriz_peso_2 = np.zeros(shape = (m1 * n1)) 
    
    for i in range(m1):
        for j in range(n1):
            for k in range(l1):
                matriz_peso[i,j,k] = centroides[i][j][k][atributo]
    
    
    cmap = cm.cm.get_cmap('gist_rainbow')
    
    
    # Escalar entre 0 y 1
    matriz_peso1 = np.zeros(shape = (m1,n1,l1))
    matriz_peso.max()
    matriz_peso.min()
    for i in range(m1):
        for j in range(n1):
            for k in range(l1):
                matriz_peso1[i,j,k] = (matriz_peso[i,j,k] - matriz_peso.min()) / (matriz_peso.max() - matriz_peso.min())


    
    ax = fig.add_subplot(8,3,1+num, projection='3d')
    
    ax.set_title(nombres3[num], fontname="serif", fontsize = 4, pad = 1)
    
    '''
    if num<16: 
        ax.set_title(nombres3[num], fontname="serif", fontsize = 4, pad = 1)
    else:
        ax.set_title(nombres3[num], fontname="serif", fontsize = 3, pad = 0)
    '''             
    #ax.set_aspect("equal")
    
    maximo = max(m1,n1,l1) * 5
    
    ax.set_xlim(0,maximo)
    ax.set_ylim(0,maximo)
    ax.set_zlim(0,maximo)
    
    ax.set_axis_off()
    
    def crear_esfera(rad, pos):
        '''
        rad: numero que corresponde al radio de la esfera
        pos: lista que corresponde a la esfera en [x,y,z]
        '''
        u = np.linspace(0, 2 * np.pi, 5) # con el el ter
        v = np.linspace(0, np.pi, 5)
        x = rad * np.outer(np.cos(u), np.sin(v)) + pos[0]
        y = rad * np.outer(np.sin(u), np.sin(v)) + pos[1]
        z = rad * np.outer(np.ones(np.size(u)), np.cos(v)) + pos[2]
        return x,y,z
            
    
        
    for i in range(m1):
        for j in range(n1):
            for k in range(l1):
                x,y,z = crear_esfera(0.75,[i*5,j*5,k*5])
                rgba = cmap(matriz_peso1[i,j,k])
                ax.plot_surface(x, y, z, color=rgba)
                

    # Plot the surface
    #plt.savefig('./Resultados_SOM/resultadosSOM3D-'+str(l_rate)[2:]+'-'+str(+n_iter)+'-'+str(m1)+'x'+str(n1)+'x'+str(l1)+'-CUBO')


sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
sm._A = []
cax = fig.add_axes([0.1,0.1,0.825,0.01]) 
cbar = plt.colorbar(sm, cax=cax, orientation='horizontal')

cbar.ax.tick_params(labelsize=6)

#plt.savefig("lector3d2.pdf", bbox_inches='tight')





















########################
### Inicio Kmeans ######
########################

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

X = np.array(centroides).reshape(m1*n1*l1,cosecha_prepared.shape[1])

# k means determine k
distortions = []
K = range(1,10)

km = [KMeans(n_clusters=i) for i in K]
score = [km[i].fit(X).score(X) for i in range(len(km))]
inercia = [km[i].fit(X).inertia_ for i in range(len(km))]

sil_coeff = [silhouette_score(X, km[i].labels_, metric='euclidean') for i in range(1,len(km))]
print("Silueta ",sil_coeff )

#km[5].labels_
#km[5].cluster_centers_
#km[5].inertia_  #Inertia: Sum of squared distances of samples to their closest cluster center

def distancia(p1,p2,p3):
    '''
    p1 es el primer punto de la recta
    p2 es el ultimo punto de la recta
    p3 es el punto al cual se le quiere obtener la distancia
    '''
    
    m_recta = (p1[1] - p2[1])/(p1[0] - p2[0])
    n_recta = p1[1] - m_recta * p1[0]

    m_punto = -m_recta
    n_punto = p3[1] - m_punto * p3[0]
    
    x_recta = (n_recta - n_punto) / (m_punto - m_recta)
    y_recta = x_recta * m_recta + n_recta
    
    p_recta = np.array([x_recta, y_recta])
    
    linea = np.array([p3, p_recta])

    return np.linalg.norm(p3 - p_recta), linea


d = []
lineas = []

for i in K:
    p1 = [1, inercia[0]]
    p2 = [len(km), inercia[len(km)-1]]
    a, b = distancia(p1,p2,np.array([i,inercia[i-1]]))
    d.append(a)
    lineas.append(b)

lineas = np.array(lineas)
print(d)
cluster_sel = np.argmax(d)+1
#cluster_sel = 2
print("Numero de cluster :",np.argmax(d)+1)

p = np.array([p1,p2])
plt.figure(figsize=(3.5,3.0))

plt.plot(K, inercia, 'bx-', label="SSE")
plt.plot(p[:,0],p[:,1], '--', label="Dashed line")     # Recta
plt.plot(K, d, 'bx-', c = 'r', label="Distance SSE to dashed line")              # Distancia

#plt.plot(lineas[2,:,0],lineas[2,:,1])
plt.xlabel("Number of cluster", fontsize=8)
plt.ylabel("SSE", fontsize=8)

plt.legend(bbox_to_anchor=(0.35, 0.98), loc=2, borderaxespad=0., fontsize=7)

#plt.savefig('./Resultados_SOM/resultadosSOM3D-'+str(l_rate)[2:]+'-'+str(+n_iter)+'-'+str(m1)+'x'+str(n1)+'x'+str(l1)+'-LINEA')
#plt.savefig("elbow_method_eng.pdf", bbox_inches='tight')

plt.show()

peso_k = km[cluster_sel-1].labels_
peso_k1 = km[cluster_sel-1].labels_.reshape(m1,n1,l1)

#### FIN kmeans ###########
###########################
















###########################
#### Grafico cubo kmeans ###

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as cm
import numpy as np
from itertools import product, combinations


# Seleccionar peso de atributo a graficar
atributo = 1
matriz_peso = np.zeros(shape = (m1,n1,l1))
ubicacion = np.zeros(shape = (m1*n1,2))
matriz_peso_2 = np.zeros(shape = (m1 * n1)) 

for i in range(m1):
    for j in range(n1):
        for k in range(l1):
            matriz_peso[i,j,k] = centroides[i][j][k][atributo]


cmap = cm.cm.get_cmap('rainbow')

# Escalar entre 0 y 1
matriz_peso2 = np.zeros(shape = (m1,n1,l1))
peso_k1.max()
peso_k1.min()
for i in range(m1):
    for j in range(n1):
        for k in range(l1):
            matriz_peso2[i,j,k] = (peso_k1[i,j,k] - peso_k1.min()) / (peso_k1.max() - peso_k1.min())


fig = plt.figure(figsize=(3.5,3.5))
ax = fig.add_subplot(111, projection='3d')

#ax.set_aspect("equal")

maximo = max(m1,n1,l1) * 5
ax.set_xlim(0,maximo)
ax.set_ylim(0,maximo)
ax.set_zlim(0,maximo)

ax.set_axis_off()

def crear_esfera(rad, pos):
    '''
    rad: numero que corresponde al radio de la esfera
    pos: lista que corresponde a la esfera en [x,y,z]
    '''
    u = np.linspace(0, 2 * np.pi, 5) # con el el ter
    v = np.linspace(0, np.pi, 5)
    x = rad * np.outer(np.cos(u), np.sin(v)) + pos[0]
    y = rad * np.outer(np.sin(u), np.sin(v)) + pos[1]
    z = rad * np.outer(np.ones(np.size(u)), np.cos(v)) + pos[2]
    return x,y,z

    
for i in range(m1):
    for j in range(n1):
        for k in range(l1):
            x,y,z = crear_esfera(1,[i*5,j*5,k*5])
            rgba = cmap(matriz_peso2[i,j,k])
            ax.plot_surface(x, y, z, color=rgba)

ax.view_init(azim=60)

# Plot the surface
plt.show()    
#fig.savefig('./Resultados_SOM/resultadosSOM3D-'+str(l_rate)[2:]+'-'+str(+n_iter)+'-'+str(m1)+'x'+str(n1)+'x'+str(l1)+'-CUBOCLUSTER')
fig.savefig("kmeans3d.pdf", bbox_inches='tight')

### Fin grafico Cubo kmeans ###
###############################






















pesos_3d_a_2d = []

for k in range(l1):
    for i in range(m1):
        if i%2==0:
            for j in range(n1):
                pesos_3d_a_2d.append(matriz_peso2[i,j,k])
        else:
            for j in range(m1-1,-1,-1):
                pesos_3d_a_2d.append(matriz_peso2[i,j,k])


pesos_3d_a_2d = np.array(pesos_3d_a_2d)


########################
### Grafico de Panal cuadrado ###

a = 0.5
b = 0.8660254 

m_nuevo = 27
n_nuevo = 27

centros = np.zeros(shape = ( m_nuevo * n_nuevo ,2))

for i in range(m_nuevo):
    for j in range(n_nuevo):
        if j % 2 == 0:
            centros[i*n_nuevo+j] = np.array([1. + i, b + b * j])
        else:
            centros[i*n_nuevo+j] = np.array([1. + i + a, b + b * j])

fig = plt.figure(figsize=(3.5,3.5))
ax = fig.add_subplot(111)

collection_bg = RegularPolyCollection(
    numsides = 6,  # a hexagon
    sizes=(35,),
    edgecolors = 'black',
    linewidths = (0.3,),
    array = pesos_3d_a_2d,
    cmap =  'rainbow',
    offsets = centros,
    transOffset = ax.transData,
)

ax.add_collection(collection_bg)
ax.set_axis_off()
#ax.set_xlim([0, 11.5])
#ax.set_ylim([0, 11.5])
ax.axis('equal')
#fig.colorbar(res, shrink = 0.7)

#fig.savefig('./Resultados_SOM/resultadosSOM-'+str(l_rate)[2:]+'-'+str(+n_iter)+'-'+str(m1)+'x'+str(n1)+'-PANALCLUSTER')

plt.savefig("kmeansPlano3d.pdf", bbox_inches='tight')

### FIN Grafico de Panal ###
############################
















#########################
### Grafico de Panal ###

a = 0.5
b = 0.8660254 

m_nuevo = 9
n_nuevo = 81

centros = np.zeros(shape = ( m_nuevo * n_nuevo ,2))

for i in range(m_nuevo):
    for j in range(n_nuevo):
        if j % 2 == 0:
            centros[i*n_nuevo+j] = np.array([1. + i, b + b * j])
        else:
            centros[i*n_nuevo+j] = np.array([1. + i + a, b + b * j])

fig = plt.figure(figsize=(1,10))
ax = fig.add_subplot(111)

collection_bg = RegularPolyCollection(
    numsides = 6,  # a hexagon
    sizes=(25,),
    edgecolors = 'black',
    linewidths = (0.3,),
    array = pesos_3d_a_2d,
    cmap =  'rainbow',
    offsets = centros,
    transOffset = ax.transData,
)

ax.add_collection(collection_bg)
ax.set_axis_off()
#ax.set_xlim([0, 11.5])
#ax.set_ylim([0, 11.5])
ax.axis('equal')
#fig.colorbar(res, shrink = 0.7)

#fig.savefig('./Resultados_SOM/resultadosSOM-'+str(l_rate)[2:]+'-'+str(+n_iter)+'-'+str(m1)+'x'+str(n1)+'-PANALCLUSTER')
plt.savefig("kmeansLargo3d.pdf", bbox_inches='tight')
### FIN Grafico de Panal ###
############################




























'''

# We need obtain all indices for cluster trough cosecha prepared 

guardar_datos = []
for i in range(cluster_sel):    
    guardar_datos.append([])

# la numeracion de abajo se hace como analogia a la numeracion del arreglo centroides.

for i,j in enumerate(mapped):
    num = j[0] * n1 * l1 + j[1] * l1 + j[2]
    guardar_datos[km[cluster_sel-1].labels_[num]].append(cosecha_datos[i])
    
  
guardar_datos = np.array(guardar_datos)

for i in range(cluster_sel):
    guardar_datos[i] = np.array(guardar_datos[i])
   

# Promedio para atributo 0
for i in range(len(cosecha_datos[0])):
    print(cosecha_nombres[i])
    for j in range(cluster_sel):
        print(np.mean(guardar_datos[j][:,i]))
        print(np.std(guardar_datos[j][:,i]))
    print('--------')

'''

guardar_datos = []
datos_cluster = []
for i in range(cluster_sel):    
    guardar_datos.append([])
    datos_cluster.append([])
    
numeros = []
for i in range(m1*n1*l1):
    numeros.append(0)

for i,j in enumerate(mapped):
    num = j[0] * n1 * l1 + j[1] * l1 + j[2]
    numeros[num] = numeros[num] + 1
    
print("Promedio de nodos: ",np.mean(numeros))
print("Desviacion estandar nodos: ",np.std(numeros))

#cuenta_ceros = 0
#for i in range(len(numeros)):
#    if numeros[i] == 0:
#        cuenta_ceros = cuenta_ceros + 1

for i,j in enumerate(mapped):
    num = j[0] * n1 * l1 + j[1] * l1 + j[2]
    #km[cluster_sel-1].labels_[num] es el cluster donde va el dato num (van por orden)
    guardar_datos[km[cluster_sel-1].labels_[num]].append(np.append(cosecha_datos[i],cosecha_prepared[i,len(cosecha_datos[0]):23]))
    datos_cluster[km[cluster_sel-1].labels_[num]].append(i)
    
    
guardar_datos = np.array(guardar_datos)

cat_attribs_bin= ["Nombre_Especie","Tipo_eq","season","Emsefor"]

for i in range(cluster_sel):
    print(pd.value_counts(cosecha_dataframe['inclinacion'].iloc[datos_cluster[i]]))

for i in range(cluster_sel):
    guardar_datos[i] = np.array(guardar_datos[i])
   
# Promedio para atributos
for i in range(23):
    print(nombres[i])
    for j in range(cluster_sel):
        print(np.mean(guardar_datos[j][:,i]))
    print('--------')
    