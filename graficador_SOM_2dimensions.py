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
 
from numpy import array,float32

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer

from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion

import matplotlib.pyplot as plt
#matplotlib.use('Agg')
from matplotlib import colors, cm
from matplotlib.collections import RegularPolyCollection

exec(open('./Resultados_SOM/resultadosSOM-05-50-10x10','r').read())

### CARGAR DATOS DF

RUTA_DATASET = "datasets/madera" # Ruta del Workspace

def cargar_datos(path=RUTA_DATASET):
    csv_path = os.path.join(path, "datos_corregidos_raleo.csv")
    return pd.read_csv(csv_path, sep = ",") # o simplemente poner la ruta completa en esta funcion

cosecha_dataframe = cargar_datos()

s_set = 1

'''
############################################################
quitar = []
s_set = 2
for i in cosecha_dataframe.index:    
    if cosecha_dataframe['Vol_NOC'].loc[i] >= cosecha_dataframe['Vol_NOC'].quantile(0.99):
        quitar.append(i)
    if cosecha_dataframe['Vol_NOC'].loc[i] <= cosecha_dataframe['Vol_NOC'].quantile(0.01):
        quitar.append(i)

cosecha_dataframe = cosecha_dataframe.drop(quitar)

cosecha_dataframe.index = range(len(cosecha_dataframe.index))

############################################################
'''

# Aleatorizar datos mediante una permutación aleatoria

np.random.seed(10) # Seed
cosecha_dataframe = cosecha_dataframe.reindex(
        np.random.permutation(cosecha_dataframe.index))

### TRANSFORMACIONES

# Funciones de Transformaciones

class DataFrameSelector(BaseEstimator, TransformerMixin):
    '''
    Entrada: Recibe todo el nombre de los atributos del dataframe que se
    utilizaran.
    Fit: Se ingresa todo el DataFrame.
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

#num_attribs = ["Dens","M3_Arb","M3_HA","N__Trab","T_EFEC"]

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
#for num in range(9):
    # Seleccionar peso de atributo a graficar
    atributo = num
    matriz_peso = np.zeros(shape = (m1,n1))
    ubicacion = np.zeros(shape = (m1*n1,2))
    matriz_peso_2 = np.zeros(shape = (m1 * n1)) 
    
    # m1 es el numero de filas
    # n1 es el numero de columnas
    # ubicacion van por columna hacia abajo (def arbitrariamente)
    for i in range(m1):
        for j in range(n1):
            matriz_peso[i,j] = centroides[i][j][atributo]
            ubicacion[i * n1 + j] = np.array([i,j])
            matriz_peso_2[i * n1 + j] = centroides[i][j][atributo]
    
    
    ## Escalar entre 0 y 1
    #matriz_peso1 = np.zeros(shape = (m1,n1))
    #matriz_peso.max()
    #matriz_peso.min()
    #for i in range(m1):
    #    for j in range(n1):atributo
    #        matriz_peso1[i,j] = (matriz_peso[i,j] - matriz_peso.min()) / (matriz_peso.max() - matriz_peso.min())
    
    
    mapa_color = 'gist_rainbow'

    
    #########################
    ### Grafico de Panal 27 ###
    
    a = 0.5
    b = 0.8660254 
    
    centros = np.zeros(shape = ( m1 * n1 ,2))
    
    for i in range(m1):
        for j in range(n1):
            if j % 2 == 0:
                centros[i*n1+j] = np.array([1. + i, b + b * j])
            else:
                centros[i*n1+j] = np.array([1. + i + a, b + b * j])
    
    ax = fig.add_subplot(8,3,1+num)
    
    if num<16: 
        ax.set_title(nombres3[num], fontname="serif", fontsize = 4, pad = 1)
    else:
        ax.set_title(nombres3[num], fontname="serif", fontsize = 3, pad = 0)
                
    collection_bg = RegularPolyCollection(
        numsides = 6,  # a hexagon
        sizes=(6,),
        edgecolors = 'black',
        linewidths = (0.0,),
        array = matriz_peso_2,
        cmap =  'gist_rainbow',
        offsets = centros,
        transOffset = ax.transData,
        #rasterized=True,
    )
    
    ax.add_collection(collection_bg)
    ax.set_axis_off()
    #ax.set_xlim([0, 11.5])
    #ax.set_ylim([0, 11.5])
    ax.axis('equal')
    #fig.colorbar(res, shrink = 0.7)
    
    #fig.savefig('./Resultados_SOM/resultadosSOM-'+str(l_rate)[2:]+'-'+str(+n_iter)+'-'+str(m1)+'x'+str(n1)+'-PANAL')
 
    
    ### FIN Grafico de Panal 27 ###
    ############################
cmap = cm.get_cmap('gist_rainbow')
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
sm._A = []
# desp horizontal,desplazamineto vert, ancho, largo
cax = fig.add_axes([0.1,0.1,0.825,0.01]) 
cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
cbar.ax.tick_params(labelsize=6)

#fig.savefig("lector2d3.pdf", bbox_inches='tight')







#########################
### Inicio Kmeans ####

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

X = np.array(centroides).reshape(m1*n1,cosecha_prepared.shape[1])

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

print("Numero de cluster :",cluster_sel )

p = np.array([p1,p2])
plt.figure()
plt.plot(K, inercia, 'bx-')
plt.plot(p[:,0],p[:,1])
plt.plot(K, d)
#plt.plot(lineas[2,:,0],lineas[2,:,1])
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()
#plt.savefig('./Resultados_SOM/resultadosSOM-'+str(l_rate)[2:]+'-'+str(+n_iter)+'-'+str(m1)+'x'+str(n1)+'-LINEA')

km[cluster_sel-1].labels_

###################
### Fin Kmeans ####
###################







############################
#### Grafico de Circles ###
#
#fig = plt.figure(figsize=(6,6))
##plt.clf()
#ax = fig.add_subplot(111)
#ax.set_axis_off()
#ax.axis('equal')
#res = ax.scatter(ubicacion[:,0], ubicacion[:,1], c = km[cluster_sel-1].labels_, cmap=mapa_color)
#fig.colorbar(res, shrink = 0.8)
#
#fig.savefig('circulos.png')
#
#### FIN Grafico de Circulos ###
################################









################################
### Grafico de Panal Cluster ###

a = 0.5
b = 0.8660254 

centros = np.zeros(shape = ( m1 * n1 ,2))

for i in range(m1):
    for j in range(n1):
        if j % 2 == 0:
            centros[i*n1+j] = np.array([1. + i, b + b * j])
        else:
            centros[i*n1+j] = np.array([1. + i + a, b + b * j])

fig = plt.figure(figsize=(3.5,3.5))
ax = fig.add_subplot(111)

collection_bg = RegularPolyCollection(
    numsides = 6,  # a hexagon
    sizes=(35,),
    edgecolors = 'black',
    linewidths = (0.3,),
    array = km[cluster_sel-1].labels_,
    cmap =  'gist_rainbow',
    offsets = centros,
    transOffset = ax.transData,
)

ax.add_collection(collection_bg)
ax.set_axis_off()
#ax.set_xlim([0, 11.5])
#ax.set_ylim([0, 11.5])
ax.axis('equal')
#fig.colorbar(res, shrink = 0.7)

#plt.savefig("kmeans2d.pdf", bbox_inches='tight')
#fig.savefig('./Resultados_SOM/resultadosSOM-'+str(l_rate)[2:]+'-'+str(+n_iter)+'-'+str(m1)+'x'+str(n1)+'-PANALCLUSTER')

### FIN Grafico de Panal ###
############################














################################
### Análisis de Información ####

'''
guardar_datos = []
for i in range(cluster_sel):    
    guardar_datos.append([])

for i,j in enumerate(mapped):
    num = j[0] * n1 + j[1]
    #km[cluster_sel-1].labels_[num] es el cluster donde va el dato num (van por orden)
    guardar_datos[km[cluster_sel-1].labels_[num]].append(cosecha_datos[i])
    
guardar_datos = np.array(guardar_datos)

for i in range(cluster_sel):
    guardar_datos[i] = np.array(guardar_datos[i])
   
# Promedio para atributos
for i in range(len(cosecha_datos[0])):
    print(cosecha_nombres[i])
    for j in range(cluster_sel):
        print(np.mean(guardar_datos[j][:,i]))
    print('--------')
'''
    
guardar_datos = []
for i in range(cluster_sel):    
    guardar_datos.append([])

numeros = []
for i in range(m1*n1):
    numeros.append(0)
    
for i,j in enumerate(mapped):
    num = j[0] * n1 + j[1]
    numeros[num] = numeros[num] + 1
    #km[cluster_sel-1].labels_[num] es el cluster donde va el dato num (van por orden)
    guardar_datos[km[cluster_sel-1].labels_[num]].append(np.append(cosecha_datos[i],cosecha_prepared[i,len(cosecha_datos[0]):23]))

print("Promedio de nodos: ",np.mean(numeros))
print("Desviacion estandar nodos: ",np.std(numeros))

guardar_datos = np.array(guardar_datos)

for i in range(cluster_sel):
    guardar_datos[i] = np.array(guardar_datos[i])
   
# Promedio para atributos
for i in range(23):
    print(nombres[i])
    for j in range(cluster_sel):
        print(np.mean(guardar_datos[j][:,i]))
    print('--------')


#### CONTAR VALORES POR CONGLOMERADO
    
for i in range(cluster_sel):
    print(np.count_nonzero(km[cluster_sel-1].labels_ == i))

'''
matriz = np.zeros(shape = (m1,n1))
matriz1 = np.zeros(shape = (m1,n1))

# Sacar promedios

ma = []
for i in range(m1):
    ma.append([])
    for j in range(n1):
        ma[i].append([])

for i, m in enumerate(mapped):
    matriz1[m[0],m[1]] += 1 
    
for i, m in enumerate(mapped):
    ma[m[0]][m[1]].append(cosecha_dataframe2["V1"].iloc[i])     
    
for i, m in enumerate(mapped):
    matriz[m[0], m[1]] = np.mean(ma[m[0]][m[1]])
    
maximo = np.max(matriz)

for i, m in enumerate(mapped):
    matriz[m[0], m[1]] = np.mean(ma[m[0]][m[1]]) / maximo

mapa_color = 'cool'

# GRAFICO 1

fig = plt.figure(figsize=(6,6))
#plt.clf()
ax = fig.add_subplot(111)
#ax.set_aspect(1)
res = ax.imshow(np.array(matriz), cmap=mapa_color)


# GRAFICO 2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as cm

#n = numero de capas
#m = numero de nodos por capa

phi = np.linspace(0, 2. * np.pi, m1 + 1) # num de capas
theta = np.linspace(0, 2. * np.pi, n1 + 1) # num de puntos por capa


theta, phi = np.meshgrid(theta, phi)

# a: radio del toroide  (borde)
# c: radio del toroide desde el centro

a, c = 0.25, 0.7
x = (c + a*np.cos(theta)) * np.cos(phi)
y = (c + a*np.cos(theta)) * np.sin(phi)
z = a * np.sin(theta)


fig = plt.figure(figsize = (6,6))
ax1 = fig.gca(projection = '3d')

ax1.set_axis_off()

ax1.set_xlim3d(-1, 1)
ax1.set_ylim3d(-1, 1)
ax1.set_zlim3d(-1, 1)

cmap = cm.cm.get_cmap(mapa_color)

for i in range(m1):
    for j in range(n1):
        rgba = cmap(matriz[i, j])
        ax1.scatter(x[i][j], y[i][j], z[i][j], color=rgba)
#ax1.view_init(36, 26)

plt.show()
'''

