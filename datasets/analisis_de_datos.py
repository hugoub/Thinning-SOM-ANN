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

### Seleccion de atributos ###
cosecha_label =  cosecha_dataframe[["Vol_NOC"]].copy()

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

#Para saber si esta bien
#cosecha_dataframe[['Mes','dia','season']]

cosecha_dataframe = cosecha_dataframe.sort_values(by='Fecha', ascending=True)

#cosecha_dataframe.to_csv('./datasets/madera/datos_corregidos.csv', sep=',', index=False)

#cosecha_dataframe_cat[["Unidad"]].describe(include = 'all')

################################################# De aca empezar a modificar

#### datos numericos y datos categoricos ###


'''
att_num = ["ano","Mes","Cod_Sip","cod_emsefor", "Dens","M3_Arb","M3_HA",
           "T_EFEC","T_PERD","dias_trab","N__Trab","H_H_efectivas",
           "Volumen_H_H","Superficie_Total","Coor_X","Coor_Y"]

att_cat = ["EMPRESA","Nombre_Especie","estacion","Unidad",
           "Tipo_eq","Sigla","Fecha","Predio","Emsefor"]

'''

num_attribs = ["Dens","M3_Arb","M3_HA","T_EFEC","dias_trab","N__Trab",
               "Coor_X","Coor_Y"]

cat_attribs_bin= ["Nombre_Especie","Unidad","Tipo_eq","season"]

cat_attribs_int = ["EMPRESA","Emsefor"]





# Se elimino ano, mes, cod_sip, cod_emsefor


cosecha_dataframe[num_attribs].describe()


'''
#################################################
########### PARTE PARA ANALIZAR DATOS ###########
#################################################

nombre_a_analizar = 'Emsefor'


cosecha_dataframe[[nombre_a_analizar]].info()
cosecha_dataframe[[nombre_a_analizar]].describe()
#histo 1 solo para num
cosecha_dataframe[nombre_a_analizar].hist()
#hist 2 solo para cat
valores = cosecha_dataframe[nombre_a_analizar].value_counts()
len(cosecha_dataframe[nombre_a_analizar].unique())
cosecha_dataframe[nombre_a_analizar].value_counts().plot.bar()

# Ver porcentajes de repeticion
acum = 0
numero = 0
for i in range(len(valores)):
    acum = valores[i] + acum
    numero = numero + 1
    if acum > 73252 * 0.99:
        print(acum)
        print(numero)
        break
       
nombre_a_analizar = 'Superficie_Total'

cosecha_dataframe[[nombre_a_analizar]].info()
cosecha_dataframe[[nombre_a_analizar]].describe()

cosecha_dataframe[nombre_a_analizar].hist()

asd = np.log(cosecha_dataframe[nombre_a_analizar]+1)
asd.hist()



# general
cosecha_dataframe.info()
cosecha_dataframe.describe()
'''



'''
cosecha_dataframe['Mes'].iloc[60797] 
cosecha_dataframe['dia'].iloc[60797] 

cosecha_dataframe['otono'].iloc[60797] 
cosecha_dataframe['invierno'].iloc[60797]
cosecha_dataframe['primavera'].iloc[60797]
cosecha_dataframe['verano'].iloc[60797]

aux= 0
for i in cosecha_dataframe.index:
    aux = cosecha_dataframe['otono'].iloc[i] + cosecha_dataframe['invierno'].iloc[i] + cosecha_dataframe['primavera'].iloc[i] + cosecha_dataframe['verano'].iloc[i]
    if aux != 1:
        print('malo %i'%i)
    
'''

####################### CORRELACION #####################
corr_matrix = cosecha_dataframe.corr()

for i in corr_matrix.columns:
    for j in corr_matrix[i].index:
        if (corr_matrix[i][j] > 0.5 or corr_matrix[i][j] < -0.5) and i!=j:
            print(i,' y ',j)

# -0.7
cosecha_dataframe.plot.scatter(x = 'Dens', y = 'M3_Arb', s =0.2 )

# 0.6
cosecha_dataframe.plot.scatter(x = 'M3_HA', y = 'M3_Arb', s =0.2 )

att = [ 'M3_HA', 'Dens', 'M3_Arb']
#scatter_matrix(cosecha_dataframe[att], figsize=(12, 8))


#####################   Algunos graficos

#Vol_Noc por coordenada

cosecha_dataframe2 = cosecha_dataframe.copy()

cosecha_dataframe2["Vol_NOC"] = cosecha_dataframe2["Vol_NOC"].apply(lambda x: min(2000,x))

graf = cosecha_dataframe2.plot.scatter(x="Coor_X",
            y="Coor_Y",
            cmap="coolwarm",
            #s = (cosecha_dataframe2["Vol_NOC"] * 0.2),
            c = cosecha_dataframe2["Vol_NOC"] / cosecha_dataframe2["Vol_NOC"].max(),
            alpha=0.7,
            figsize=(5,5),
            sharex = False,
            title = 'Predios'
            )
graf.get_figure().savefig('mapa.png')


# Vol Noc por fecha
a = cosecha_dataframe.groupby(by='Fecha')['Vol_NOC'].sum()
x = a.plot(figsize=(20,5), linestyle='', marker='.').get_figure()
x.savefig('linea.png')



# Plot por predio
b = cosecha_dataframe.groupby(by=['Predio','Fecha'])['Vol_NOC'].sum()
b = b.reset_index()             # Transforma en dataframe

fig, ax = plt.subplots(figsize=(20,5))
i=0
for name, group in b.groupby('Predio'):  #group es un dataframe
    group.plot(x='Fecha', y='Vol_NOC', ax=ax, label=name)
    i = i +1
    if i > 200:
        break
fig.savefig('linea2.png')

plt.show()



################### PCA

'''
att_num = ["ano","Mes","Cod_Sip","cod_emsefor",
                                           "Dens","M3_Arb","M3_HA","T_EFEC",
                                           "T_PERD","dias_trab","N__Trab",
                                           "H_H_efectivas",
                                           "Volumen_H_H","Superficie_Total",
                                           "Coor_X","Coor_Y"]

att_cat = ["EMPRESA","Nombre_Especie","estacion","Unidad",
                            "Tipo_eq","Sigla","Fecha","Predio","Emsefor"]

'''
cosecha_dataframe["Dens_al_cuad"] = np.square(cosecha_dataframe["Dens"])
cosecha_dataframe["Dens_root_square"] = np.sqrt(cosecha_dataframe["Dens"])
cosecha_dataframe["Dens_log"] = np.log(cosecha_dataframe["Dens"])
cosecha_dataframe["Dens_div"] = 1/cosecha_dataframe["Dens"]


cosecha_dataframe["Dens"].hist()
plt.suptitle('Histograma de Densidad')
plt.savefig('Histograma1.png')

fig, axes = plt.subplots(nrows=2, ncols=2)
plt.suptitle('Histogramas de transformaciones ')
plt.subplots_adjust(hspace = 0.4, wspace = 0.4)
axes[0,0].set_title('y = x²')
cosecha_dataframe["Dens_al_cuad"].hist(ax=axes[0,0])
axes[0,1].set_title('y = sqrt(x)')
cosecha_dataframe["Dens_root_square"].hist(ax=axes[0,1])
axes[1,0].set_title('y = log(x)')
cosecha_dataframe["Dens_log"].hist(ax=axes[1,0])
axes[1,1].set_title('y = 1/x')
cosecha_dataframe["Dens_div"].hist(ax=axes[1,1])
plt.savefig('Histograma2.png')

#cosecha_dataframe["Dens"] = np.log(cosecha_dataframe["Dens"])
#cosecha_dataframe["M3_Arb"] = np.log(cosecha_dataframe["M3_Arb"])

att_num = ["Dens","M3_Arb","M3_HA","T_EFEC", "dias_trab","N__Trab",
           "Coor_X","Coor_Y","Vol_NOC"]

att_cat = ["EMPRESA","Nombre_Especie","estacion","Unidad",
           "Tipo_eq"]

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names] 
    
num_pipeline = Pipeline([
        ('selector',DataFrameSelector(att_num)),
        ('imputer', Imputer(strategy="median")),
        ('std_scaler', StandardScaler()),
        ])
    
cosecha_prepared = num_pipeline.fit_transform(cosecha_dataframe)


# PCA en 2d

pca = PCA(n_components=2)
X2D = pca.fit_transform(cosecha_prepared)

val = np.matmul(np.transpose(cosecha_prepared),X2D)

plt.scatter(x = X2D[:,0], y = X2D[:,1], s = 8, cmap="coolwarm",
            c = cosecha_dataframe["Vol_NOC"] / cosecha_dataframe["Vol_NOC"].max())#, figsize=(10,10))
plt.colorbar(label='Volumen_NOC')
plt.title('Principal Component Analysis\n 2 Dimensiones')
plt.savefig('PCA2D.png')

borrar = []
aux = 0
for i,j in X2D:
    if i > 0.0 and j > 8.0:
        print('punto: ',X2D[aux])
        print('numero: ', aux)
        borrar.append(cosecha_dataframe[att_num].iloc[aux].name)
    aux = aux + 1

#pts : 50935, 58672, 59155, 72266, 73188

cosecha_dataframe[att_num].iloc[50935].name
cosecha_dataframe[att_num].iloc[73188]


cosecha_dataframe.drop(32227)

aux = 0
for i,j in X2D:
    if i > 4.8 and j < 1.0:
        print('punto: ',X2D[aux])
        print('numero: ', aux)
        borrar.append(cosecha_dataframe[att_num].iloc[aux].name)
    aux = aux + 1


#pts : 69170, 69173, 69183
cosecha_dataframe[att_num].loc[borrar[0]]
cosecha_dataframe[att_num].iloc[30227]

#loc busca por label
#iloc busca por el lugar en el dataframe

cosecha_dataframe = cosecha_dataframe.drop(borrar)

# PCA en 3d
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.colorbar import colorbar
pca3d = PCA(n_components=3)
X3D = pca3d.fit_transform(cosecha_prepared)

fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111, projection='3d')
xs = X3D[:,0]
ys = X3D[:,1]
zs = X3D[:,2]
p = ax.scatter(xs, ys, zs, cmap="coolwarm", 
           c= cosecha_dataframe["Vol_NOC"] / cosecha_dataframe["Vol_NOC"].max())
#ax.colorbar(label='Volumen_NOC')
ax.set_title('Principal Component Analysis\n 3 Dimensiones')
fig.colorbar(p, shrink = 0.8).set_label('Volumen_NOC')
fig.savefig('PCA3D.png')
plt.show()


aux = 0
for i,j,k in X3D:
    if i > -4.0 and j > 0.0 and k > 7:
        print('punto: ',X3D[aux])
        print('numero: ', aux)
        if cosecha_dataframe[att_num].iloc[aux].name not in borrar:
            borrar.append(cosecha_dataframe[att_num].iloc[aux].name)
    aux += 1
    
cosecha_dataframe[att_num].iloc[51675]
