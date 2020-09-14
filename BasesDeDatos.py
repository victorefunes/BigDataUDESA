* Este archivo es para que lo copien y peguen como script en Python 
#==================================================#
#                                                  # 
#               CLASE TUTORIAL N°3                 #
#       Bases de datos y regresión lineal          # 
#                                                  # 
#==================================================#

# Profesor: Walter Sosa Escudero
# Tutora: Carla Srebot 

# Objetivo: armar y cargar bases de datos. Realizar análisis descriptivo y correr regresiones lineales. Estimar polinomios y encontrar el ECM.


# -------------------------------- #
#       NumPy y scikit-learn                   
# -------------------------------- #

# El paquete NumPy es fundamental en Python al permitir operaciones complejas con bases de datos de distintos tamaños. Para más información, ver la [guía oficial de uso de NumPy](https://docs.scipy.org/doc/numpy/user/index.html).

# El paquete scikit-learn es una biblioteca de Python usada para machine learning, construida encima de NumPy y otros paquetes. Permite procesar datos, reducir la dimensionalidad de la base, implementar regresiones, clasificaciones, clustering y más.


# -------------------------------- #
#       CARGANDO DATA FRAMES  
# -------------------------------- #

# csv
import os  
os.getcwd()  
os.chdir('C:\\Users\\csreb\\Dropbox\\Tutoriales\\Big Data 2020\\Tutoriales\\Tutorial 3 - Modelo Linea y base de datos')
import pandas as pd
data = pd.read_csv("Datos.csv",delimiter=';')
print(data)

# xlsx
import pandas as pd
df = pd.read_excel (r'Datos.xlsx', sheet_name='Hoja1')
print (df)


# -------------------------------- #
#       ARMANDO DATA FRAMES
# -------------------------------- #
import pandas as pd
import numpy as np

# Creamos una base de datos llamada df:
df = pd.DataFrame({
    'nombre': ['Flor', 'Juli', 'Anto', 'Alan', 'Gaby', 'Pau', 'Juan'],
    'cantidad_gatos': [0, 2, 3, 0, 0, 1, None],
    'cantidad_perros': [0, 1, 0, 2, None, 2, 1],
    'barrio': ['Nu', 'Re', 'Vi', 'Al', 'Pa', 'Ma', 'Re'],
})
df

# y otra base llamada df2:
df2 = pd.DataFrame({
    'nombre': ['Jorge', 'Martin', 'Tommy', 'Walter', 'Christian', 'Mariana', 'Gabriel'],
    'animal_favorito': ['perro', 'gallina', 'gato', 'perro', 'gato', 'gato', 'hamster']
})
df2

# Ahora, unimos los data frames
df3 = pd.merge(df, df2, how='outer', on='nombre')
df3


# Podemos unir los dataframes de varias formas:
# 1. pd.merge(df, df2, how='outer', on='nombre'): quedarse con las filas que aparezcan en df o en df2
# 2. pd.merge(df, df2, how='inner', on='nombre'): quedarse con las filas que aparezcan en df y en df2
# 3. pd.merge(df, df2, how='left', on='nombre'): quedarse con las filas de df
# 4. pd.merge(df, df2, how='right', on='nombre'): quedarse con las filas de df2


# Seleccionando una sola columna:
df3['nombre']

# Seleccionando muchas columnas:
df3[['nombre', 'cantidad_gatos', 'barrio']]

# Generando una columna que sea combinación de otras:
df3['cantidad_patas'] = 4 * (df['cantidad_gatos'] + df['cantidad_perros'])
df3

# Calculando medidas útiles:
print(df3['cantidad_patas'].sum())
print(df3['cantidad_patas'].median())
df3['animal_favorito'].value_counts()

# Eliminación de columnas:
df3 = df3.drop(['barrio'], axis=1) # Axis 1 es para que busque la fila de los nombres de las columnas 
 
# Para seleccionar cierta cantidad de filas:
df3.head(3) #del principio
df3.tail(5) #del final

# Seleccionar con un criterio: nos quedamos con los que tengan más perros que gatos...
df3.loc[df3['cantidad_perros'] > df3['cantidad_gatos']]

#... y quedarnos solo con columnas que nos interesen:
df3.loc[df3['cantidad_perros'] > df3['cantidad_gatos'], ['cantidad_perros', 'cantidad_gatos']]

# Seleccionar por un índice: primero hay que establecer cuál es el índice...
df3 = df3.set_index('animal_favorito')
df3

# ... ahora podemos acceder por animal favorito
df3.loc['gallina']

# Eliminar filas por índice: 
df3.drop(['gallina', 'hamster'])

# Pero si llamo a df3... 
df3

#... gallina y hamster siguen estando! ¿Por qué? Porque pandas hace copias siempre. Solo vas a eliminar
# estos dos animales de la base df3 si la reescribís:
df3 = df3.drop(['gallina', 'hamster'])  
df3 

# Para tirar todas las observaciones con datos faltantes:
df3.dropna(subset=['cantidad_gatos', 'cantidad_perros'])

# Agrupar por columna y luego agregar:
df3.groupby('animal_favorito').sum()

# Imputarles 0 a todos los indefinidos:
df3.fillna(0)

# Imputarles 0 a las filas donde el número de gatos está indefinido
df3.loc[df3['cantidad_gatos'].isnull(), 'cantidad_gatos'] = 0 


# -------------------------------- #
#       CARGANDO DATA FRAMES                  
# -------------------------------- #
# Archivo 'pluto_shorter.csv'
import pandas as pd
import numpy as np

url = 'https://github.com/worldbank/Python-for-Data-Science/raw/master/Spring%202019%208-week%20course/week%203/pluto_shorter.csv'

df = pd.read_csv(url)
print('Cargado el csv con {} filas y {} columnas'.format(df.shape[0], df.shape[1]))

# Veamos la data:
df.head()

# Listo las columnas y selecciono algunas:
df.columns

my_cols = ['borough','numfloors','yearbuilt', 'landuse', 'zipcode', 'unitstotal', 'assesstot']
df = df[my_cols] 

# Renombro las columnas:
df.columns

df.rename(columns = {'zipcode': 'zip', 'yearbuilt': 'anio_construccion', 'unitstotal': 'unidades', 'assesstotal': 'valor_usd'},
         inplace = True)

# Veamos la distribución de anio_construccion. ¿Algo raro?
df.anio_construccion.hist()
 
# Saco las observaciones problemáticas:
df.anio_construccion[df.anio_construccion < 1000] = np.nan 

df.anio_construccion.hist()

# Cuántos missing values tienen las columnas? 
df.isnull().sum()

print('Tamaño original: ', df.shape)
df.dropna(inplace=True) #Elimna las filas que no tienen datos
print('Nuevo tamaño: ', df.shape)

# Veamos características de algunas variables:

print("Max pisos: ", df.numfloors.max())
print("Valor medio: {:.3f}".format(df.assesstot.mean()))

# Hagamos un gráfico para explorar las relaciones de los datos:

df.plot(x = 'anio_construccion', y = 'numfloors', kind = 'scatter', title = 'Edificios de NYC: Año construido vs. número de pisos');


# --------------------------------- #
# REGRESIÓN LINEAL CON SCIKIT-LEARN                
# --------------------------------- #
import numpy as np
from sklearn.linear_model import LinearRegression 

x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1)) #reshape para que defina que es un vector columna
y = np.array([5, 20, 14, 32, 22, 38])

print(x)
print(y)

# Ahora utilizaremos la función LinearRegression(). 
    
# Se pueden proveer muchos parámetros opcionales para esta función:
#   fit_intercept es un Boolean que decide si calcular el intercepto (True) o considerarlo igual a cero (False). Por default es True.
#   normalize es un Boolean que decide si normalizar las variables input (True) o no (False). Es False por default.
#   copy_X es Boolean que decide si copiar (True) o sobreescribir las variables input (False). Es True por default.


# Primero tenemos que estimar el modelo, lo que haremos con fit():
model = LinearRegression()

model.fit(x, y)

# Podríamos haber escrito directamente:
model = LinearRegression().fit(x, y)

# Veamos ahora los resultados

# Calculo el R2
r_sq = model.score(x, y)
print('Coeficiente de determinación:', r_sq)

# El intercepto
print('Intercepto:', model.intercept_)

# La pendiente
print('Pendiente:', model.coef_)


# Supongamos que estamos contentos con este modelo, por lo que ahora queremos predecir. Vean que es bastante intuitivo: cuando aplicamos .predict(), metemos los valores del regresor en el modelo estimado y obtenemos la correspondiente respuesta predicha.
y_pred = model.predict(x)
print('Respuesta predicha:', y_pred, sep='\n')

# Si quiero probar valores nuevos de x (no los que usé para estimar el modelo):
x_new = np.arange(5).reshape((-1, 1))
print(x_new)
y_new = model.predict(x_new)
print('Nueva respuesta predicha:', y_new, sep='\n')

# Para regresión lineal múltiple es lo mismo:

# Armamos un vector para la variable dependiente y una matriz de regresores:
x = [[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]]
y = [4, 5, 20, 14, 32, 22, 38, 43]
x, y = np.array(x), np.array(y)

print(x)
print(y)

# Estimamos el modelo
model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)

# Miramos qué dio
print('Coeficiente de determinación:', r_sq)
print('Intercepto:', model.intercept_)
print('Pendiente:', model.coef_)

# Vemos la respuesta predicha para los valores originales de los regresores
y_pred = model.predict(x)
print('Respuesta predicha:', y_pred, sep='\n')

# Vemos la predicción para nuevos valores de X
x_new = np.arange(10).reshape((-1, 2))
print(x_new)
y_new = model.predict(x_new) 
print('Nueva respuesta predicha:', y_new, sep='\n')



# -------------------------------- #
#       REGRESIÓN POLINÓMICA           
# -------------------------------- #

# Necesitamos un nuevo paquete
from sklearn.preprocessing import PolynomialFeatures

x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1)) #¿por qué reshape?
y = np.array([15, 11, 2, 8, 25, 32])

# Hasta acá lo mismo que antes, pero resulta que lo que quiero es correr una regresión de y contra x y x al cuadrado, por lo que necesito generar los datos de la nueva variable independiente.

# Para eso vamos a usar la función nueva a la que llamaremos transformer por si Michael Bay viene a hacer el curso. Se pueden cambiar varios parámetros de PolynomialFeatures:
#   degree es un entero (2 por default) que representa el grado de la función de regresión polinómica.
#   include_bias es un Boolean (True por default) que decide si incluir la columna de 1 que corresponde al intercepto (True) o no (False).

transformer = PolynomialFeatures(degree=2, include_bias=False)
transformer.fit(x) 
x_ = transformer.transform(x)

# También podíamos ahorrarnos los tres pasos y correr x_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(x)

print(x_)
model = LinearRegression().fit(x_, y)



# -------------------------------- #
# Imitando a Stata con statsmodels          
# -------------------------------- #

# Todo muy lindo, pero no sé la significatividad de las variables, el estadístico F... Estarán pensando "¡Quiero volver a Stata!". No teman, viene otro paquete al rescate:

import statsmodels.api as sm

x = [[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]]
y = [4, 5, 20, 14, 32, 22, 38, 43]
x, y = np.array(x), np.array(y)

x = sm.add_constant(x)
model = sm.OLS(y, x)
results = model.fit()
print(results.summary())

print(results.summary().as_latex())

# Se puede obtener la respuesta predicha con los valores de x utilizados en el entrenamiento del modelo usando .fittedvalues o .predict() con la matriz de variables independientes como argumento:

print('predicted response:', results.fittedvalues, sep='\n')

print('predicted response:', results.predict(x), sep='\n')

x_new = sm.add_constant(np.arange(10).reshape((-1, 2)))
print(x_new)

y_new = results.predict(x_new)
print(y_new)


# -------------------------------- #
#      ERROR CUADRÁTICO MEDIO         
# -------------------------------- #

# Importamos paquetes
import numpy as np
import matplotlib.pyplot as plt

# Generamos un dataset aleatorio
np.random.seed(0)
x = np.random.rand(100, 1)
y = 2 + 3 * x + np.random.rand(100, 1)

# Graficamos
plt.scatter(x,y,s=10)  # s indica el tamaño de los puntos del scatter.
plt.xlabel('x')
plt.ylabel('y')
plt.show()

x = sm.add_constant(x)
model = sm.OLS(y, x)
results = model.fit()
print(results.summary())

y_pred = results.predict(x)
y_pred

# Error Cuadrático Medio (Mean Squared Error en inglés):
mse = (np.sum((y_pred - y)**2))/100
mse
