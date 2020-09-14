#==================================================#
#                                                  #
#               CLASE TUTORIAL N°4                 #
#                 Clasificación                    #
#                                                  #
#==================================================#

# Profesor: Walter Sosa Escudero
# Tutora: Carla Srebot 

# Objetivo: entender la "diferencia" entre clasificación y regresión. Utilizar Bayes, análisis de discriminante lineal y cuadrático y KNN. Análisis de la curva ROC.


# -------------------------------- #
#           Clasificación                  
# -------------------------------- #

# Vamos a clasificar 'y' en base a 'x', donde 'y' es una variable cualitativa. 
    # ¿Estamos ante un caso de aprendizaje supervisado o no supervisado?
    # ¿Qué es el clasificador de Bayes?

# Métodos:
    
    # 1. Logit - regresión logística 
    # 2. Análisis de discriminante: lineal y cuadrático
    # 3. KNN

# -------------------------------- #
#             Logit                  
# -------------------------------- #

# Algoritmo de clasificación que se usa para predecir la probabilidad de una variable dependiente categórica. El modelo logit predice P(Y=1) como una función de X.

# Vamos a usar datos del S&P Stock Market. Esta base contiene los retornos porcentuales del S&P 500 stock index por 1250 días, desde inicios de 2001 hasta el final de 2005. Para cada fecha, tenemos:

    # Lag1, Lag2,..., Lag5: retornos porcentuales de cada uno de los días anteriores.
    # Volume: volumen de acciones negociadas (número de acciones diarias negociadas en miles de millones de dólares)
    # Today: retorno porcentual de hoy
    # Direction: variable binaria que toma valores "Down" y "Up" indicando si el mercado tuvo un retorno positivo o negativo.
    
import os  
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt  # Para matriz de correlaciones
import seaborn as sns            # Para gráficos bonitos
import statsmodels.api as sm     # Para agregar la columna de 1 a la matriz X

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

os.getcwd()  
os.chdir('C:\\Users\\csreb\\Dropbox\\Tutoriales\\Big Data 2020\\Tutoriales\\Tutorial 4 - Clasificación')
data = pd.read_csv("smarket.csv")
print(data)

# Para calcular la correlación
datacor=data[['Year','Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume', 'Today']]
np.corrcoef(datacor)

# Matriz de correlación
plt.matshow(datacor.corr())
plt.xticks(range(len(datacor.columns)), datacor.columns) # Etiqueta del eje X
plt.yticks(range(len(datacor.columns)), datacor.columns) # Etiqueta del eje Y 
plt.colorbar() # Leyenda de los colores del gráfico
plt.show() 

data['Direction'].value_counts()

sns.countplot(x='Direction', data=data, palette='hls')
plt.show()
# Para más colores: https://seaborn.pydata.org/tutorial/color_palettes.html}

data.groupby('Direction').mean()

# Ahora, vamos a predecir si el S&P 500 index sube o baja:  

y = data['Direction']
y = y.replace('Up', 1)
y = y.replace('Down', 0)
 
X=data[['Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume']]
X = sm.add_constant(X)

logit_model=sm.Logit(y.astype(float),X.astype(float))
result=logit_model.fit()
print(result.summary2())
print(result.summary2().as_latex())

y_new = result.predict(X)

y_new=np.where(y_new>0.5, 1, y_new)
y_new=np.where(y_new<=0.5, 0, y_new)


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 

confusion_matrix = confusion_matrix(y, y_new)

print('Confusion Matrix :')
print(confusion_matrix) 
print('Accuracy Score :',accuracy_score(y, y_new))

# OJO: en Python la matriz de confusión tiene en las filas los valores ciertos y en las columnas los valores predichos

# -------------------------------- #
#       Medidas de precisión                 
# -------------------------------- #

# False Alarm Rate o False Positive Rate: FP rate = FP/N
# Recall o True Positive Rate o Sensitivity: TP rate = TP/P
# Precision: TP/(TP+FP)
# Accuracy: (TP+TN)/P+N
# Specificity: 1 - FP rate


# -------------------------------- #
#           Curva ROC                  
# -------------------------------- #

# ROC: Receiver Operating Characteristics

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()  

auc = roc_auc_score(y, y_new)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y, y_new)
plot_roc_curve(fpr, tpr)

# Parto la base:
train=data[data.Year < 2005]
test=data[data.Year >= 2005]
    
ytrain = train['Direction']
ytrain = ytrain.replace('Up', 1)
ytrain = ytrain.replace('Down', 0) 

ytest = test['Direction']
ytest = ytest.replace('Up', 1)
ytest = ytest.replace('Down', 0)

Xtrain=train[['Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume']]
Xtrain = sm.add_constant(Xtrain) 

Xtest=test[['Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume']]
Xtest = sm.add_constant(Xtest)

# Estimo:
logit_model=sm.Logit(ytrain.astype(float),Xtrain.astype(float))
result=logit_model.fit()
print(result.summary2())
print(result.summary2().as_latex()) 

y_pred = result.predict(Xtest)
y_pred=np.where(y_pred>0.5, 1, y_pred)
y_pred=np.where(y_pred<=0.5, 0, y_pred)

# AUC y ROC
auc = roc_auc_score(ytest, y_pred)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(ytest, y_pred)
plot_roc_curve(fpr, tpr)

# ¿Y si quiero implementar una regresión logística multinomial? Ver https://dataaspirant.com/2017/05/15/implement-multinomial-logistic-regression-python/


# -------------------------------- #
#    Análisis discriminante                 
# -------------------------------- #

# ¿Cuál es la diferencia entre LDA y QDA?

# -------------------------------- #
# Lineal

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

clf = LinearDiscriminantAnalysis()
clf.fit(Xtrain, ytrain)
resultslda=clf.predict(Xtest)
y_pred_lda=pd.Series(resultslda.tolist())

from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(ytest, y_pred_lda)
print(cm2)   

auc = roc_auc_score(ytest, y_pred_lda)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(ytest, y_pred_lda)
plot_roc_curve(fpr, tpr)

# Más información: http://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html


# -------------------------------- #
# QDA:
    
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import numpy as np

clf = QuadraticDiscriminantAnalysis()
clf.fit(Xtrain, ytrain)
resultslda=clf.predict(Xtest)
y_pred_lda=pd.Series(resultslda.tolist())

from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(ytest, y_pred_lda)
print(cm2) 

auc = roc_auc_score(ytest, y_pred_lda)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(ytest, y_pred_lda)
plot_roc_curve(fpr, tpr)

# Más información: https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html

# -------------------------------- #
#               KNN                 
# -------------------------------- #

# KNN o "dime con quién andas y te diré quién eres"

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
knn.fit(Xtrain, ytrain) 
y_pred = knn.predict(Xtest)
confusion_matrix(ytest, y_pred) 
auc = roc_auc_score(ytest, y_pred)

print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(ytest, y_pred)
plot_roc_curve(fpr, tpr)
