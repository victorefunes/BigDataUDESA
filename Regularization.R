#==================================================#
#                                                  #
#               CLASE TUTORIAL NÂ°7                 #
#                 Ridge y LASSO                    #
#                                                  #
#==================================================#

# Profesor: Walter Sosa Escudero
# Tutora: Carla Srebot 

# Referencias:
  # James, Witten, Hastie y Tibshirani(2013) Chaper 6 Lab 1 y 2:    
  # Subset Selection Methods, Ridge regression and Lasso        


# Cargamos los paquetes que necesitamos

library(MASS)
library(ISLR)
library(ggplot2)
library(class)

library(Matrix) # Estos dos paquetes se necesitan para el paquete glmnet
library(foreach)

install.packages("glmnet") # Este paquete tiene la funciones de ridge y lasso
library(glmnet)


# ----------> ORDENAMOS LA BASE

# Usaremos la base de datos de jugadores de baseball
fix(Hitters)
?Hitters
names(Hitters)
dim(Hitters) # dimension original: 322 20

# Queremos predecir el salario (Salary) de los jugadores en funciÃ³n de estadisticas 
# Â¿QuÃ© columnas tienen missing values?
list_na <- colnames(Hitters)[ apply(Hitters, 2, anyNA)]

sum(is.na(Hitters$Salary))  # is.na() nos cuenta las observaciones missing en salario

# Eliminamos los missing values de la base de datos 
Hitters=na.omit(Hitters)
dim(Hitters)  # nueva dimensiÃ³n: 263 20
sum(is.na(Hitters)) # chequeamos que no haya mÃ¡s missing values
sum(is.na(Hitters$Salary))

# O podrÃ­a asignares la mediana
# mediana=median(Hitters$Salary, na.rm=TRUE)
# Hitters$Salary[is.na(Hitters$Salary)] <- mediana


# ---------->  METODOS DE REGULARIZACIÃ“N

# Guardamos los predictores y la variable de respuesta como matrices y vectores:
# Â¿Para quÃ© sirve model.matrix? Para transformar a las posibles variables cualitativas en 
# variables dummy. Esto es importante para que no falle glmnet(), que solo toma variables
# cuantitativas.

x=model.matrix(Salary~.,Hitters)[,-1]
y=Hitters$Salary


# ---------->  REGRESION RIDGE

# Usaremos la funciÃ³n glmnet() para Ridge y Lasso

# sintaxis: glmnet(x, y, alpha= ... , lambda = ...)
# donde:      x      Matriz de Predictores (SI O SI TODOS CUANTITATIVOS)
#             Y      Vector con la variable de respuesta
#             alpha  indica el modelo:
#               - si es =0  ->  RIDGE REGRESSION
#               - si es =1  ->  LASSO 
#             lambda:  valores que puede tomar lambda


# LAMBDA: es un parÃ¡metro de complejidad que controla la "cantidad de reducciÃ³n" de los betas.
# A mayor lambda, mayor disminuciÃ³n. 

# PREGUNTA: Â¿quÃ© pasa si lambda tiende a infinito?

# Nota: la funciÃ³n glmnet() estandariza las variables. Esto se puede
# cambiar con la opciÃ³n standarize=F (OJO con cambiar esta opciÃ³n:
# si los predictores NO estan estandarizados los coeficientes de 
# ridge cambian mucho por la escala de los mismos)

# Primero, indicamos una secuencia de valores que puede tomar lambda. En realidad, glmnet()
# corre el modelo en un rango de lambda predeterminado, pero en este caso definimos una serie
# de valores de este parÃ¡metro para que vaya a los extremos (desde sÃ³lo el intercepto hasta MCO).
grid=10^seq(10,-2,length=100) 

# EstimaciÃ³n del modelo de regresiÃ³n Ridge
ridge.mod=glmnet(x,y,alpha=0,lambda=grid)

# Para un lambda dado, hay un vector de coeficientes B_ridge
# PREGUNTA: Â¿quÃ© dimensiÃ³n tiene la matriz de coeficientes? Â¿Por quÃ©?
dim(coef(ridge.mod)) # dimensiÃ³n de la matriz con los betas para cada Lambda

# Veamos los coeficientes para dos lambdas distintos:
# A) Lambda "grande" = 11497.57
ridge.mod$lambda[50]

# Sus coeficientes (asociados al valor de lambda que fijÃ©):
coef(ridge.mod)[,50]

# l2 norm para este lambda:
# Â¿QUÃ‰ ES L2 NORM?
# Es una medida de la distancia de los coeficientes a cero. 

# Cuando aumenta lambda, l2 norm... Â¿Aumenta o cae?
sqrt(sum(coef(ridge.mod)[-1,50]^2))

# PREGUNTA: cÃ³mo esperamos que sean las estimaciones de los coeficientes con un lambda mucho mayor?

# B) Lambda "chico" =  705.4802

ridge.mod$lambda[60]

# Sus coeficientes:
coef(ridge.mod)[,60]

# l2 norm: Â¿va a ser mayor o menor al anterior?
sqrt(sum(coef(ridge.mod)[-1,60]^2))


# Podemos computar los coeficientes para otros valores de lambda que no consideramos 
# usando la funciÃ³n predict()        
predict(ridge.mod,s=50,type="coefficients")[1:20,]
# "s" es el valor que le queremos dar a lambda 


# --------> Enfoque de validaciÃ³n para la RegresiÃ³n Ridge

set.seed(1) # para reproducir los resultados

# Base de entrenamiento
train=sample(1:nrow(x), nrow(x)/2)

# Base de prueba
test=(-train)

# Vector de respuesta para testear el error de prediccion
y.test=y[test]

# EstimaciÃ³n del Modelo de RegresiÃ³n de Ridge para el entrenamiento
ridge.mod=glmnet(x[train,],y[train],alpha=0,lambda=grid)

# GrÃ¡fico de los posibles coeficientes para distintos lambda
plot(ridge.mod,xvar="lambda", label=TRUE)

# PredicciÃ³n para lambda=4 y las observaciones de prueba 
ridge.pred=predict(ridge.mod,s=4,newx=x[test,])

# ECM de validaciÃ³n
mean((ridge.pred-y.test)^2)

# CASO EXTREMO: lambda -> infinito, entonces B_Ridge -> 0
# Si B_Ridge -> 0, Â¿quÃ© predecimos para cada observaciÃ³n?
ridge.pred=predict(ridge.mod,s=1e10,newx=x[test,])

# ECM de validaciÃ³n
mean((ridge.pred-y.test)^2)

# Alternativa:
# ECM de validaciÃ³n para el caso extremo con solo la constante es
# equivalente a:
mean((mean(y[train])-y.test)^2)

# CASO EXTREMO: lambda -> 0, entonces B_Ridge -> B_MCO  
ridge.pred=predict(ridge.mod,s=0,newx=x[test,])

# ECM de validacion
mean((ridge.pred-y.test)^2)
# Vemos que con lambda=4 tenÃ­amos menor error de predicciÃ³n 
# en la base de prueba que en el caso extremo con solo el intercepto. 
# Ergo, para mejorar la predicciÃ³n sesgamos los coeficientes de MCO.

# Chequeamos los coeficientes de MCO
coef(lm(y~x, subset=train))

# Comparamos con los coeficientes de Ridge con lambda=0
predict(ridge.mod,s=0,type="coefficients")[1:20,]  


# --------> 10-fold CROSS-VALIDATION para elegir lambda

set.seed(1) # para reproducir los resultados

# El paquete glmnet viene con su propia funcion de CV para 10 grupos de 
# datos aleatoriamente seleccionados
cv.out=cv.glmnet(x[train,],y[train],alpha=0)

# GrÃ¡fico del ECM y ln de lambda
plot(cv.out)
# Una de las lÃ­neas indica el valor de lambda con menor MSE, la otra el mayor valor de
# lambda cuyo MSE estÃ¡ a un error estÃ¡ndar del menor MSE.

# Lambda Ã³ptimo
bestlam=cv.out$lambda.min
bestlam
# El lambda que resulta en el menor error de cross validation es 326.0828.
# PREGUNTA: cuÃ¡ntas variables selecciona?

# El ECM para lambda=326.0828 es:
ridge.pred=predict(ridge.mod,s=bestlam,newx=x[test,])
mean((ridge.pred-y.test)^2)

# ESTIMAMOS EL MODELO DE REGRESION RIDGE PARA TODOS LOS DATOS CON lambda=326.0828
out=glmnet(x,y,alpha=0)
predict(out,type="coefficients",s=bestlam)[1:20,]
# NingÃºn coeficiente se iguala a cero y por lo tanto Ridge no nos
# selecciona variables



# ----------> LASSO: Least Absolute Shrinkage and Selection Operator

# Como dijimos antes, con alpha=1 estimamos LASSO
lasso.mod=glmnet(x[train,],y[train],alpha=1,lambda=grid)

# GrÃ¡fico de los posibles coeficientes para distintos lambda
plot(lasso.mod,xvar="lambda", label=TRUE)
# Algunas variables son cero dependiendo del parametro de complejidad


# --------> 10-fold CROSS-VALIDATION para elegir lambda

set.seed(1) # para reproducir los resultados

# Usamos la misma funcion de CV del paquete glmnet
cv.out=cv.glmnet(x[train,],y[train],alpha=1)

# GrÃ¡fico del ECM y log de lambda
plot(cv.out)

# Lambda Ã³ptimo
bestlam=cv.out$lambda.min
bestlam
# El lambda que resulta en el menor error de cross validation es 5.83

# El ECM para lambda=5.83 es:
lasso.pred=predict(lasso.mod,s=bestlam,newx=x[test,])
mean((lasso.pred-y.test)^2)

# ESTIMAMOS EL MODELO LASSO PARA TODOS LOS DATOS CON lambda=5.83
out=glmnet(x,y,alpha=1,lambda=grid)

# Computamos los coeficientes para dicho lambda
lasso.coef=predict(out,type="coefficients",s=bestlam)[1:20,]
round(lasso.coef,2)
# A diferencia de Ridge, 6 de las 19 variables tienen coeficientes
# iguales a cero

# VARIABLES SELECCIONADAS POR LASSO USANDO 10-fold CV son:
round(lasso.coef[lasso.coef!=0],2)

# ------------------------------------------------------------------------#

# ALTERNATIVA: para la selecciÃ³n del mejor lambda

# Estimamos sin imponer posibles valores de lambda:
lasso.tr=glmnet(x[train,],y[train])

# Veamos los elementos de la estimaciÃ³n por LASSO
lasso.tr

# Predecimos con una nueva base
pred=predict(lasso.tr,x[-train,])

# Calculamos el error cuadrÃ¡tico medio
rmse= sqrt(apply((y[-train]-pred)^2,2,mean))

# Graficamos  
plot(log(lasso.tr$lambda),rmse,type="b",col="red",xlab="Log(lambda)")

# Buscamos el mejor lambda
lam.best=lasso.tr$lambda[order(rmse)[1]]
lam.best

# Vemos los coeficientes seleccionados  
coef(lasso.tr,s=lam.best)

# notar las diferencias con el mÃ©todo anterior


# ------------------------------------------------------------------------#

# --------------------------- AUN HAY MÃS! -------------------------------#
#                       Subset Selection Methods                          #
# ------------------------------------------------------------------------#

# Cargamos los paquetes que necesitamos

install.packages("leaps")    
library(leaps) # Este paquete tiene la funcion regsubsets()

# ---------->  BEST SUBSET SELECTION (BSS)

#  sintaxis: regsubsets(FORMULA, DATOS,...)
#  donde los argumentos son: FORMULA= y ~ x 
#                            DATOS

regfit.full=regsubsets(Salary~.,Hitters)
summary(regfit.full)

# InterpretaciÃ³n: El mejor modelo con 1, 2, 3, ... predictores 
#                 es el que tiene las variables con *.

# Si queremos ver modelos con mÃ¡s variables usamos la opciÃ³n nvmax  
regfit.full=regsubsets(Salary~.,data=Hitters,nvmax=19)
summary(regfit.full)
reg.summary=summary(regfit.full)

# Seleccionamos el mejor modelo segÃºn R^2, RSS (Suma de Residuos al Cuadrado), etc.  
# Pero para eso tengo que saber cÃ³mo le llama el comando a cada cosa!
names(summary(regfit.full))

# Supongamos que nos interesa R^2
reg.summary$rsq
# Interpretacion: El estadÃ­stico R^2 crece monÃ³tonamente a medida que 
#                 incorporamos mÃ¡s predictores en el modelo.

# GrÃ¡fico de la RSS y nÃºmero de variables
ggplot(data=data.frame(1:19,reg.summary$rss), aes(x=1:19, y=reg.summary$rss)) +
  geom_line(colour="#660066") +
  geom_point(colour="#660066", size=3) +
  ggtitle("RSS y nÂ° de variables") +
  labs(x = "Number of Variables", y = "RSS") 

# GrÃ¡fico de la R^2 ajustado y nÃºmero de variables
ggplot(data=data.frame(1:19,reg.summary$adjr2), aes(x=1:19, y=reg.summary$adjr2)) +
  geom_line(colour="#660066") +
  geom_point(colour="#660066", size=3) +
  ggtitle("R^2 ajustado y nÂ° de variables") +
  labs(x = "Number of Variables", y = "R^2 ajustado") 

# Usamos which.max() para ver el modelo con mayor R^2 ajustado
which.max(reg.summary$adjr2)

# GrÃ¡fico de la R^2 ajust y nÃºmero de variables, con el modelo destacado
ggplot(data=data.frame(1:19,reg.summary$adjr2), aes(x=1:19, y=reg.summary$adjr2)) +
  geom_line(colour="#660066") +
  geom_point(colour=ifelse(1:19==11, "red", "#660066"), size=3) +
  ggtitle("R^2 ajustado y nÂ° de variables") +
  labs(x = "Number of Variables", y = "R^2 ajustado") 

# Usamos which.min() para ver el modelo con menor Cp
which.min(reg.summary$cp)

# GrÃ¡fico de Cp y nÃºmero de variables, con el modelo destacado
ggplot(data=data.frame(1:19,reg.summary$cp), aes(x=1:19, y=reg.summary$cp)) +
  geom_line(colour="#660066") +
  geom_point(colour=ifelse(1:19==10, "red", "#660066"), size=3) +
  ggtitle("Cp y nÂ° de variables") +
  labs(x = "Number of Variables", y = "Cp")

# InterpretaciÃ³n: El estadÃ­stico Cp penaliza a la SRC por la cantidad de 
#                 predictores utilizados y nos da una mejor idea sobre el
#                 ECM de prueba.

# Usamos which.min() para ver el modelo con menor bic
which.min(reg.summary$bic)

# GrÃ¡fico de BIC y nÃºmero de variables, con el modelo destacado
ggplot(data=data.frame(1:19,reg.summary$bic), aes(x=1:19, y=reg.summary$bic)) +
  geom_line(colour="#660066") +
  geom_point(colour=ifelse(1:19==6, "red", "#660066"), size=3) +
  ggtitle("BIC y n? de variables") +
  labs(x = "Number of Variables", y = "BIC")

# Variables seleccionadas para el mejor modelo por criterio
plot(regfit.full,scale="r2")
plot(regfit.full,scale="adjr2")
plot(regfit.full,scale="Cp")
plot(regfit.full,scale="bic")
#Las que estÃ¡n en negro son las seleccionadas. 

# Para ver los coeficientes del mejor modelo segun BIC
coef(regfit.full,6)


# ---------->  FORWARD AND BACKWARD STEPWISE SELECTION

#Utilizamos la opciÃ³n method para indicar el metodo de selecciÃ³n de variables   

regfit.fwd=regsubsets(Salary~.,data=Hitters,nvmax=19,method="forward")
summary(regfit.fwd)

regfit.bwd=regsubsets(Salary~.,data=Hitters,nvmax=19,method="backward")
summary(regfit.bwd)

# InterpretaciÃ³n: Al igual que Best Subset Selection, tanto con FW como con BW
# el mejor modelo con un predictor es el que incluye la variable CRBI.
# CRBI: Number of runs batted in during his career

# A partir del modelo con 7 predictores, la selecciÃ³n de variables cambia:
coef(regfit.full,7)
coef(regfit.fwd,7)
coef(regfit.bwd,7)


# ---------->  ELECCIÃ“N DEL MODELO USANDO EL ENFOQUE DE VALIDACIÃ“N

set.seed(1) # Para poder reproducir los resultados

# Base de entrenamiento 
train=sample(c(TRUE,FALSE), nrow(Hitters),rep=TRUE)

# Base de prueba 
test=(!train)

# Best Subset selection  
regfit.best=regsubsets(Salary~.,data=Hitters[train,],nvmax=19)

# FunciÃ³n model.matrix para guardar los datos de la 
# base de validaciÃ³n en formato de matriz. 
test.mat=model.matrix(Salary~.,data=Hitters[test,]) # Notar la base test
View(test.mat) # vemos esta matriz particular (ver notacion matricial de clases)

# VENTAJA: crea automÃ¡ticamente las dummies para variables cualitativas    

# Creo el vector donde guardaremos el ECM de cada uno de los 19 mejores
# modelos  
val.errors=rep(NA,19)

# Loop: 
# I) Para el i-esimo modelo guardo los coeficientes de las variables
# seleccionadas en un vector B_train.
# II) Para el i-esimo modelo hago la predicciÃ³n 
#               Y_estimado.test= X_test * B_train
# III) Computo ECM

for(i in 1:19){
  coefi=coef(regfit.best,id=i)
  pred=test.mat[,names(coefi)]%*%coefi
  val.errors[i]=mean((Hitters$Salary[test]-pred)^2)
}

# Vemos el ECM de validaciÃ³n en los 19 mejores modelos segun best subset selection
val.errors

# Buscamos al de menor error cuadrÃ¡tico medio
which.min(val.errors) # El modelo con 10 variables es el mejor
coef(regfit.best,10) # Vemos sus coeficientes


# ----------------------------------------- #  
# FUNCION predict para regsubsets() 

predict.regsubsets=function(object,newdata,id,...){
  form=as.formula(object$call[[2]]) # extrae la fÃ³rmula utilizada
  mat=model.matrix(form,newdata)    # crea la matriz X_test
  coefi=coef(object,id=id)          # crea el vector B_train
  xvars=names(coefi)                # nombres de las variables seleccionadas
  mat[,xvars]%*%coefi               # predicciÃ³n: Y_estimado.test= X_test * B_train
}

# Esta funciÃ³n nos sirve para cuando hagamos CV 
# ----------------------------------------- #  


# REESTIMAMOS CON BEST SUBSET SELECCIÃ“N Y TODAS LAS OBSERVACIONES
regfit.best=regsubsets(Salary~.,data=Hitters,nvmax=19)

# Por lo anterior, sabemos que el mejor modelo es aquel con 10 variables
coef(regfit.best,10)

# En sÃ­ntesis, las varibles del mejor modelo (de entre los 19) en 
# el entrenamiento, pueden diferir de las variables del mejor modelo
# (de entre los 19) usando TODAS las observaciones

# ---------->  ELECCION DEL MODELO USANDO CROSS-VALIDACION 

k=10 # nÃºmero de grupos

set.seed(1) # Para poder reproducir los resultados

# Vector con los 10 grupos de observaciones aleatorias
folds=sample(1:k,nrow(Hitters),replace=TRUE)
table(folds) # cantidad de observaciones por grupo

# Creamos la matriz donde guardaremos los resultados 
cv.errors=matrix(NA,k,19, dimnames=list(NULL, paste(1:19)))
# con dimnames indico los nombres por filas y por columnas

# Loop:
# I) Estimo BSS usando todas las observaciones, MENOS aquellas en el 
# j-Ã©simo grupo
# II) Para el j-Ã©simo grupo de observaciones, estimo el modelo usando i 
# observaciones de prueba
# III) Computo ECM y lo guardo en el elemento (j,i) de la matriz

for(j in 1:k){
  best.fit=regsubsets(Salary~.,data=Hitters[folds!=j,],nvmax=19)
  for(i in 1:19){
    pred=predict(best.fit,Hitters[folds==j,],id=i)
    cv.errors[j,i]=mean( (Hitters$Salary[folds==j]-pred)^2)
  }
}

# PROMEDIO DEL ERROR DE CV 

# Usamos la funciÃ³n apply() para sacar el promedio de los ECM entre
# los j grupos de prueba para cada uno de los modelos
# con i variables.

# sintÃ¡xis de apply(X, MARGIN, FUN, ...)
#     donde: X la matriz 
#            MARGIN indica a quiÃ©nes les aplica la funciÃ³n 
#               1      indica por fila
#               2      indica por columna
#               c(1,2) indica por fila y por columna
#            FUN indica la funciÃ³n que se aplica

mean.cv.errors=apply(cv.errors,2,mean) 
mean.cv.errors # chequeamos
which.min(mean.cv.errors) # buscamos el menor

# GrÃ¡fico PROMEDIO DEL ERROR DE CV 
ggplot(data=data.frame(1:19,mean.cv.errors), aes(x=1:19, y=mean.cv.errors)) +
  geom_line(colour="#660066") +
  geom_point(colour=ifelse(1:19==11, "red", "#660066"), size=3) +
  ggtitle("Promedio del error de CV y nÃºmero de variables") +
  labs(x = "NÃºmero de variables", y = "Promedio del error de CV") 

# REESTIMAMOS CON BEST SUBSET SELECCIÃ“N Y TODAS LAS OBSERVACIONES
regfit.best=regsubsets(Salary~.,data=Hitters,nvmax=19)

# Por lo anterior, sabemos que el mejor modelo es aquel con 11 variables
coef(regfit.best,11)

# ------------------------------------------------------------------------#
# ------------------------------------------------------------------------#
