#====================================================#
#                                                    #
#               CLASE TUTORIAL NÂ°11                  #
#             Componentes principales                #      
#                                                    #
#====================================================#

# Profesor: Walter Sosa Escudero
# Tutora: Carla Srebot 

# Referencias:
# James, Witten, Hastie y Tibshirani(2013) 
# Chaper 10 Lab 1: Unsupervised Learning    


# ---------->  PRINCIPAL COMPONENTS ANALYSIS (PCA)

# Cargamos los paquetes que necesitamos

library("ISLR")
library("ggplot2")
library("RColorBrewer")
library("corrplot")
library("pastecs")
#install.packages("pls") # Este paquete tiene PCR y PLS
library(pls)

#Base de arrestos en EEUU con sus motivos
fix(USArrests)
states =row.names(USArrests)
states

#Â¿Por quÃ© escalamos las variables?
pr.out =prcomp(USArrests , scale =TRUE)
#Con scale=TRUE, hacemos que el desvÃ­o sea igual a 1.

#Â¿QuÃ© informaciÃ³n guardÃ³?
names(pr.out)
#Â¿QuÃ© son scale y center? 
pr.out$center
pr.out$scale
#Veamos
stat.desc(USArrests$Murder) 
#Otro modo:
apply(USArrests , 2, mean)
apply(USArrests , 2, var)

#Para ver los loadings, usamos rotation:
pr.out$rotation
#Tomemos los loadings del primer componente principal
(-0.5358995)^2+(-0.5831836)^2+(-0.2781909)^2+(-0.5434321)^2 
#Â¿Por quÃ© igual a 1?

#x tiene los scores:
pr.out$x
#Tomemos Alabama:
(-0.5358995)*((13.2-7.788)/sqrt(18.97047))+(-0.5831836)*((236-170.76)/sqrt(6945.16571))+(-0.2781909)*((58-65.54)/sqrt(209.51878))+(-0.5434321)*((21.2-21.232)/sqrt(87.72916))

#Grafiquemos los primeros componentes principales:
biplot (pr.out , scale =0)

#Para ver el desvÃ­o estÃ¡ndar de cada componente principal:
pr.out$sdev

#Y para ver la varianza explicada por cada componente principal:
pr.var =pr.out$sdev^2
pr.var

#Pero, Â¿quÃ© proporciÃ³n de la varianza explica cada componente?
#Hay que dividir la varianza explicada por cada CP por la varianza total explicada
#por los cuatro componentes.
pve=pr.var/sum(pr.var )
pve

#Esto mismo lo podemos graficar:
plot(pve , xlab=" Principal Component ", ylab=" Proportion of Variance Explained ", ylim=c(0,1) ,type='b')
plot(cumsum (pve ), xlab=" Principal Component ", ylab ="Cumulative Proportion of Variance Explained ", ylim=c(0,1) ,
     type='b')


# ------------------------------------------------------------------------#
#         James, Witten, Hastie y Tibshirani(2013) Chaper 6 Lab 3:        #
#         Principal Component Regression and Partial Least Squares        #
# ------------------------------------------------------------------------#

# Ahora vamos a tratar de predecir los salarios de los beisbolistas, Â¡otra vez!.

# Veremos dos mÃ©todos para reducir la dimensiÃ³n de
# los coeficientes que se estimarÃ¡n. 

#            ANTES                          AHORA
#       ----------------               ------------------
#             MCO                        PCR - PLS
#        p+1 coeficientes               M+1 coeficientes
#
# con M<<p

# OBJETIVO: Reducir la varianza de los coeficientes -> reducir la varianza en la predicciÃ³n

# ----------> ORDENAMOS LA BASE
rm(list=ls())

# Usaremos la base de datos de jugadores de baseball
fix(Hitters)

# Eliminamos los missing values de la base de datos 
Hitters=na.omit(Hitters)

# GrÃ¡fico de Matriz: Â¿por quÃ© esos nÃºmeros?
corS<-cor(Hitters[,-c(14,15,20)])
corrplot.mixed(corS, tl.pos = "lt", tl.col = "black" )


# ---------->  PRINCIPAL COMPONENTS REGRESSION (PCR)

#  sintaxis: pcr(FORMULA, DATOS, scale= T...)
#  donde los argumentos son: FORMULA= y ~ x 
#                            DATOS
#                            scale= T      nos estandariza las variables
#                            validation = "CV"  10-fold cross-validation

set.seed(2) # Para poder reproducir los resultados

# Usamos PCR para predecir el salario
pcr.fit=pcr(Salary~., data=Hitters,scale=TRUE,validation="CV")

# Vemos los resultados
summary(pcr.fit)
#El primer CP xplica el 38% de la variabilidad de las variables explicativas y el 40% 
#de la variabilidad del salario. 
#El componente 19 explica 100% de la muestra, pero de la variabilidad del salario sÃ³lo el 54%.

# Obtenemos por CV la raÃ­z del ECM para modelos con 0 (constante) a 19 componentes.
# Para conocer el ECM hay que elevar cada valor de CV al cuadrado.

# Para conocer el ECM podemos usar la funciÃ³n MSEP() del paquete psl
msep=MSEP(pcr.fit)
msep

# TambiÃ©n podemos ver el "porcentaje de varianza explicada" de 
# los M componentes. Por ejemplo, 2 componentes (M=2) capturan
# el 60.16% de de toda la varianza o informaciÃ³n de los predictores
# y el 41.58% de la variabilidad del salario.  

# GrÃ¡fico del ECM para cada nÂ° de componente
validationplot(pcr.fit,val.type="MSEP", legendpos = "top")
#CV y CV ajustado
#MSEP: Mean Squared Error of Prediction

#Alternativa: plot(msep) 

# Para conocer el nÂ° de componentes con menor MSEP
which.min(msep$val[1,1,1:20])    

# Para ver los coeficientes cuando M=18
pcr.fit$coefficients[,,18]


# ------->  PCR usando el enfoque de validaciÃ³n para la elecciÃ³n 
#                      del nÂ° de componentes

x=model.matrix(Salary~.,Hitters)[,-1]
y=Hitters$Salary

set.seed(1) # para reproducir los resultados

# Base de entrenamiento
train=sample(1:nrow(x), nrow(x)/2)

# Base de prueba
test=(-train)

# Vector de respuesta para testear el error de predicciÃ³n
y.test=y[test]

set.seed(1) # Para poder reproducir los resultados

# Usamos PCR para predecir el salario
pcr.fit=pcr(Salary~., data=Hitters,subset=train,scale=TRUE, validation="CV")

# GrÃ¡fico del ECM para cada nÂ° de componente
validationplot(pcr.fit,val.type="MSEP")

# Para conocer el nÂ° de componentes con menor MSEP
msep=MSEP(pcr.fit)
which.min(msep$val[1,1,1:20]) 

# Ahora el modelo con 5 componentes presenta el menor de cross-validation

# Estimamos la predicciÃ³n en la base de prueba con 5 componentes
pcr.pred=predict(pcr.fit,x[test,],ncomp=5)

# Computamos ECM en la base de prueba con 5 componentes
mean((pcr.pred-y.test)^2)

# Â¿Realiza selecciÃ³n de variables como...? Â¿QuiÃ©n realiza selecciÃ³n de variables?

# EstimaciÃ³n por PCR con 5 componentes para toda la base
pcr.fit=pcr(y~x,scale=TRUE,ncomp=5)

# Vemos los resultados
summary(pcr.fit)

# El porcentaje de la varianza explicada de los salarios con 5 
# componentes es 44.90%.

# Para ver los coeficientes cuando usamos M=5
pcr.fit$coefficients[,,7]


# ---------->   PARTIAL LEAST SQUARES (PLS)
#Diferencia con lo anterior: se incorpora quÃ© queremos predecir. 
#No sÃ³lo busca usar la mÃ¡xima info posible, sino la que mÃ¡s sirve.

set.seed(1) # Para poder reproducir los resultados

# Usamos la funciÃ³n plsr() con la misma sintaxis que para pcr()
pls.fit=plsr(Salary~., data=Hitters,subset=train,scale=TRUE, validation="CV")

#Vemos los resultados
summary(pls.fit)
#La salida se interpreta igual que en PCR. Arriba: raÃ­z del error cuadrÃ¡tico medio.

# GrÃ¡fico del error de cross-validation
validationplot(pls.fit,val.type="MSEP", legendpos = "top")

# Para conocer el nÂ° de componentes con menor MSEP
msep=MSEP(pls.fit)
which.min(msep$val[1,1,1:20]) 

# Estimamos la predicciÃ³n en la base de prueba con 2 componentes
pls.pred=predict(pls.fit,x[test,],ncomp=2)

# Computamos ECM en la base de prueba con 2 componentes
mean((pls.pred-y.test)^2)
# El ECM en la base de testeo es ligeramente a PCR, ridge y Lasso

# EstimaciÃ³n por PLS con 2 componentes para toda la base
pls.fit=plsr(Salary~., data=Hitters,scale=TRUE,ncomp=2)

# Vemos los resultados
summary(pls.fit)

# El porcentaje de la varianza explicada de los salarios con 2 
# componentes es 46.40%, muy cercano a PCR pero levemente menor.
# Esto se debe a que PCR solo maximiza la varianza explicada en 
# los predictores, mientras que PLS maximiza la varianza 
# explicada tando en los predictores como en la respuesta.


# ------------------------------------------------------------------------#
# Para mÃ¡s referencias ver:
#     SecciÃ³n 6.3 de Introdution to Statistical Learning. James et al (2015)
#     SecciÃ³n 3.5 de Elements of Statistical Learning Hastie (2008) 
#     Paper Mevik & Wehrens (2007): https://www.jstatsoft.org/article/view/v018i02
# ------------------------------------------------------------------------#
