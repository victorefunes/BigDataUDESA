#==================================================#
#                                                  #
#               CLASE TUTORIAL NÂ°9                 #
#       Bagging, Random Forests & Boosting         #
#                                                  #
#==================================================#

# Profesor: Walter Sosa Escudero
# Tutora: Carla Srebot 

# Referencias:
# James, Witten, Hastie y Tibshirani(2013) Chaper 8 Lab
# CART, Bagging, Random Forest, Boosting Conditional Tree     

# Directorio
setwd("C:\\Users\\CARLA SREBOT\\Google Drive\\UdeSA\\Tutoriales - UdeSA\\Primavera 2020 Big Data\\Tutoriales\\Tutorial 9")


# Cargamos los paquetes que necesitamos

library(ISLR)
library(MASS)
library(ggplot2)

#install.packages("corrplot") #Paquete para matriz de correlaciones fachera
library(corrplot)

#install.packages("RColorBrewer") 
library(RColorBrewer)

#install.packages("tree") # paquete CART 
library(tree)

#install.packages("rpart") # otro paquete con las funciones de CART
library(rpart)

# PAQUETES PARA HACER LINDOS GRÃFICOS DE ÃRBOLES
#install.packages("rattle")
#install.packages("rpart.plot")
#install.packages("RGtk2")

library(rattle)
library(rpart.plot)
library(RGtk2)

# install.packages("randomForest") 
library(randomForest)

# install.packages("gbm") # Paquete de boosting
library(gbm)

# install.packages("party") # Paquete con Conditional Trees
library(party)


# ------------ > BAGGING

# Es un caso especial de Random Forest con m=p

# Semilla para reproducir los resultados          
set.seed(1)

# Usaremos la base Boston 
fix(Boston)

# EstimaciÃ³n por Bagging: con mtry=13 le indicamos que use todos los predictores
train = sample(1:nrow(Boston), nrow(Boston)/2)
bag.boston=randomForest(medv~.,data=Boston,subset=train,mtry=13,importance=TRUE)

# Vemos los resultados  
bag.boston

# Estimamos la predicciÃ³n por bagging en el conjunto de prueba
yhat.bag = predict(bag.boston,newdata=Boston[-train,])
boston.test=Boston[-train,"medv"]

# Graficamos  
plot(yhat.bag, boston.test)
abline (0,1)

# o bien:
ggplot(data.frame(boston.test,yhat.bag), aes(x=yhat.bag, y=boston.test)) +
  geom_point(shape=1) +   
  geom_smooth(method=lm, se= F)   

# ECM de bagging
ECM4=mean((yhat.bag-boston.test)^2)

# Comparamos con los errores de los Ã¡rboles anteriores
list(round(c(ECM1,ECM2,ECM3, ECM4),2))
# Aguante Bagging no me importa nada!!

# Por default estima 500 Ã¡rboles, si cambiamos el nÃºmero de Ã¡rboles a 25:
bag.boston=randomForest(medv~.,data=Boston,subset=train,mtry=13,ntree=25,do.trace=T)
#con do.trace vemos como cae el ECM OBB lo podemos graficar de la siguiente manera
plot(bag.boston,main="Error de los Ã¡rboles - Bagging",lwd=c(2,2,2),lty=c(1,2,2),col = c("blue","blue","blue"),type = c("l","l","l"))

yhat.bag = predict(bag.boston,newdata=Boston[-train,])
ECM5=mean((yhat.bag-boston.test)^2)
list(round(c(ECM1,ECM2,ECM3, ECM4, ECM5),2))


# ------------ > RANDOM FOREST

# Estimamos usando RF con un nÃºmero aleatorio de variables a usar en cada nodo
set.seed(1)
rf.boston=randomForest(medv~.,data=Boston,subset=train,mtry=6,importance=TRUE)
# Si no aclaramos mtry=6, elige solito 4.

# Estimamos la predicciÃ³n de prueba de RF
yhat.rf = predict(rf.boston,newdata=Boston[-train,])

# ECM de RF
ECM6=mean((yhat.rf-boston.test)^2)
list(round(c(ECM1,ECM2,ECM3, ECM4, ECM5,ECM6),2))
# Vemos que tiene menor error de predicciÃ³n que los Ã¡rboles y bagging (esto se debe 
# a que random forest calcula Ã¡rboles descorrelacionados)

# IMPORTANACIA DE LOS FACTORES
importance(rf.boston)

# Graficamos las 5 variables mÃ¡s importantes
varImpPlot(rf.boston, n.var =  5, pt.cex=1.5, 
           pch=15,lty="dotted",lcolor="blue",cex.lab=2, main="")
# Vemos que las dos variables mÃ¡s importantes son el nivel de riqueza "lstat" y 
# el tamaÃ±o de la casa "rm"

# ------------ > BOOSTING

# Estimamos  
set.seed(1)
boost.boston=gbm(medv~.,data=Boston[train,],distribution="gaussian",n.trees=5000,interaction.depth=4)
# n.trees=5000 indica que queremos 5000 Ã¡rboles;
# interaction.depth=4 limita la profundidad de cada Ã¡rbol (el nivel mÃ¡ximo de
# interacciones permitidas; por default es =1).

# En este caso la funciÃ³n summary() grafica la influencia relativa y el estadÃ­stico
summary(boost.boston)
# Vemos nuevamente que lstat y rm son las mÃ¡s importantes para predecir

#  PARTIAL PLOTS de las dos variables mÃ¡s importantes
par(mfrow=c(1,2))
plot(boost.boston,i="rm", col= "blue")
plot(boost.boston,i="lstat",col= "blue")
par(mfrow=c(1,1))

# INTERPRETACIÃ“N: En estos grÃ¡ficos vemos el efecto marginal de cada variable en la
# variable de respuesta. En nuestro caso, el precio de las casas aumenta con el
# tamaÃ±o de las casas "rm" y disminuye con el nivel socioeconÃ³mico bajo "lstat"

# PREDICCIÃ“N de BOOSTING
yhat.boost=predict(boost.boston,newdata=Boston[-train,],n.trees=5000)

# ECM Boosting
ECM7=mean((yhat.boost-boston.test)^2)

# Comparamos
list(round(c(ECM1,ECM2,ECM3, ECM4, ECM5,ECM6,ECM7),1))
# Vemos que el error es muy cercano al de RF

# Se puede modificar la penalidad de boosting con la opciÃ³n shrinkage (por default es =0.1)
# (shrinkage=learning rate; a menor tasa, se necesitan mÃ¡s Ã¡rboles)
boost.boston=gbm(medv~.,data=Boston[train,],distribution="gaussian",n.trees=5000,interaction.depth=4,shrinkage=0.2,verbose=F)
# PredicciÃ³n
yhat.boost=predict(boost.boston,newdata=Boston[-train,],n.trees=5000)
# ECM Boosting con lambda=0.2
ECM8=mean((yhat.boost-boston.test)^2)
# Comparamos
list(round(c(ECM1,ECM2,ECM3, ECM4, ECM5,ECM6,ECM7,ECM8),1))
# Da un toque mejor.


# Material extra:

# ------------ > CONDITIONAL TREE

# REFERENCIA: https://cran.r-project.org/web/packages/party/vignettes/party.pdf

# Estimamos un Ã¡rbol condicional:
set.seed(1)      
ctree.boston=ctree(medv~.,data=Boston[train,])     

# Graficamos el Ã¡rbol condicional  
plot(ctree.boston, main="Conditional Inference Tree for medv")
# Podemos ver que lstat y rm son las primeras variables y las mÃ¡s importantes 
# en la predicciÃ³n del precio de las casas.
# El grÃ¡fico nos realiza un box plot en cada "hoja" para entender cÃ³mo es 
# el precio de las casas en cada nodo terminal.

# PredicciÃ³n
yhat.ctree=predict(ctree.boston,newdata=Boston[-train,],n.trees=5000)

# ECM ctree con lambda=0.2
ECM9=mean((yhat.ctree-boston.test)^2)

# Comparamos
list(round(c(ECM1,ECM2,ECM3, ECM4, ECM5,ECM6,ECM7,ECM8,ECM9),1))
# Podemos ver que tiene menor error cuadrÃ¡tico medio que los Ã¡rboles, pero mayor a
# los estimados por bagging, RF y boosting.

# ------> RF con conditional trees

# Estimamos un RF con Ã¡rboles condicionales
set.seed(1)
cforest.boston=cforest(medv~.,data=Boston[train,])

# TambiÃ©n podemos ver la importancia de las variables
varimp(cforest.boston)

# GrÃ¡ficamente
dotchart(varimp(cforest.boston),pt.cex=1.5, 
         pch=15,lty="dotted",lcolor="blue",cex.lab=2, main="")

# PredicciÃ³n
yhat.cforest=predict(cforest.boston,newdata=Boston[-train,])

# ECM cforest con lambda=0.2
ECM10=mean((yhat.cforest-boston.test)^2)

# Comparamos
list(round(c(ECM1,ECM2,ECM3, ECM4, ECM5,ECM6,ECM7,ECM8,ECM9,ECM10),1))
# Vemos que el ECM de RF con Ã¡rboles condicionales no es tan chico como Bagging, 
# RF, Boosting

# ------------------------------------------------------------------------#
