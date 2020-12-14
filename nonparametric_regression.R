====================================================#
#                                                    #
#               CLASE TUTORIAL NÂ°10                  #
# Estimaciones no paramÃ©tricas y Density Regression  #
#                                                    #
#====================================================#

# Profesor: Walter Sosa Escudero
# Tutora: Carla Srebot 

# Cargamos los paquetes que necesitamos
# install.packages("ISLR")
# install.packages("ggplot2")
library(ISLR)
library(ggplot2)

# install.packages("ks") #con este paquete estimaremos el h Ã³ptimo
library(ks)


# ----------> ORDENAMOS LA BASE

# Usaremos la base de datos de jugadores de baseball (otra vez)
fix(Wage)

# Eliminamos los missing values de la base de datos 
Wage=na.omit(Wage)

# ----------> ESTIMACIÃ“N DE KERNELS GAUSSIANO, RECTANGULAR Y TRIANGULAR

# OBJETIVO: Encontrar la distribuciÃ³n del ingreso en la base de datos Wage

w <- Wage$wage
logw <- Wage$logwage

# Vamos a crear un grÃ¡fico en el que podamos comparar cada mÃ©todo:
layout(matrix(1:6, nrow = 2,ncol = 3))

# Histograma
hist(w, xlab = "Salario", ylab = "Frequency",
     probability = TRUE, main = "Gaussian kernel",
     border = "gray")
# Kernel gaussiano: usamos la funciÃ³n density
lines(density(w, kernel = "gaussian"), lwd = 2) #lwd: ancho de la lÃ­nea
rug(w) # las lÃ­neas verticales muestran la cantidad de puntos usados en la estimaciÃ³n


hist(logw, xlab = "Log Salario", ylab = "Frequency",
     probability = TRUE, main = "Gaussian kernel",
     border = "gray")
lines(density(logw, kernel = "gaussian"), lwd = 2)
rug(logw)

#-----------------------------------------------------------#
# Histograma
hist(w, xlab = "Salario", ylab = "Frequency",
     probability = TRUE, main = "Rectangular kernel",
     border = "gray")
# Kernel rectangular: usamos la opciÃ³n window
lines(density(w, window = "rectangular"), lwd = 2)
rug(w)


hist(logw, xlab = "Log Salario", ylab = "Frequency",
     probability = TRUE, main = "Rectangular kernel",
     border = "gray")

lines(density(logw, window = "rectangular"), lwd = 2)
rug(logw)
#-----------------------------------------------------------#
# Histograma
hist(w, xlab = "Salario", ylab = "Frequency",
     probability = TRUE, main = "Triangular kernel",
     border = "gray")
# Kernel Triangular: usamos la opciÃ³n window
lines(density(w,window = "triangular"), lwd = 2)
rug(w)


hist(logw, xlab = "Log Salario", ylab = "Frequency",
     probability = TRUE, main = "Triangular kernel",
     border = "gray")
lines(density(logw,window = "triangular"), lwd = 2)
rug(logw)


dev.off()

# Estos datos presentan una particularidad: parecen ser BImodales: una segunda campana
# pequeÃ±a en valores altos del ingreso

# ver: Charpentier & Flachaire(2014) sobre la aplicaciÃ³n logarÃ­tmica en la estimaciÃ³n de kernels
#          http://web5.uottawa.ca/ssms/vfs/.horde/eventmgr/001774_001412597609_Flachaire.pdf


# ObtenciÃ³n de los valores la estimaciÃ³n de densidad de kernels
density.wage = density(w, kernel = "gaussian")
density.logwage = density(logw, kernel = "gaussian")

# Numero de observaciones utilizadas en la estimaciÃ³n:
n.density1 = density.wage$n
n.density2 = density.logwage$n

# bandwidth utilizados en el plot
bw.density1 = density.wage$bw
bw.density2 = density.logwage$bw

 
# ----------> ELECCIÃ“N DEL BANDWIDTH

# La funciÃ³n density utilizÃ³ los siguientes valores de bandwith:
bw.density1
bw.density2

# 1) Una opciÃ³n es ir probando visualmente:

layout(matrix(1:3, ncol = 3))

hist(logw, xlab = "Log Salario", ylab = "Frequency",
     probability = TRUE, main = "Gaussian kernel - bw = 0.05",
     border = "gray")
lines(density(logw, bw = 0.05), lwd = 2)
rug(logw)

#----------------------------------------------------------

hist(logw, xlab = "Log Salario", ylab = "Frequency",
     probability = TRUE, main = "Gaussian kernel - bw = 0.01",
     border = "gray")
lines(density(logw, bw = 0.01))
rug(logw)

#----------------------------------------------------------

hist(logw, xlab = "Log Salario", ylab = "Frequency",
     probability = TRUE, main = "Gaussian kernel - bw = 1 ",
     border = "gray")
lines(density(logw, bw = 1), lwd = 2)
rug(logw)

dev.off()


# 2) Utilizar un mÃ©todo de selecciÃ³n del paquete ks:  

# Para profundizar en la teorÃ­a de cada mÃ©todo ver: 

# Duong, T. (2007). ks: Kernel density estimation and kernel discriminant analysis for 
# multivariate data in R. Journal of Statistical Software, 21(7), 1-16.

# Cross-validation selectors
# 1) Smoothed cross-validation (SCV) bandwidth selector: 
h1<-hscv(logw)

# 2) Plug-in bandwidth selector: es el mÃ©todo que usa la funciÃ³n density()
h2<-hpi(logw)

# Libro de referencia de este mÃ©todo de seleciÃ³n: Wand, M.P. & Jones, M.C. (1994) Multivariate plugin bandwidth selection. Computational Statistics. 9, 97-116.

dev.off()
# Observamos los resultados:

hist(logw, xlab = "Log Salario", ylab = "Frequency",
     probability = TRUE, main = "Kernels para distintos h Ã³ptimos",
     border = "gray")
lines(density(logw), lwd = 2)
lines(density(logw, bw = h1), lty = 2, col=4) #lty: tipo de lÃ­nea
lines(density(logw, bw = h2), col=2)
legend("topright", 
       c("original", "por SCV","por Plug-in"), 
       lty=c(1, 1, 2), 
       col=c("black","red","green"),
       cex = 0.6)



# ----------> Nadaraya-Watson kernel regression estimate

# Utilizamos la funciÃ³n ksmooth

#  sintaxis: ksmooth(x, Y, "normal", bandwidth)
#  Nota: La variable x solo puede ser numÃ©rica

with(Wage, {
  plot(age, logwage, cex=.5,col="darkgrey", main = "RegresiÃ³n de kernel de Nadaraya-Watson")
  lines(ksmooth(age, logwage, "normal", bandwidth = 2), col = 2,lwd=2)
  lines(ksmooth(age, logwage, "normal", bandwidth = 5), col = 3,lwd=2)
  legend("topright", 
         c("bw = 2", "bw = 5"), 
         lty = 1,
         col=c("red","green"),lwd=2,
         cex = 0.6)
})    

# El estimador de Nadaraya-Watson es en realidad un caso particular de una gama mÃ¡s amplia de 
# estimadores no paramÃ©tricos llamados "local polynomial estimators"


# ---------->  LOCAL REGRESSION

# loess: Local Polynomial Regression Fitting
# Usamos la funciÃ³n loess() para la regresiÃ³n local

fit=loess(logwage~age,span=.2,data=Wage)
fit2=loess(logwage~age,span=.5,data=Wage)

# Estas dos estimaciones con distinto span tienen distinta flexibilidad
# en la predicciÃ³n no lineal: 
#      A menor "s" --> mÃ¡s ondulada. span controla el grado de smoothing (suavidad).
#      A mayor "s" --> la estimaciÃ³n serÃ¡ mÃ¡s global y mÃ¡s suave.

# Graficamos ambas estimaciones 

# Creamos la grilla de valores de Age
agelims=range(Wage$age) #rango de age
age.grid=seq(from=agelims[1],to=agelims[2]) 
# tomar de 18 a 80, de 1 en 1

with(Wage, {
  plot(age, logwage, cex=.5,col="darkgrey", main = "RegresiÃ³n Local")
  lines(age.grid,predict(fit,data.frame(age=age.grid)),col="red",lwd=2)
  lines(age.grid,predict(fit2,data.frame(age=age.grid)),col="green",lwd=2)
  legend("topright", 
         c("Span=0.2","Span=0.5"), 
         lty = 1,
         col=c("red","green"),lwd=2,
         cex = 0.6)
}) 

# Â¿Y si lo quiero hacer con ggplot2? Ver la funciÃ³n stat_smooth()
