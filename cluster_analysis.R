#==================================================#
#                                                  #
#               CLASE TUTORIAL NÂ°12                #
#                     Clusters                     #
#                                                  #
#==================================================#

# Profesor: Walter Sosa Escudero
# Tutora: Carla Srebot 


# rm(list=ls())

# Instalamos los paquetes necesarios:
# install.packages("tidyverse") # para manipular los datos
# install.packages("factoextra") # algoritmos de clustering y visualizaciÃ³n

library(tidyverse)  
library(factoextra) 
library(cluster) # algoritmos de clustering

# Abrimos la base
df <- USArrests

# Borramos los missing values
df <- na.omit(df)

# Escalamos las variables (estandarizamos)... Â¿Por quÃ©?
df <- scale(df)
 
# Computar la matriz de distancia entre las filas de la matriz de datos. La distancia
# calculada por default es la EuclÃ­dea. Otros mÃ©todos que permite el comando get_dist:
# "euclidean", "maximum", "manhattan", "canberra", "binary", "minkowski", "pearson", 
# "spearman" o "kendall".
distance <- get_dist(df, method = "euclidean")

# Para visualizar la matriz de distancias
fviz_dist(distance, gradient = list(low = "#00AFBB", mid = "white", high = "#FC4E07"))

# FunciÃ³n kmeans: agrupamos a los datos en dos clusters (centers = 2);
# nstart = 25 genera 25 configuraciones iniciales, reportando solo la mejor.
k2 <- kmeans(df, centers = 2, nstart = 25)
# nstart: particiones iniciales. Se queda con la mejor. Se recomienda poner
# nstart >= 25

# Veamos quÃ© dio de manera compacta
str(k2)

# Â¡DEMASIADA INFORMACION! 
# No desesperÃ©is, lo importante es:
#      1) cluster: vector de integrales (de 1:k) indicando a quÃ© cluster fue asignado cada punto.
#      2) centers: matriz de centros de clusters.
#      3) totss: suma de cuadrados totales. --> Â¿depende de la particiÃ³n?
#      4) withinss: vector de suma de cuadrados dentro de cada cluster.
#      5) tot.withinss: sum(withinss).
#      6) betweenss: suma de cuadrados entre clusters.
#      7) size: cantidad de puntos en cada cluster.

# Para agregar la clasificaciÃ³n a la base
dd <- cbind(USArrests, cluster_means = k2$cluster)

# Ahora a usar esto para interpretar los resultados:
k2

# En un lindo grÃ¡fico de colores:
fviz_cluster(k2, data = df)
# Ojo: si hay mÃ¡s de dos dimensiones, este comando calcularÃ¡ Componentes Principales y 
# graficarÃ¡ los puntos de acuerdo a los primeros dos componentes principales que expliquen
# la mayor parte de la varianza.

# Otro lindo grÃ¡fico, ahora eligiendo un par de variables:
df %>%
  as_tibble() %>%
  mutate(cluster = k2$cluster,
         state = row.names(USArrests)) %>%
  ggplot(aes(UrbanPop, Murder, color = factor(cluster), label = state)) +
  geom_text()

# Veamos quÃ© pasa si usamos distinta cantidad de clusters:
k3 <- kmeans(df, centers = 3, nstart = 25)
k4 <- kmeans(df, centers = 4, nstart = 25)
k5 <- kmeans(df, centers = 5, nstart = 25)

# Comparemos con grÃ¡ficos:
p1 <- fviz_cluster(k2, geom = "point", data = df) + ggtitle("k = 2")
p2 <- fviz_cluster(k3, geom = "point",  data = df) + ggtitle("k = 3")
p3 <- fviz_cluster(k4, geom = "point",  data = df) + ggtitle("k = 4")
p4 <- fviz_cluster(k5, geom = "point",  data = df) + ggtitle("k = 5")

library(gridExtra)
grid.arrange(p1, p2, p3, p4, nrow = 2)


# Lindos grÃ¡ficos, sÃ­, Â¿pero cÃ³mo sabemos quÃ© k usar?  

# Hay 3 mÃ©todos:
#   1) MÃ©todo de Elbow
#   2) MÃ©todo de silueta promedio
#   3) Gap statistic

# 1) MÃ©todo de Elbow:               

# FunciÃ³n para computar la suma de cuadrados totales dentro de cada cluster: 
set.seed(101)
fviz_nbclust(df, kmeans, method = "wss", k.max =14) #wss: total within sum of square
# Parece que 4 es el nÃºmero Ã³ptimo de clusters... Â¿o 2? Muy subjetivo

# Forma manual de hacer lo mismo:
# wss <- function(k) {kmeans(df, k, nstart = 10 )$tot.withinss}
# Calculamos y graficamos wss para k=1 hasta k=15
# k.values <- 1:15
# wss_values <- map_dbl(k.values, wss) #map_dbl sirve para aplicar una funciÃ³n a cada elemento de un vector
# plot(k.values, wss_values,
#type="b", pch = 19, frame = FALSE, 
#xlab="NÃºmero de clusters K",
#ylab="Suma de cuadrados totales dentro de los clusters")

# 2) MÃ©todo de la silueta: buscamos que sea alto.
set.seed(101)
fviz_nbclust(df, kmeans, method = "silhouette")
# Parece que ganan los 2 clusters; le sigue 4.

# Forma manual de hacer lo mismo:
#avg_sil <- function(k) {
#km.res <- kmeans(df, centers = k, nstart = 25)
#ss <- silhouette(km.res$cluster, dist(df))
#mean(ss[, 3])
#}
# Computamos la suma de cuadrados totales para k = 2 hasta k = 15
#k.values <- 2:15
# Extraemos siluetas primedio para 2-15 clusters
#avg_sil_values <- map_dbl(k.values, avg_sil)
#plot(k.values, avg_sil_values,
#type = "b", pch = 19, frame = FALSE, 
#xlab = "NÃºmero de clusters",
#ylab = "Siluetas promedio")

# 3) Gap statistic:
set.seed(101)
gap_stat <- clusGap(df, FUN = kmeans, nstart = 25, K.max = 10, B = 50)

# Imprimimos el resultado:
print(gap_stat, method = "firstmax")

# Visualmente:
set.seed(101)
fviz_gap_stat(gap_stat)

# Ver paquete NbClust para otras alternativas.

# Como dos de los mÃ©todos nos dieron que k=4, veamos los resultados finales con esta separaciÃ³n:
set.seed(101)
final <- kmeans(df, 4, nstart = 25)
print(final)
fviz_cluster(final, data = df)

# Para ver estadÃ­stica descriptiva por cluster:
USArrests %>%
  mutate(Cluster = final$cluster) %>%
  group_by(Cluster) %>%
  summarise_all("mean")

# k-mediods:
set.seed(101)
mediods.res <- pam(df, 2)
print(mediods.res)

# Para agregar la clasificaciÃ³n a la base
dd <- cbind(dd, cluster_mediods = mediods.res$cluster)

set.seed(101)
final <- pam(df, 4)
print(final)
fviz_cluster(final, data = df)

# Ver mÃ¡s en https://uc-r.github.io/hc_clustering
# "Practical Guide to Cluster Analysis in R", Kassambra
