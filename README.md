# Implementación Kmeans en Spark
Topicos Especiales en Telematica, Universidad EAFIT

Autores: Carvajal, Lope. Sarabia, Samuel.

# Descripción
Implementación del algoritmo de clustering de documentos por similaridad utolozando Kmeans.
Utiliza un pipeline con las transformaciónes implementadas en la libreria de machine learning de spark.

Recibe como parametro la ruta a una carpeta que contenga archivos de texto, ejecuta las transformaciones y muestra las primeras 20 tuplas.

# Ejecución
Para ejecutar el programa en el cluster ejecutar el archivo exec.sh
     
     sh exec.sh
     
Dentro de este archivo se pueden editar parametros de memoria maxima o numero de nodos

    spark-submit --master yarn --deploy-mode cluster --executor-memory {4G} --num-executors {4} spark-kmeans.py  
    
# Parametros
Dentro del archivo python se puede editar varios campos:
### Data Set
El dataset se recibe dentro de la linea 25 puede ser la dirección a cualquier carpeta. En la linea 19, 22 y 25 se puede escoger entre 3 Datasets:
* Gutenberg Completo: 3000 Documentos
* Muestra de Gutenber: 180 Documentos
* Data set de Prueba: 5 Documentos

### KMeans
En la linea 49 se define en cuantos textos debe aparecer un termino para ser significativo
     
    idf = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq={5})
    
En la linea 50 se define el numero de clusters a intentar buscar

    kmeans = KMeans(k={5})
    
