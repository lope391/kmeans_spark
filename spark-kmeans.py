from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql.types import *
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, CountVectorizer, StopWordsRemover
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline

#Contexto Local de SPARK
sc = SparkContext('local')
spark = SparkSession(sc)

print("PROCESO COMENZADO")

#Lee todos los contenidos de la carpeta y crea un RDD (path, contenido)
files = sc.wholeTextFiles("hdfs:///user/lcarva12/pruebaKmeans/")

#Imprime 5 primeros registros en el RDD, IMPRIME EL CONTENIDO DEL ARCHIVO COMPLETO
print("RDD:--------------------------")
#print(files.take(5))
for r in files.take(5):
  print(r)

#Esquema a aplicarle al RDD para volverlo un dataframe con path y contenido del archivo
schema =  StructType([StructField ("path" , StringType(), True) , 
StructField("text" , StringType(), True)]) 
df = spark.createDataFrame(files,schema)

#Imprime el Dataframe generado
print("DATAFRAME:--------------------")
df.show()

#Imprime la cantidad de archivos leidos
print("NUMERO DE ARCHIVOS: ", df.cache().count(), "-------------")

#Definicion de las transformaciones a ser aplicadas
tokenizer = Tokenizer(inputCol="text", outputCol="tokens")
remover = StopWordsRemover(inputCol="tokens", outputCol="stopWordsRemovedTokens")
hashingTF = HashingTF(inputCol="stopWordsRemovedTokens", outputCol="rawFeatures", numFeatures=2000)
idf = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=1)
kmeans = KMeans(k=2)

#creacion del mapa de transformaciones
pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, kmeans])

#inserta el dataframe como el inicio de las transformaciones
model = pipeline.fit(df)

#ejecuta las trasformaciones mapeadas y guarda el resultado
results = model.transform(df)
results.cache()

#Imprime los Resultados
print("RESULTADOS:------------------")
results.show()

print("PROCESO TERMINADO")

#display(results.groupBy("prediction").count())  # Note "display" is for Databricks; use show() for OSS Apache Spark
