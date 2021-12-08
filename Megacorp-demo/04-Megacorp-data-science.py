# Databricks notebook source
dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"], "Reset all data")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scalable Machine Learning with MLFlow on the Delta Lakehouse

# COMMAND ----------

# MAGIC %run ./resources/00-setup $reset_all=$reset_all_data

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Data Exploration
# MAGIC What do the distributions of sensor readings look like for our turbines? 

# COMMAND ----------

#read in gold table from ingestion process
dataset = spark.read.table("dec21_flightschool_team2.turbines_gold")
dataset_siver = spark.read.table("dec21_flightschool_team2.turbines_silver")
dataset_siver.printSchema()

# COMMAND ----------

pandas_data = dataset.toPandas()
pandas_data.head()

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Train Model and Track Experiments

# COMMAND ----------

#once the data is ready, we can train a model
import mlflow
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.mllib.evaluation import MulticlassMetrics

with mlflow.start_run():

  training, test = dataset.limit(1000).randomSplit([0.9, 0.1], seed = 5)
  
  gbt = GBTClassifier(labelCol="label", featuresCol="features").setMaxIter(5)
  grid = ParamGridBuilder().addGrid(gbt.maxDepth, [3,4,5,10,15,25,30]).build()

  metrics = MulticlassClassificationEvaluator(metricName="f1")
  cv = CrossValidator(estimator=gbt, estimatorParamMaps=grid, evaluator=metrics, numFolds=2)

  featureCols = ["AN3", "AN4", "AN5", "AN6", "AN7", "AN8", "AN9", "AN10"]
  stages = [VectorAssembler(inputCols=featureCols, outputCol="va"), StandardScaler(inputCol="va", outputCol="features"), StringIndexer(inputCol="status", outputCol="label"), cv]
  pipeline = Pipeline(stages=stages)

  pipelineTrained = pipeline.fit(training)
  
  predictions = pipelineTrained.transform(test)
  metrics = MulticlassMetrics(predictions.select(['prediction', 'label']).rdd)
  
  #TODO: how can you use MLFLow to log your metrics (precision, recall, f1 etc) 
  #Tips: what about auto logging ?
  
  #TODO: log your model under "turbine_gbt"
  mlflow.spark.log_model("/turbine_pred_gbt")
  mlflow.set_tag("model", "turbine_gbt")

# COMMAND ----------

# MAGIC %md ## Save to the model registry
# MAGIC Get the model having the best metrics.AUROC from the registry

# COMMAND ----------

#get the best model from the registry
best_model = mlflow.search_runs(filter_string='tags.model="turbine_gbt" and attributes.status = "FINISHED" and metrics.f1 > 0', max_results=1).iloc[0]
#TODO: register the model to MLFLow registry
model_registered = mlflow.register_model("runs:/ ... 

# COMMAND ----------

# DBTITLE 1,Flag version as staging/production ready
client = mlflow.tracking.MlflowClient()
print("registering model version "+model_registered.version+" as production model")
#TODO: transition the model version = model_registered.version to the stage Production
client...

# COMMAND ----------

# MAGIC %md #Deploying & using our model in production
# MAGIC 
# MAGIC Now that our model is in our MLFlow registry, we can start to use it in a production pipeline.

# COMMAND ----------

# MAGIC %md ### Scaling inferences using Spark 
# MAGIC We'll first see how it can be loaded as a spark UDF and called directly in a SQL function:

# COMMAND ----------

#TODO: load the model from the registry
get_status_udf = mlflow.pyfunc....
#TODO: define the model as a SQL function to be able to call it in SQL

# COMMAND ----------

# MAGIC %sql
# MAGIC --TODO: call the model in SQL using the udf registered as function
# MAGIC select *, ... as status_forecast from turbine_gold_for_ml

# COMMAND ----------

# MAGIC %md #TODO: recap
# MAGIC 
# MAGIC What has been done so far ?
# MAGIC 
# MAGIC What's the value for you customer?
# MAGIC 
# MAGIC Where are you in your story ?
# MAGIC 
# MAGIC This was a bit annoying to write all this code, and it's not coming with best practices like hyperparameter tuning. What could you leverage within databricks to accelerate this implementation / your customer POC?

# COMMAND ----------

import os
import requests
import numpy as np
import pandas as pd

def create_tf_serving_json(data):
  return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(dataset):
  url = 'https://e2-demo-field-eng.cloud.databricks.com/model/dec21_flightschool_team2_predict_turbine_status/1/invocations'
  headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}'}
  data_json = dataset.to_dict(orient='split') if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
  response = requests.request(method='POST', headers=headers, url=url, json=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  return response.json()

# COMMAND ----------

dataset.printSchema

# COMMAND ----------

pandas_data = dataset.sample(.000001).toPandas()
pandas_data.dtypes

# COMMAND ----------

df_string_2 = pandas_data.to_json()

# COMMAND ----------

type(df_string_2)

# COMMAND ----------

dbutils.fs.put("/home/nicholas.barretta@databricks.com/inference_text_6.txt", df_string_2)

# COMMAND ----------

dataset.sample(0.1)

# COMMAND ----------



# COMMAND ----------

inference_dataset = dataset.drop('status').sample(0.25)

# COMMAND ----------

score_model(pandas_data)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## START HERE

# COMMAND ----------

inference_table_doan = dataset.sample(0.000001).drop("status")

# COMMAND ----------



# COMMAND ----------

inference_table_doan.write.format("delta").saveAsTable("dec21_flightschool_team2.inference_table")

# COMMAND ----------

import mlflow
logged_model = 'runs:/1b18933689434a79ab6d0ed785dd4388/model'

# Load model as a Spark UDF.
loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model)

# Predict on a Spark DataFrame.
columns = list(inference_table_doan.columns)
inference_table_doan.withColumn('predictions', loaded_model(*columns)).collect()

# COMMAND ----------

itpd = inference_table_doan.toPandas()

# COMMAND ----------

import mlflow.pyfunc

model_name = "dec21_flightschool_team2_predict_turbine_status"
model_version = 1

model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/{model_version}"
)

model.predict(itpd)

# COMMAND ----------

inference_table_doan.printSchema()

# COMMAND ----------


