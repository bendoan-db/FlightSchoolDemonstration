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
dataset = spark.read.table("turbine_gold_for_ml")
display(dataset)

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
  mlflow.log_metric("accuracy", metrics.accuracy)
  
  #TODO: log your model under "turbine_gbt"
  mlflow.spark.log_model(pipelineTrained, "ssm_turbine_pred_gbt")
  mlflow.set_tag("model", "turbine_gbt")

# COMMAND ----------

# MAGIC %md ## Save to the model registry
# MAGIC Get the model having the best metrics.AUROC from the registry

# COMMAND ----------

best_model = mlflow.search_runs(filter_string='tags.model="turbine_gbt" and attributes.status = "FINISHED"', max_results=1).iloc[0]
model_registered = mlflow.register_model("dbfs:/databricks/mlflow-tracking/29251126759236/68b9dfc586e344f499da48ed51659d67/artifacts/ssm_turbine_pred_gbt", "smm_registered_model")

# COMMAND ----------

#get the best model from the registry
#best_model = mlflow.search_runs(filter_string='tags.model="turbine_gbt" and attributes.status = "FINISHED" and metrics.avg_f1 > 0', max_results=1).iloc[0]
#TODO: register the model to MLFLow registry
#model_registered = mlflow.register_model("runs:/ ... 

# COMMAND ----------

# DBTITLE 1,Flag version as staging/production ready

client = mlflow.tracking.MlflowClient()
print("registering model version "+model_registered.version+" as production model")
#TODO: transition the model version = model_registered.version to the stage Production
#client...
client.transition_model_version_stage(
  name=model_registered.name,
  version=model_registered.version,
  stage="Production",
)


# COMMAND ----------

# MAGIC %md #Deploying & using our model in production
# MAGIC 
# MAGIC Now that our model is in our MLFlow registry, we can start to use it in a production pipeline.

# COMMAND ----------

# MAGIC %md ### Scaling inferences using Spark 
# MAGIC We'll first see how it can be loaded as a spark UDF and called directly in a SQL function:

# COMMAND ----------

#TODO: load the model from the registry
get_status_udf = mlflow.pyfunc.spark_udf(spark, f"models:/{model_registered.name}/production", "string")
spark.udf.register("predict_status", get_status_udf)

#TODO: define the model as a SQL function to be able to call it in SQL

# COMMAND ----------

# MAGIC %sql
# MAGIC --TODO: call the model in SQL using the udf registered as function
# MAGIC select *, predict_status(AN3, AN4, AN5, AN6, AN7, AN8, AN9, AN10) as status_forecast from turbine_gold_for_ml LIMIT 5

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * from turbine_gold_for_ml

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
