# Databricks notebook source
#read in table -- replace table with mocked inference data
turbines_silver_table = spark.read.table("dec21_flightschool_team2.turbines_silver")

# COMMAND ----------

#sample the table and convert to pandas -- can remove sampling for inbound batches since they're small

turbines_inference_table = turbines_silver_table.sample(0.001).toPandas()

# COMMAND ----------

#python udf for 

import mlflow.pyfunc
import pandas as pd

model_name = "dec21_flightschool_team2_predict_turbine_status"
model_version = 1

model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/{model_version}"
)

predictions = model.predict(turbines_inference_table)

# COMMAND ----------

predictions_df = pd.DataFrame(predictions, columns=['predictions'])

# COMMAND ----------

predictions_df.head(10)

# COMMAND ----------


