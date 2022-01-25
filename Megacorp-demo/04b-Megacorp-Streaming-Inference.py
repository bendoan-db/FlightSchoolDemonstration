# Databricks notebook source
#read in table -- replace table with mocked inference data
turbines_silver_table = spark.read.table("dec21_flightschool_team2.inferences_silver")

# COMMAND ----------

#sample the table and convert to pandas -- can remove sampling for inbound batches since they're small

turbines_inference_table = turbines_silver_table.toPandas()

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

import mlflow
logged_model = 'runs:/9bdd4ec83e774b7891b3a0fe72f79224/model'

# Load model as a Spark UDF.
loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model)

# Predict on a Spark DataFrame.
columns = list(turbines_gold_table.columns)
turbines_gold_table.withColumn('predictions', loaded_model(*columns)).collect()

# COMMAND ----------



# COMMAND ----------

turbines_gold_table = spark.read.table("dec21_flightschool_team2.turbines_gold").drop("status")

# COMMAND ----------

from pyspark.sql.functions import struct
# Apply the model to the new data
# Load model as a Spark UDF.

limited = turbines_silver_table.limit(1)

loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model)

udf_inputs = struct(*(limited.columns))

limited = limited.withColumn(
  "prediction",
  loaded_model(udf_inputs)
)

display(turbines_silver_table)

# COMMAND ----------

json_dict = {"columns": ["ID", "AN3", "AN4", "AN5", "AN6", "AN7", "AN8", "AN9", "AN10", "SPEED", "TORQUE", "TIMESTAMP"], "data": [[155.0, -0.44134, 2.7103, -0.88699, 0.33465, -1.0796, 1.3363, -1.6837, -3.1602, 0.00064316, 0.0, "2020-05-13T06:30:13"], [303.0, 0.57618, -1.9939, 2.0863, -2.1487, 3.6466, 1.0769, 0.2658, 0.39624, 0.00060586, None, "2020-05-27T14:24:06"], [642.0, -0.69298, -5.5987, -3.5991, -7.5765, -0.32353, -0.50265, 2.573, -0.73202, 0.0006429483711344697, 0.0, "2020-06-09T11:55:20"], [13.0, -0.26817, -3.7197, -1.6498, -4.894, 1.981, -1.1741, -0.38546, -1.0892, 0.00063931, None, "2020-05-27T02:13:36"], [696.0, -4.4812, 5.756, 4.8432, -2.1428, -1.7681, 11.705, -0.91144, -8.3663, 0.0009203704923383678, None, "2020-06-09T11:56:14"]]}

json_dict

# COMMAND ----------

import pandas as pd
import json

df_pandas = pd.DataFrame(json_dict["data"], columns=json_dict["columns"])

# COMMAND ----------

df_pandas.head()

# COMMAND ----------

model_name = "dec21_flightschool_team2_predict_turbine_status"
model_version = 1

model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/{model_version}"
)

predictions = model.predict(df_pandas)

predictions_df = pd.DataFrame(predictions, columns=['predictions'])
predictions_df.head(10)

# COMMAND ----------

sparkDF = spark.createDataFrame(df_pandas)

# COMMAND ----------

import mlflow
logged_model = 'runs:/5d20ed6aaccd497e91e038044a736a55/model'

# Load model as a Spark UDF.
loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model)

# Predict on a Spark DataFrame.
columns = list(sparkDF.columns)
sparkDF.withColumn('predictions', loaded_model(*columns)).collect()

# COMMAND ----------


