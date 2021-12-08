# Databricks notebook source
# MAGIC %md #Introducing Delta Live Table
# MAGIC 
# MAGIC TODO: Explain the typical challenges a customer will face without DLT
# MAGIC 
# MAGIC TODO: Explain what DLT is adding as value for the customer

# COMMAND ----------

#go in Job => DLT, create your DLT pointing to your notebook
#help: https://docs.databricks.com/data-engineering/delta-live-tables/index.html
#Remember: you can use the retail DLT as example

import dlt
from pyspark.sql.functions import *
from pyspark.sql.types import *

json_path = "/mnt/quentin-demo-resources/turbine/incoming-data"

@dlt.table(
  comment="The raw turbine sensor data - unexploded JSON."
)
def turbines_bronze():
  return (spark.read.parquet(json_path))

@dlt.table(
  comment="The silver turbine sensor data - processed so that JSON is exploded into columns.",
  table_properties={"delta.autoOptimize.autoCompact" : "true", "delta.autoOptimize.optimizeWrite" : "true"}
)
@dlt.expect_or_drop("idNotNegative", "ID > 0")
def turbines_silver():
  jsonSchema = StructType([StructField(col, DoubleType(), False) for col in ["AN3", "AN4", "AN5", "AN6", "AN7", "AN8", "AN9", "AN10", "SPEED", "TORQUE", "ID"]] + [StructField("TIMESTAMP", TimestampType())])
  df = dlt.read('turbines_bronze') \
       .withColumn("jsonData", from_json(col("value"), jsonSchema)) \
       .select("jsonData.*")
  return df

@dlt.table(
  comment="The gold turbine status data."
)
@dlt.expect_or_drop("idNotNegative", "ID > 0")
def turbines_status_gold():
  return (spark.read.parquet('/mnt/quentin-demo-resources/turbine/status'))

@dlt.table(
  comment="Turbine sensor data joined with status - gold table.",
  table_properties={"delta.autoOptimize.autoCompact" : "true", "delta.autoOptimize.optimizeWrite" : "true"}
)
def turbines_gold():
  turbine_stream = dlt.read('turbines_silver')
  turbine_status = dlt.read('turbines_status_gold')
  df = turbine_stream.join(turbine_status, ['id'], 'left')
  return df
