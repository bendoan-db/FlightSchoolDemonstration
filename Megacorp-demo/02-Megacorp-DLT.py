# Databricks notebook source
# MAGIC %md #Introducing Delta Live Table
# MAGIC 
# MAGIC TODO: Explain the typical challenges a customer will face without DLT
# MAGIC 
# MAGIC TODO: Explain what DLT is adding as value for the customer

# COMMAND ----------

#TODO
#In python or SQL, rewrite the transformations from notebook 01-Megacorp-data-ingestion using DLT, and building the DLT graph
#Add some expectations
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
  df = spark.readStream.option("ignoreChanges", "true").table('dec21_flightschool_team2.turbines_bronze') \
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
  turbine_stream = spark.readStream.table('dec21_flightschool_team2.turbines_silver')
  turbine_status = spark.read.table("dec21_flightschool_team2.turbines_status_gold")
  df = turbine_stream.join(turbine_status, ['id'], 'left')
  return df

# COMMAND ----------

#TODO: review the the expectation dashboard and how the data can be used to track ingestion quality
#more details: https://e2-demo-field-eng.cloud.databricks.com/?o=1444828305810485#notebook/3469214860228002/command/3418422060417474
#data quality tracker dashboard: https://e2-demo-field-eng.cloud.databricks.com/sql/dashboards/6f73dd1b-17b1-49d0-9a11-b3772a2c3357-dlt---retail-data-quality-stats?o=1444828305810485
#do not try to reproduce a dashboard for this specific use-case
