# Databricks notebook source
# MAGIC %md #Leveraging Databricks SQL to Visualize Streaming Data and ML Predictions
# MAGIC 
# MAGIC Using DBSQL we can read our bronze/silver/gold tables from our delta lakehouse into our notebook, and do some high level aggregations and visualizations with inline SQL inject

# COMMAND ----------

# DBTITLE 1,Load tables from the delta lakehouse
# MAGIC %sql
# MAGIC create database if not exists demo_turbine;
# MAGIC CREATE TABLE if not exists `demo_turbine`.`turbine_bronze` USING delta LOCATION 'dbfs:/mnt/quentin-demo-resources/turbine/bronze/data';
# MAGIC CREATE TABLE if not exists `demo_turbine`.`turbine_silver` USING delta LOCATION 'dbfs:/mnt/quentin-demo-resources/turbine/silver/data';
# MAGIC CREATE TABLE if not exists `demo_turbine`.`turbine_gold`   USING delta LOCATION 'dbfs:/mnt/quentin-demo-resources/turbine/gold/data' ;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loading ML Predictions from our Model Predictions Table
# MAGIC Leveraging Databricks' Unified Analytics approach, we can also load in our model predictions and do some quick analysis

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE if not exists `demo_turbine`.`turbine_outage_predictions` USING delta LOCATION 'dbfs:/mnt/quentin-demo-resources/turbine/gold/data';
# MAGIC select id, status as predicted_status from demo_turbine.turbine_outage_predictions

# COMMAND ----------

# MAGIC %md
# MAGIC ## Leveraging DSQL Dashboards!
# MAGIC 
# MAGIC Finally, we can create a comprehensive view of Megacorp's live sensor data, as well as its ML model predictions in a single dashboard, that provides near real-time updates based on incoming sensor data

# COMMAND ----------

# MAGIC %md
# MAGIC [MegaCorp Turbine Dashboard](https://e2-demo-field-eng.cloud.databricks.com/sql/dashboards/b16116e2-9783-4d24-a6c5-7bca9e5451b4-megacorp-turbine-health-and-outage-predictions?o=1444828305810485)

# COMMAND ----------


