# Databricks notebook source
dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"], "Reset all data")

# COMMAND ----------

# MAGIC %run ./resources/00-setup $reset_all_data=$reset_all_data

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC #YOUR DEMO INTRO HERE
# MAGIC TODO: use this cell to present your demo at a high level. What are you building ? What's your story ? How is it linked to megacorp powerplant and its gaz turbine ?
# MAGIC 
# MAGIC What's the data used and what are the fields (be creative!)
# MAGIC 
# MAGIC Need to display images ? check https://docs.databricks.com/data/filestore.html#filestore

# COMMAND ----------

TODO: Use %fs to visualize the incoming data under /mnt/quentin-demo-resources/turbine/incoming-data-json

# COMMAND ----------

# MAGIC %sql
# MAGIC --TODO : Select and display the entire incoming json data using a simple SQL SELECT query

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1/ Bronze layer: ingest data stream

# COMMAND ----------

#Tips: Python path variable is available, use it to store intermediate data or your checkpoints
print(f"path={path}")
#Just save create the database to the current database, it's been initiliazed locally to your user to avoid conflict
print("your current database has been initialized to:")
print(sql("SELECT current_database() AS db").collect()[0]['db'])

# COMMAND ----------

# DBTITLE 1,Stream landing files from cloud storage
bronzeDF = spark.readStream .....
#TODO: ingest data using cloudfile.
#Incoming data is available under /mnt/quentin-demo-resources/turbine/incoming-data-json
#Goal: understand autoloader value and functionality (schema evolution, inference)
#What your customer challenges could be with schema evolution, schema inference, incremental mode having lot of small files? 
#How do you fix that ?
#Tips: use .option("cloudFiles.maxFilesPerTrigger", 1) to consume 1 file at a time and simulate a stream during the demo
                  
# Write Stream as Delta Table
bronzeDF.writeStream ...
#TODO: write the output as "turbine_bronze" delta table, with a trigger of 10 seconds
#Tips: use y

# COMMAND ----------

# DBTITLE 1,Our raw data is now available in a Delta table
# MAGIC %sql
# MAGIC select * from turbine_bronze;
# MAGIC 
# MAGIC --TODO: which table property should you define to solve small files issue ? What's the typicall challenge running streaming operation? And the value for your customer.
# MAGIC ALTER TABLE turbine_bronze SET TBLPROPERTIES (...)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2/ Silver layer: transform JSON data into tabular table

# COMMAND ----------

#Our bronze silver now have "KEY, JSON" as schema. We need to extract the json and expand it as a full table, having 1 column per sensor entry

silverDF = spark.readStream.table('turbine_bronze') ....
#TODO: use pyspark from_json to explode the JSON: https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.functions.from_json.html

silverDF.writeStream ...
#TODO: write it back to your "turbine_silver" table

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from turbine_silver;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3/ Gold layer: join information on Turbine status to add a label to our dataset

# COMMAND ----------

#TODO: our data is available under /mnt/quentin-demo-resources/turbine/status. Use dbutils.fs to display the folder content

# COMMAND ----------

spark.read.format("parquet").load("/mnt/quentin-demo-resources/turbine/status")

# COMMAND ----------

#TODO: save the status data as our turbine_status table
spark.readStream.format("parquet").load("/mnt/quentin-demo-resources/turbine/status").writeStream.....saveAsTable("turbine_status")

# COMMAND ----------

# DBTITLE 1,Join data with turbine status (Damaged or Healthy)
turbine_stream = spark.readStream.table('turbine_silver')
turbine_status = spark.read.table("turbine_status")

#TODO: do a left join between turbine_stream and turbine_status on the 'id' key and save back the result as the "turbine_gold" table
turbine_stream.join(....

# COMMAND ----------

# MAGIC %sql
# MAGIC --Our turbine gold table should be up and running!
# MAGIC select TIMESTAMP, AN3, SPEED, status from turbine_gold;

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Run DELETE/UPDATE/MERGE with DELTA ! 
# MAGIC We just realized that something is wrong in the data before 2020! Let's DELETE all this data from our gold table as we don't want to have wrong value in our dataset

# COMMAND ----------

# MAGIC %sql
# MAGIC DELETE FROM turbine_gold where timestamp < '2020-00-01';

# COMMAND ----------

# MAGIC %sql
# MAGIC -- TODO: show some Delta Love.
# MAGIC -- What's unique to Delta and how can it be usefull for your customer?

# COMMAND ----------

# MAGIC %md
# MAGIC ### Grant Access to Database
# MAGIC If on a Table-ACLs enabled High-Concurrency Cluster

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Note: this won't work with standard cluster. 
# MAGIC -- DO NOT try to make it work during the demo.
# MAGIC -- Understand what's required as of now (which cluster type) and the implications
# MAGIC -- explore Databricks Unity Catalog initiative (go/uc) 
# MAGIC 
# MAGIC GRANT SELECT ON DATABASE turbine_demo TO `data.scientist@databricks.com`
# MAGIC GRANT SELECT ON DATABASE turbine_demo TO `data.analyst@databricks.com`

# COMMAND ----------

# MAGIC %md
# MAGIC ### Don't forget to Cancel all the streams once your demo is over

# COMMAND ----------

for s in spark.streams.active:
  s.stop()
