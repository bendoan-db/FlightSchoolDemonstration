# Databricks notebook source
dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"], "Reset all data")

# COMMAND ----------

# MAGIC %run ./resources/00-setup $reset_all_data=$reset_all_data

# COMMAND ----------

# MAGIC %md
# MAGIC -sandbox
# MAGIC # Wind Turbine Predictive Maintenance
# MAGIC 
# MAGIC In this example, we demonstrate anomaly detection for the purposes of finding damaged wind turbines. A damaged, single, inactive wind turbine costs energy utility companies thousands of dollars per day in losses.
# MAGIC 
# MAGIC 
# MAGIC <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/iot-wind-turbine/resources/images/turbine-demo-flow.png" width="90%"/>
# MAGIC 
# MAGIC 
# MAGIC <div style="float:right; margin: -10px 50px 0px 50px">
# MAGIC   <img src="https://s3.us-east-2.amazonaws.com/databricks-knowledge-repo-images/ML/wind_turbine/wind_small.png" width="400px" /><br/>
# MAGIC   *locations of the sensors*
# MAGIC </div>
# MAGIC Our dataset consists of vibration readings coming off sensors located in the gearboxes of wind turbines. 
# MAGIC 
# MAGIC We will use Gradient Boosted Tree Classification to predict which set of vibrations could be indicative of a failure.
# MAGIC 
# MAGIC One the model is trained, we'll use MFLow to track its performance and save it in the registry to deploy it in production
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC *Data Source Acknowledgement: This Data Source Provided By NREL*
# MAGIC 
# MAGIC *https://www.nrel.gov/docs/fy12osti/54530.pdf*

# COMMAND ----------

# MAGIC %sql 
# MAGIC select * from parquet.`/mnt/quentin-demo-resources/turbine/incoming-data`

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
bronzeDF = spark.readStream \
                .format("cloudFiles") \
                .option("cloudFiles.format", "parquet") \
                .option("cloudFiles.maxFilesPerTrigger", 1) \
                .schema("value string, key double") \
                .load("/mnt/quentin-demo-resources/turbine/incoming-data") 

bronzeDF.writeStream \
        .option("ignoreChanges", "true") \
        .trigger(processingTime='10 seconds') \
        .table("turbine_bronze")

# COMMAND ----------

# DBTITLE 1,Our raw data is now available in a Delta table
# MAGIC %sql
# MAGIC create table if not exists turbine_bronze (key double not null, value string) using delta ;
# MAGIC   
# MAGIC -- Turn on autocompaction to solve small files issues on your streaming job, that's all you have to do!
# MAGIC alter table turbine_bronze set tblproperties ('delta.autoOptimize.autoCompact' = true, 'delta.autoOptimize.optimizeWrite' = true);

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from turbine_bronze;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2/ Silver layer: transform JSON data into tabular table

# COMMAND ----------

#Our bronze silver now have "KEY, JSON" as schema. We need to extract the json and expand it as a full table, having 1 column per sensor entry

jsonSchema = StructType([StructField(col, DoubleType(), False) for col in ["AN3", "AN4", "AN5", "AN6", "AN7", "AN8", "AN9", "AN10", "SPEED", "TORQUE", "ID"]] + [StructField("TIMESTAMP", TimestampType())])

spark.readStream.table('turbine_bronze') \
     .withColumn("jsonData", from_json(col("value"), jsonSchema)) \
     .select("jsonData.*") \
     .writeStream \
     .option("ignoreChanges", "true") \
     .format("delta") \
     .trigger(processingTime='10 seconds') \
     .table("turbine_silver")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- let's add some constraints in our table, to ensure or ID can't be negative (need DBR 7.5)
# MAGIC ALTER TABLE turbine_silver ADD CONSTRAINT idGreaterThanZero CHECK (id >= 0);
# MAGIC -- let's enable the auto-compaction
# MAGIC alter table turbine_silver set tblproperties ('delta.autoOptimize.autoCompact' = true, 'delta.autoOptimize.optimizeWrite' = true);

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Select data
# MAGIC select * from turbine_silver;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3/ Gold layer: join information on Turbine status to add a label to our dataset

# COMMAND ----------

# MAGIC %sql 
# MAGIC create table if not exists turbine_status_gold (id int, status string) using delta;
# MAGIC 
# MAGIC COPY INTO turbine_status_gold
# MAGIC   FROM '/mnt/quentin-demo-resources/turbine/status'
# MAGIC   FILEFORMAT = PARQUET;

# COMMAND ----------

# MAGIC %sql select * from turbine_status_gold

# COMMAND ----------

spark.read.format("parquet").load("/mnt/quentin-demo-resources/turbine/status")

# COMMAND ----------

#TODO: save the status data as our turbine_status table
spark.readStream.format("parquet").load("/mnt/quentin-demo-resources/turbine/status").writeStream.....saveAsTable("turbine_status")

# COMMAND ----------

# DBTITLE 1,Join data with turbine status (Damaged or Healthy)
turbine_stream = spark.readStream.table('turbine_silver')
turbine_status = spark.read.table("turbine_status_gold")

turbine_stream.join(turbine_status, ['id'], 'left') \
              .writeStream \
              .option("ignoreChanges", "true") \
              .format("delta") \
              .trigger(processingTime='10 seconds') \
              .table("turbine_gold")

# COMMAND ----------

# MAGIC %sql
# MAGIC --Our turbine gold table should be up and running!
# MAGIC select TIMESTAMP, id, AN3, SPEED, status from turbine_gold where status not like "healthy";

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run DELETE/UPDATE/MERGE with DELTA ! 
# MAGIC We just realized that something is wrong in the data before 2020! Let's DELETE all this data from our gold table as we don't want to have wrong value in our dataset

# COMMAND ----------

# MAGIC %sql
# MAGIC DELETE FROM turbine_gold where timestamp < '2020-00-01';

# COMMAND ----------

# MAGIC %sql
# MAGIC -- DESCRIBE HISTORY turbine_gold;
# MAGIC -- If needed, we can go back in time to select a specific version or timestamp
# MAGIC SELECT * FROM turbine_gold TIMESTAMP AS OF '2020-12-01'
# MAGIC 
# MAGIC -- And restore a given version
# MAGIC -- RESTORE turbine_gold TO TIMESTAMP AS OF '2020-12-01'
# MAGIC 
# MAGIC -- Or clone the table (zero copy)
# MAGIC -- CREATE TABLE turbine_gold_clone [SHALLOW | DEEP] CLONE turbine_gold VERSION AS OF 32

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
