# Databricks notebook source
dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"], "Reset all data")

# COMMAND ----------

# MAGIC %run ./resources/00-setup $reset_all_data=$reset_all_data

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC # Wind Turbine Predictive Maintenance
# MAGIC 
# MAGIC In this example, we demonstrate anomaly detection for the purposes of finding damaged wind turbines. A damaged, single, inactive wind turbine costs energy utility companies thousands of dollars per day in losses.
# MAGIC 
# MAGIC [test](resources/images/etl-diagram-full)
# MAGIC 
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

# DBTITLE 1,View our Raw IoT Data Stream
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

# DBTITLE 1,Create Database using In-Line SQL Syntax
# MAGIC %sql
# MAGIC CREATE database if not exists doan_iot_turbine_demo

# COMMAND ----------

# DBTITLE 1,Stream Raw IoT into the Bronze Table
bronzeDF = spark.readStream \
                .format("cloudFiles") \
                .option("cloudFiles.format", "parquet") \
                .option("cloudFiles.maxFilesPerTrigger", 1) \
                .schema("value string, key double") \
                .load("/mnt/quentin-demo-resources/turbine/incoming-data") 

bronzeDF.writeStream \
        .option("ignoreChanges", "true") \
        .trigger(processingTime='1 seconds') \
        .table("doan_iot_turbine_demo.turbine_data_bronze")

# COMMAND ----------

# DBTITLE 1,Set Auto-compaction and Optimize Write Partitions to Maximize Stream Efficiency
# MAGIC %sql
# MAGIC create table if not exists doan_iot_turbine_demo.turbine_data_bronze (key double not null, value string) using delta ;
# MAGIC   
# MAGIC -- Turn on autocompaction to solve small files issues on your streaming job, that's all you have to do!
# MAGIC alter table doan_iot_turbine_demo.turbine_data_bronze set tblproperties ('delta.autoOptimize.autoCompact' = true, 'delta.autoOptimize.optimizeWrite' = true);

# COMMAND ----------

# DBTITLE 1,Check Bronze Table
# MAGIC %sql
# MAGIC select * from doan_iot_turbine_demo.turbine_data_bronze;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2/ Silver layer: transform JSON data into tabular table

# COMMAND ----------

# DBTITLE 1,Extract Nested JSON Data and Join with Batch Data on State Geography
#Our bronze now has "KEY, JSON" as schema. We need to extract the json and expand it as a full table, having 1 column per sensor entry

jsonSchema = StructType([StructField(col, DoubleType(), False) for col in ["AN3", "AN4", "AN5", "AN6", "AN7", "AN8", "AN9", "AN10", "SPEED", "TORQUE", "ID"]] + [StructField("TIMESTAMP", TimestampType())])

#join our streaming silver table to batch table on geographic data
states = spark.read.table("doan_turbine_demo_states_gold")

spark.readStream.table('doan_iot_turbine_demo.turbine_data_bronze') \
     .withColumn("jsonData", from_json(col("value"), jsonSchema)) \
     .select("jsonData.*") \
     .join(states, ["ID"], 'left')\
     .writeStream \
     .option("mergeSchema", "true")\
     .format("delta") \
     .trigger(processingTime='1 seconds') \
     .table("doan_iot_turbine_demo.turbine_data_silver")

# COMMAND ----------

# DBTITLE 1,Build Constraints into the Silver Table
# MAGIC %sql
# MAGIC -- let's add some constraints in our table, to ensure or ID can't be negative (need DBR 7.5)
# MAGIC ALTER TABLE doan_iot_turbine_demo.turbine_data_silver ADD CONSTRAINT idGreaterThanZero CHECK (id >= 0);
# MAGIC -- let's enable the auto-compaction
# MAGIC alter table doan_iot_turbine_demo.turbine_data_silver set tblproperties ('delta.autoOptimize.autoCompact' = true, 'delta.autoOptimize.optimizeWrite' = true);

# COMMAND ----------

silver_turbine_df = spark.readStream.table('doan_iot_turbine_demo.turbine_data_silver')
silver_turbine_df.createOrReplaceTempView('temp_silver')

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Select data
# MAGIC select avg(speed) as average_speed, timestamp as date 
# MAGIC from temp_silver
# MAGIC group by timestamp
# MAGIC having average_speed > 0;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3/ Gold layer: Final Joins and Enrichments

# COMMAND ----------

# DBTITLE 1,Create a Gold Table with Labels for Training
# MAGIC %sql 
# MAGIC create table if not exists doan_iot_turbine_demo.turbine_labels_gold (id int, status string) using delta;
# MAGIC 
# MAGIC COPY INTO doan_iot_turbine_demo.turbine_labels_gold
# MAGIC   FROM '/mnt/quentin-demo-resources/turbine/status'
# MAGIC   FILEFORMAT = PARQUET;

# COMMAND ----------

# MAGIC %sql select * from doan_iot_turbine_demo.turbine_labels_gold

# COMMAND ----------

# DBTITLE 1,Join Silver Stream Date with Labels to Create Gold Training Table
turbine_silver_stream = spark.readStream.table('doan_iot_turbine_demo.turbine_data_silver')
turbine_status = spark.read.table("doan_iot_turbine_demo.turbine_labels_gold")

turbine_gold_stream = turbine_silver_stream.join(turbine_status, ['id'], 'left') \

turbine_gold_stream.writeStream \
              .option("mergeSchema", "true") \
              .format("delta") \
              .trigger(processingTime='10 seconds') \
              .table("doan_iot_turbine_demo.turbine_training_data_gold ")


# COMMAND ----------

# MAGIC %sql
# MAGIC --Our turbine gold table should be up and running!
# MAGIC select * from doan_iot_turbine_demo.turbine_training_data_gold

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
# MAGIC ### With Streaming predictions for our turbines, your plant operators can take proactive measures and prevent unplanned downtime
# MAGIC   
# MAGIC   <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/iot-wind-turbine/resources/images/turbine-demo-flow-da.png" width="90%"/>

# COMMAND ----------

# note-to-self: don't forget to clean up after one's own self

for s in spark.streams.active:
  s.stop()

# COMMAND ----------

# DBTITLE 1,Predict Health Status of Turbine Using a Registered Model
import mlflow.pyfunc

model_name = "dec21_flightschool_team2_predict_turbine_status"
model_version = 1

model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/{model_version}"
)

pdf = pd.DataFrame(model.predict(df), columns=["STATUS"])
df['STATUS'] = pdf['STATUS']

# COMMAND ----------

sparkDF=spark.createDataFrame(df)
sparkDF.write.format("delta").mode("append").saveAsTable("dec21_flightschool_team2.turbine_inferences_silver")

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC <h2> Overall Value: </h2>
# MAGIC 
# MAGIC <li> Highly scalable and easy to read pipelines </li>
# MAGIC <li> Multi language, same tech for any cloud platform </li>
# MAGIC <li> Read and write to almost any data source </li>
# MAGIC <li> Easy to build and debug </li>
# MAGIC <li> Logic is in one place for all teams to see, increasing speed, transparency, and knowledge transfer </li>
# MAGIC <li> Less code to write overall since streaming takes care of the incremental data checkpoint for you </li>

# COMMAND ----------


