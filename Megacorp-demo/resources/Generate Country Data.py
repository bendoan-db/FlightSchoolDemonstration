# Databricks notebook source
countries = spark.read.table("default.covid_us_states_delta")
display(countries)

# COMMAND ----------

distinct_states = countries.select("state").distinct()
display(distinct_states)

# COMMAND ----------

distinct_states.write.format("delta").saveAsTable("doan_states_index")

# COMMAND ----------

from random import randint
print(randint(0, 750))

# COMMAND ----------

from pyspark.sql.functions import row_number
from pyspark.sql.window import Window
from pyspark.sql.functions import rand
w = Window().orderBy("state")
indexedStates = distinct_states.withColumn("state_index", row_number().over(w))

# COMMAND ----------

display(indexedStates)

# COMMAND ----------

indexedStates.write.mode("overwrite").option("mergeSchema", True).format("delta").saveAsTable("ben_doan_databricks_com.doan_states_index")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from ben_doan_databricks_com.doan_states_index

# COMMAND ----------

## to do - create range mapping 750 to rand ints 1-56
from pyspark.sql.types import IntegerType
test = [*range(1, 751)]
type(test)
df = spark.createDataFrame(test, IntegerType())

# COMMAND ----------

from pyspark.sql.functions import rand,when,col
from random import randint
indexed_df = df.withColumn("s_index", (rand()*56).cast(IntegerType()))

# COMMAND ----------

states = spark.read.table("ben_doan_databricks_com.doan_states_index")
joinedStates = states.join(indexed_df, states.state_index == indexed_df.s_index).select("value", "state").distinct().withColumnRenamed("value", "id")

# COMMAND ----------

display(joinedStates)

# COMMAND ----------

joinedStates.write.mode("overwrite").option("mergeSchema", True).format("delta").saveAsTable("ben_doan_databricks_com.doan_turbine_demo_states_gold")

# COMMAND ----------


