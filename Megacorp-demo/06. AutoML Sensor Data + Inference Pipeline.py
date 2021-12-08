# Databricks notebook source
# MAGIC %sh
# MAGIC pip install faker
# MAGIC pip install pydbgen

# COMMAND ----------

import random
from datetime import datetime
from pyspark.sql.functions import *
from pyspark.sql.types import *
import numpy as np
import pandas as pd

def randomAN3(size = 100):
  return (np.array([random.uniform(-12.12,12.55) for i in range(size)]))

def randomAN4(size = 100):
  return (np.array([random.uniform(-14.5,18.5) for i in range(size)]))

def randomAN5(size = 100):
  return (np.array([random.uniform(-19.94,22.37) for i in range(size)]))

def randomAN6(size = 100):
  return (np.array([random.uniform(-21.72,23.02) for i in range(size)]))

def randomAN7(size = 100):
  return (np.array([random.uniform(-20.79,21.41) for i in range(size)]))

def randomAN8(size = 100):
  return (np.array([random.uniform(-45.28,41.16) for i in range(size)]))

def randomAN9(size = 100):
  return (np.array([random.uniform(-35.53,35.86) for i in range(size)]))

def randomAN10(size = 100):
  return (np.array([random.uniform(-25.51,19.11) for i in range(size)]))

def randomSpeed(size = 100):
  return (np.array([random.uniform(-0.43,5.3) for i in range(size)]))

def randomID(size = 100):
  return (np.array([random.randint(1,750) for i in range(size)]))

now = datetime.utcnow()

def genDataset(size = 100):
    data = {'AN3': randomAN3(),
            'AN4': randomAN4(),
            'AN5': randomAN5(),
            'AN6': randomAN6(),
            'AN7': randomAN7(),
            'AN8': randomAN8(),
            'AN9': randomAN9(),
            'AN10': randomAN10(),
            'SPEED': randomSpeed(),
            'ID': randomID(),
            'TORQUE': None,
            'TIMESTAMP': now
           }
    return (pd.DataFrame(data))

df = genDataset()
df = df.astype({'ID': 'float64', 'TORQUE': 'float64'})

# COMMAND ----------

import mlflow.pyfunc

model_name = "dec21_flightschool_team2_predict_turbine_status"
model_version = 1

model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/{model_version}"
)

model.predict(df)

# COMMAND ----------

sparkDF=spark.createDataFrame(df)
sparkDF.write.format("delta").mode("append").saveAsTable("dec21_flightschool_team2.inferences_silver")
