# Databricks notebook source
import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
#mlflow
import mlflow
import mlflow.sklearn

# COMMAND ----------

import sklearn
sklearn.__version__

# COMMAND ----------

import mlflow
mlflow.__version__

# COMMAND ----------

def load_data():
  boston_dataset = load_boston()
  bdf = pd.DataFrame(boston_dataset['data'],
                     columns = boston_dataset['feature_names'])
  bdf["MEDV"] = boston_dataset['target']
  # Sanity check that it should have 13 features
  assert bdf.shape[1] == 14
  print("Asserted 13 Features and 1 target variable")
  print("printing Head of boston df")
  print(bdf.head(10))
  return bdf

# COMMAND ----------

# Data Feature selection
def feature_selection(df):
  df = df[["RM", "LSTAT", "MEDV"]]
  return df

# COMMAND ----------

def train_test(df):
  X = df[["RM", "LSTAT"]]
  y = df[["MEDV"]]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 5)
  return X_train, X_test, y_train, y_test

# COMMAND ----------

def train_model(X_train, y_train):
  lm = LinearRegression()
  model = lm.fit(X_train, y_train)
  return model

# COMMAND ----------

# save the model to adls
# filepath ='/dbfs/mnt/inst_DataScienceProject/MSS/DEV/MSS_Workspace_Backups/linearmodel.pkl'
def save_model(filepath,model):
  pickle.dump(model, open(filepath, 'wb'))

# COMMAND ----------

def load_model(filepath):
  loaded_model =    pickle.load(open(filepath, 'rb'))
  return loaded_model

# COMMAND ----------

def predict(df, model):
  pred = model.predict(df)
  return pred

# COMMAND ----------

def eval_metrices(actual, pred):
  rmse = np.sqrt(mean_squared_error(actual, pred))
  mae = mean_absolute_error(actual, pred)
  r2 = r2_score(actual, pred)
  print("rmse = ", rmse)
  print("mae = ", mae)
  print("r2 = ", r2)
  return rmse, mae, r2

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

# COMMAND ----------

df = load_data()
df = feature_selection(df)

# COMMAND ----------

with mlflow.start_run():
  X_train, X_test, y_train, y_test = train_test(df)
  model = train_model(X_train, y_train)
  pred = predict(X_test, model)
  (rmse, mae, r2) = eval_metrices(y_test, pred)
  mlflow.log_metric("rmse", rmse)
  mlflow.log_metric("r2", r2)
  mlflow.log_metric("mae", mae)
  mlflow.sklearn.log_model(model, "model")
  

# COMMAND ----------

model

# COMMAND ----------

# filepath ='/dbfs/mnt/inst_DataScienceProject/MSS/DEV/MSS_Workspace_Backups/linearmodel_prod.pkl'
# save_model(filepath,model)

# COMMAND ----------

pred = predict(X_test, model)

# COMMAND ----------

pred

# COMMAND ----------

# df = spark.createDataFrame(X_test)

# COMMAND ----------

# # Prediction using pyfunc and mlflow
# import mlflow
# logged_model = 'runs:/647c7ca953034f51b74a2db3c1d3381c/model'

# # Load model as a Spark UDF.
# loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model)

# # Predict on a Spark DataFrame.
# # df.withColumn('predictions', loaded_model()).collect()
# df1 = df.withColumn('predictions', loaded_model())

# COMMAND ----------

# type(df1)

# COMMAND ----------

# # Using pandas
# import mlflow
# logged_model = 'runs:/647c7ca953034f51b74a2db3c1d3381c/model'

# # Load model as a PyFuncModel.
# loaded_model = mlflow.pyfunc.load_model(logged_model)

# # Predict on a Pandas DataFrame.
# import pandas as pd
# loaded_model.predict(pd.DataFrame(X_test))

# COMMAND ----------

# MAGIC %sh 
# MAGIC databricks fs -h

# COMMAND ----------

# MAGIC %sh ls /loac

# COMMAND ----------

# MAGIC %sh
# MAGIC databricks fs cp "dbfs:/databricks/mlflow-tracking/2376261929218913/647c7ca953034f51b74a2db3c1d3381c/artifacts/model/" "/local_disk0/tmp/jay/"

# COMMAND ----------

# # Loading model from artifacts
# lm = mlflow.sklearn.load_model("dbfs:/databricks/mlflow-tracking/2376261929218913/647c7ca953034f51b74a2db3c1d3381c/artifacts/model/")


# COMMAND ----------

# lm.predict(X_test)

# COMMAND ----------

# import mlflow
# p_uri = "dbfs:/databricks/mlflow-tracking/2376261929218913/3b3e5bc8425d4635afe3816d68f2f313/artifacts/model"
# mlflow.run(p_uri)
