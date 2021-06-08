# Databricks notebook source
import ost
import warnings
import sys
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import mlflow
import mlflow.sklearn

def load_data():
  boston_dataset = load_boston()
  bdf = pd.DataFrame(boston_dataset['data'],
                     columns = boston_dataset['feature_names'])
  bdf["MEDV"] = boston_dataset['target']
  # Sanity check that it should have 13 features
  #assert bdf.shape[1] == 14
  #print("Asserted 13 Features and 1 target variable")
  #print("printing Head of boston df")
  #print(bdf.head(10))
  return bdf

# # COMMAND ----------

# Data Feature selection
def feature_selection(df):
  df = df[["RM", "LSTAT", "MEDV"]]
  return df

# # COMMAND ----------

def train_test(df):
  x = df[["RM", "LSTAT"]]
  y = df[["MEDV"]]
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 5)
  return x_train, x_test, y_train, y_test

# # COMMAND ----------

def train_model(x_train, y_train):
  lm = LinearRegression()
  model = lm.fit(x_train, y_train)
  return model

# # COMMAND ----------

# save the model to adls
# filepath ='/dbfs/mnt/inst_DataScienceProject/MSS/DEV/MSS_Workspace_Backups/linearmodel.pkl'
def save_model(filepath,model):
  pickle.dump(model, open(filepath, 'wb'))

# # COMMAND ----------

def load_model(filepath):
  loaded_model =    pickle.load(open(filepath, 'rb'))
  return loaded_model

# # COMMAND ----------

def predict(df, model):
  pred = model.predict(df)
  return pred

# # COMMAND ----------

def eval_metrices(actual, pred):
  rmse = np.sqrt(mean_squared_error(actual, pred))
  mae = mean_absolute_error(actual, pred)
  r2 = r2_score(actual, pred)
  #print("rmse = ", rmse)
  #print("mae = ", mae)
  #print("r2 = ", r2)
  return rmse, mae, r2

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

# # COMMAND ----------

df = load_data()
df = feature_selection(df)

# COMMAND ----------

with mlflow.start_run():
  x_train, x_test, y_train, y_test = train_test(df)
  model = train_model(x_train, y_train)
  pred = predict(x_test, model)
  (rmse, mae, r2) = eval_metrices(y_test, pred)
  mlflow.log_metric("rmse", rmse)
  mlflow.log_metric("r2", r2)
  mlflow.log_metric("mae", mae)
  mlflow.sklearn.log_model(model, "model")
