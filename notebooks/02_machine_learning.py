# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## Training and running our model

# COMMAND ----------

from pyspark.sql import DataFrame
from pyspark.sql.functions import regexp_replace, trim, substring, round
from sklearn.model_selection import train_test_split

predictors = [
    "term",
    "home_ownership",
    "purpose",
    "addr_state",
    "verification_status",
    "application_type",
    "loan_amnt",
    "emp_length",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "revol_util",
    "total_acc",
    "credit_length_in_years",
    "int_rate",
    "net",
    "issue_year",
]
target = "bad_loan"

# COMMAND ----------

df = spark.table()

# COMMAND ----------

display(df)

# COMMAND ----------

import pandas as pd

def handle_cat_types(df: pd.DataFrame):
  for col in df.columns:
    if df.dtypes[col] == "object":
      df[col] = df[col].astype("category").cat.codes
    df[col] = df[col].fillna(0)
  return df

pandas_df = df.toPandas()
pandas_df = handle_cat_types(pandas_df)

x_train, x_test, y_train, y_test = train_test_split(pandas_df.loc[:,predictors], pandas_df[target])

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier
import mlflow
from mlflow.models import infer_signature

mlflow.sklearn.autolog()

with mlflow.start_run(run_name="Training"):
  cl = RandomForestClassifier(n_estimators=20)
  cl.fit(x_train, y_train)
  signature = infer_signature(x_train, y_train)
  model_name = "lending_club_random_forest"
  mlflow.sklearn.log_model(
      cl, "model", registered_model_name=model_name, signature=signature
  )
  mlflow.set_tag("action", "training")
  mlflow.sklearn.eval_and_log_metrics(cl, x_test, y_test, prefix="val_")

# COMMAND ----------


