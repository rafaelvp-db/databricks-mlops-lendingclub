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

df = spark.sql("select * from robeco.loan")

df = df.select(
    "loan_status",
    "int_rate",
    "revol_util",
    "issue_d",
    "earliest_cr_line",
    "emp_length",
    "verification_status",
    "total_pymnt",
    "loan_amnt",
    "grade",
    "annual_inc",
    "dti",
    "addr_state",
    "term",
    "home_ownership",
    "purpose",
    "application_type",
    "delinq_2yrs",
    "total_acc",
)

df = df.filter(
    df.loan_status.isin(["Default", "Charged Off", "Fully Paid"])
).withColumn("bad_loan", (~(df.loan_status == "Fully Paid")).cast("string"))

df = (
    df.withColumn("int_rate", regexp_replace("int_rate", "%", "").cast("float"))
    .withColumn(
        "revol_util", regexp_replace("revol_util", "%", "").cast("float")
    )
    .withColumn("issue_year", substring(df.issue_d, 5, 4).cast("double"))
    .withColumn(
        "earliest_year", substring(df.earliest_cr_line, 5, 4).cast("double")
    )
)
df = df.withColumn("credit_length_in_years", (df.issue_year - df.earliest_year))

df = df.withColumn(
    "emp_length",
    trim(regexp_replace(df.emp_length, "([ ]*+[a-zA-Z].*)|(n/a)", "")),
)
df = df.withColumn(
    "emp_length", trim(regexp_replace(df.emp_length, "< 1", "0"))
)
df = df.withColumn(
    "emp_length",
    trim(regexp_replace(df.emp_length, "10\\+", "10")).cast("float"),
)

df = df.withColumn(
    "verification_status",
    trim(regexp_replace(df.verification_status, "Source Verified", "Verified")),
)

df = df.withColumn("net", round(df.total_pymnt - df.loan_amnt, 2))
df = df.select(
    "bad_loan",
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
)

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
  model_name = "my_model"
  mlflow.sklearn.log_model(
      cl, "model", registered_model_name=model_name, signature=signature
  )
  mlflow.set_tag("action", "training")
  mlflow.sklearn.eval_and_log_metrics(cl, x_test, y_test, prefix="val_")
