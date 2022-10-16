# Databricks notebook source
import databricks.automl
import mlflow
import pandas
import sklearn.metrics

df = spark.sql("select * from finance.lending_club_features")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Run AutoML

# COMMAND ----------

df.groupBy("loan_status").count().show()

# COMMAND ----------

clean_df = df.filter("loan_status not in ('null', 'Oct-2015')")
print(f"Before cleaning: {df.count()} records")
print(f"After cleaning:", clean_df.count(), "records")
clean_df.write.saveAsTable("finance.silver_loans")

# COMMAND ----------

dfsummary = databricks.automl.classify(
  dataset = clean_df,
  target_col = "loan_status",
  exclude_columns = ["id"],
  primary_metric = "f1",
  timeout_minutes = 30,
)

# COMMAND ----------

