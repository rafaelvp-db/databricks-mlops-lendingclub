# Databricks notebook source
import requests

def enable_endpoint(host, token, model_name):

  auth_header = {"Authorization": "Bearer " + token}
  endpoint_path = "/api/2.0/mlflow/endpoints-v2/enable"
  payload = {"registered_model_name": model_name}
  full_url = f"{host}{endpoint_path}"
  response = requests.post(
      url=full_url,
      json=payload,
      headers=auth_header
  )

  if response.status_code != 200:
      raise ValueError(f"Error making POST request to Mlflow API - [{response.status_code}]: {response.text}")
  return True

# COMMAND ----------

host = f"https://{spark.conf.get('spark.databricks.workspaceUrl')}"
model_name = "lending_club_random_forest"
token = (
  dbutils
    .notebook
    .entry_point
    .getDbutils()
    .notebook()
    .getContext()
    .apiToken()
    .get()
)
enable_endpoint(host, token, model_name)

# COMMAND ----------


