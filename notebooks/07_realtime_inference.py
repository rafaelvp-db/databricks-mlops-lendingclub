# Databricks notebook source
import os
import requests
import numpy as np
import pandas as pd
import json
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()
model_name = "LendingClubScoringModelRVP"
model_details = client.get_registered_model(name = model_name)
print(f"Model details for {model_name}:\n {model_details}")

# COMMAND ----------

input_json = {
  "term": "30",
  "home_ownership": "0",
  "purpose": "0",
  "addr_state":	"0",
  "verification_status": "0",
  "application_type": "0",
  "loan_amnt": "100000.0",
  "emp_length":	"0.5",
  "annual_inc":	10000.0,
  "dti": 0.9,
  "delinq_2yrs": 0.0,
  "revol_util":	0.0,
  "total_acc": 0.0,
  "credit_length_in_years":	5.0,
  "int_rate": 0.05,
  "net": 0.5,
  "issue_year": 2015.0
}

payload = {
  "dataframe_records": [input_json]
}

data = json.dumps(payload)

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

version = 7
url = f'https://adb-984752964297111.11.azuredatabricks.net/model-endpoint/{model_name}/{version}/invocations'
headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
response = requests.request(method='POST', headers=headers, url=url, data=data)
if response.status_code != 200:
  raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  
response.json()

# COMMAND ----------


