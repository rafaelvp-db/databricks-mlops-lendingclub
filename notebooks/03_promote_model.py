# Databricks notebook source
from mlflow.tracking import MlflowClient

model_name = "lending_club_credit_default"
client = MlflowClient()
model_version = client.get_registered_model(model_name)
latest_version = model_version.latest_versions[-1]
latest_version

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Basic Sanity Checks

# COMMAND ----------

# DBTITLE 1,Model Version Description
# Check if there's a mode description
if latest_version.description == '':
  raise ValueError("Invalid model description - please add a valid text description for the model")


# COMMAND ----------

client.update_model_version(name = model_name, description = "This is a LightGBM model for predicting loan credit risk", version = latest_version.version)

# COMMAND ----------

# Check if there's a mode description
model_version = client.get_registered_model(model_name)
latest_version = model_version.latest_versions[-1]

if latest_version.description == '':
  raise ValueError("Invalid model description - please add a valid text description for the model")
else:
  print(f"Description OK: {latest_version.description}")

# COMMAND ----------

# DBTITLE 1,Model Version Tags
# Check if there's a mode description
if latest_version.tags == {}:
  raise ValueError("Invalid model tags - please add a tag indicating the name of the delta table used for training")

# COMMAND ----------

client.set_model_version_tag(
  name = model_name,
  key = "dataset",
  value = "loanstats.train",
  version = latest_version.version
)

# COMMAND ----------

# Check if there's a mode description
model_version = client.get_registered_model(model_name)
latest_version = model_version.latest_versions[-1]

if latest_version.tags == {}:
  raise ValueError("Invalid model tags - please add a tag indicating the name of the delta table used for training")
else:
  print(f"Tags OK: {latest_version.tags}")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Basic Tests

# COMMAND ----------

import mlflow

def test_prediction(run_id = "d39001405ae94d18986c4213399d771f"):
  model = mlflow.pyfunc.load_model(model_uri = f"runs:/{run_id}/model")
  test_data = table("loanstats_train").toPandas()
  pred = model.predict(test_data.drop("bad_loan", axis=1).sample(10))
  print(f"Predictions: {pred}")
  assert pred is not None
  
test_prediction()

# COMMAND ----------

# DBTITLE 1,Transition to Staging
client.transition_model_version_stage(model_name, latest_version.version, stage="Staging")
