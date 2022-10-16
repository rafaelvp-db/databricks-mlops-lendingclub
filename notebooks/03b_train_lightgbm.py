# Databricks notebook source
# MAGIC %md
# MAGIC ## Train LightGBM Model

# COMMAND ----------

# DBTITLE 1,Read in Training and Validation Data
train_df = table('loanstats_train').toPandas()
valid_df = table('loanstats_valid').toPandas()

# COMMAND ----------

# DBTITLE 1,Define Categorical and Numeric Features
categoricals = ["term", "home_ownership", "purpose", "addr_state",
                "verification_status","application_type"]
numerics = ["loan_amnt","emp_length", "annual_inc","dti",
            "delinq_2yrs","revol_util","total_acc",
            "credit_length_in_years"]

features = categoricals + numerics

X = train_df[features]
y = (train_df['bad_loan'] == 'true').astype('int')

X_valid = valid_df[features]
y_valid = (valid_df['bad_loan'] == 'true').astype('int')

# COMMAND ----------

display(train_df)

# COMMAND ----------

# DBTITLE 1,Define Data Preprocessing Steps
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

from lightgbm import LGBMClassifier
import numpy as np
from numpy import exp, log

preprocessor = ColumnTransformer(
  transformers=[('num', SimpleImputer(), numerics),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categoricals)])

preprocessor.fit(X)

# to help speed up hyperparameter search preprocess the training data
X_processed = preprocessor.transform(X)

# COMMAND ----------

# MAGIC %md
# MAGIC The training data set is actually quite small -- 54k rows by about 18 columns. It's trivial to fit a model to this data with standard packages like `scikit-learn` or `xgboost`. However each of these models requires tuning, and needs building of 100 or more models to find the best combination.
# MAGIC 
# MAGIC In Databricks, the tool `hyperopt` can be use to build these models on a Spark cluster in parallel. The results are logged automatically to `mlflow`.

# COMMAND ----------

import mlflow
from hyperopt import fmin, hp, tpe, SparkTrials, STATUS_OK

def create_model(params):
  return LGBMClassifier(
    max_depth = int(params['max_depth']),
    learning_rate = exp(params['log_learning_rate']),
    reg_alpha = exp(params['log_reg_alpha']),
    reg_lambda = exp(params['log_reg_lambda']),
    min_child_weight = exp(params['log_min_child_weight']),
    random_state = 0)

search_space = {
  'max_depth':            hp.quniform('max_depth', 20, 60, 1),
  # use uniform over loguniform here simply to make metrics show up better in mlflow comparison, in logspace
  'log_learning_rate':    hp.uniform('log_learning_rate', -3, 0),
  'log_reg_alpha':        hp.uniform('log_reg_alpha', -5, -1),
  'log_reg_lambda':       hp.uniform('log_reg_lambda', 1, 8),
  'log_min_child_weight': hp.uniform('log_min_child_weight', -1, 3)
}

def train_model(params):
  clf = create_model(params)
  score = np.mean(cross_val_score(clf, X_processed, y, scoring='roc_auc', cv=3))
  return {'status': STATUS_OK, 'loss': -score}

spark_trials = SparkTrials(parallelism=20)
best_params = fmin(fn=train_model, space=search_space, algo=tpe.suggest, max_evals=96, trials=spark_trials)

print('-- best parameters --')
best_params

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train Final Model
# MAGIC 
# MAGIC Train full model pipeline with best parameters and training dataset. 

# COMMAND ----------

from sys import version_info

import cloudpickle
import lightgbm
import mlflow.sklearn
import sklearn
from sklearn.metrics import roc_auc_score
from mlflow.models.signature import infer_signature

# Defining an explicit conda environment is needed since we are using a lightgbm model in a scikit-learn pipeline. If we were just saving the classifier/regressor only then it would define a basic environment automatically.

PYTHON_VERSION = f'{version_info.major}.{version_info.minor}.{version_info.micro}'

conda_env = {
  'channels': [
    'defaults',
    'conda-forge'
  ],
  'dependencies': [
    f'python={PYTHON_VERSION}',
    f'scikit-learn={sklearn.__version__}',
    f'lightgbm={lightgbm.__version__}',
    'pip',
    {
      'pip': [
        'mlflow',
        f'cloudpickle=={cloudpickle.__version__}'
      ]
    }
  ],
  'name': 'pipeline_env'
}


with mlflow.start_run() as run:  
  pipeline_model = make_pipeline(preprocessor, create_model(best_params))

  auc_train = np.mean(cross_val_score(pipeline_model, X, y, scoring='roc_auc', cv=5))
  # train final model on entire dataset
  pipeline_model.fit(X, y)
  mlflow.log_params(best_params)
  # log metrics along with model
  pred_valid = pipeline_model.predict(X_valid)
  auc_valid = roc_auc_score(y_valid, pred_valid)
  mlflow.log_metric('auc_train', auc_train)
  mlflow.log_metric('auc_valid', auc_valid)
  
  X_sample = X.iloc[0:10, :]
  # create input example that will be available in MLFlow Serving
  input_example = X.iloc[0:1, :].to_dict(orient='records')[0]
  # infer model input and output schemas that will be documented in MLFlow
  signature = infer_signature(X_sample, pipeline_model.predict(X_sample)) 
  # log model to MLFlow Tracking Server
  mlflow.sklearn.log_model(pipeline_model, artifact_path='model', 
                           conda_env=conda_env, signature=signature,
                           input_example=input_example)  
  best_run = run.info

# COMMAND ----------

## verify the number of default predictions that we should see later in the production spark scoring example
import numpy as np

pred_valid = pipeline_model.predict(X_valid)
np.asarray(np.unique(pred_valid, return_counts=True)).T

# COMMAND ----------

# MAGIC %md
# MAGIC ### Registering Model
# MAGIC 
# MAGIC This model can then be registered as the current candidate model for further evaluation in the Model Registry.  
# MAGIC https://docs.microsoft.com/en-us/azure/databricks/applications/mlflow/model-registry

# COMMAND ----------

import time

model_name = "lending_club_credit_default"
client = mlflow.tracking.MlflowClient()
try:
  client.create_registered_model(model_name)
except Exception as e:
  pass

model_version = client.create_model_version(model_name, f"{best_run.artifact_uri}/model", best_run.run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Moving to Production

# COMMAND ----------

latest_model = client.get_registered_model(model_name).latest_versions[-1]

print('latest model version:', latest_model.version)
print('latest model stage:', latest_model.current_stage)

# COMMAND ----------

# DBTITLE 1,Transition latest model version to Production stage
client.transition_model_version_stage(model_name, latest_model.version, stage="Production")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Batch Scoring Example

# COMMAND ----------

loanstats_df = table('loanstats_valid')
display(loanstats_df)

# COMMAND ----------

import mlflow.pyfunc

model_udf = mlflow.pyfunc.spark_udf(spark, 'models:/carlson_credit_default/Production', result_type='long')

features = ["term", "home_ownership", "purpose", "addr_state",
            "verification_status","application_type",
            "loan_amnt","emp_length", "annual_inc","dti",
            "delinq_2yrs","revol_util","total_acc",
            "credit_length_in_years"]

quoted_cols = list(map(lambda c: f"`{c}`", features))

loanstats_scored_df = loanstats_df.withColumn('default_prediction', model_udf(*quoted_cols))

loanstats_scored_df.groupBy('default_prediction').count().show()