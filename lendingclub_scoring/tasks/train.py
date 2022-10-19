from lendingclub_scoring.config.ConfigProvider import setup_mlflow_config
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from lendingclub_scoring.data.data_provider import LendingClubDataProvider
from typing import Dict
from lendingclub_scoring.common import Task


class TrainingPipeline:
    def __init__(self, spark: SparkSession, conf: Dict[str, str], limit=100000):
        self.spark = spark
        self.conf = conf
        self.input_path = self.conf["data-path"]
        self.model_name = self.conf["model-name"]
        self.limit = limit
        self.data_provider = LendingClubDataProvider(spark, self.input_path, limit)

    def run(self):
        x_train, x_test, y_train, y_test = self.data_provider.run()
        self.train(x_train, x_test, y_train, y_test)

    def train(self, x_train, x_test, y_train, y_test):
        mlflow.set_experiment(self.conf['experiment-path'])
        mlflow.sklearn.autolog()
        # cl = LogisticRegression(random_state=42, max_iter=10)
        with mlflow.start_run(run_name="Training"):
            cl = RandomForestClassifier(n_estimators=20)
            cl.fit(x_train, y_train)
            pred = cl.predict(x_test)
            roc_auc = roc_auc_score(y_test, pred)
            mlflow.log_metric("roc_auc_val", roc_auc)
            signature = infer_signature(x_train, y_train)
            _model_name = None
            if self.conf.get("training_promote_candidates", False):
                _model_name = self.model_name
            mlflow.sklearn.log_model(
                cl, "model", registered_model_name=_model_name, signature=signature
            )
            mlflow.set_tag("action", "training")


class TrainTask(Task):
    def init_adapter(self):
        self.experiment_id = setup_mlflow_config(self.conf)

    def launch(self):
        self.logger.info("Launching training job")

        p = TrainingPipeline(self.spark, self.conf)
        p.run()

        self.logger.info("Training job finished!")


# if you're using python_wheel_task, you'll need the entrypoint function to be used in setup.py
def entrypoint():  # pragma: no cover
    task = TrainTask()
    task.launch()

# if you're using spark_python_task, you'll need the __main__ block to start the code execution
if __name__ == '__main__':
    entrypoint()
