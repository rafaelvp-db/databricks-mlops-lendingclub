import mlflow
from mlflow.tracking import MlflowClient
import requests
from lendingclub_scoring.common import Task
from lendingclub_scoring.data.data_provider import LendingClubDataProvider
from pyspark.dbutils import DBUtils


class ModelDeploymentPipeline:
    def __init__(
        self,
        model_name,
        token,
        host
    ):
        self._model_name = model_name
        self._token = token
        self._host = host


    def _enable_endpoint(self):

        auth_header = {"Authorization": "Bearer " + self._token}
        endpoint_path = "/api/2.0/mlflow/endpoints-v2/enable"
        payload = {"registered_model_name": self._model_name}
        full_url = f"{self._host}{endpoint_path}"
        response = requests.post(
            url=full_url,
            json=payload,
            headers=auth_header
        )

        if response.status_code != 200:
            raise ValueError(f"Error making POST request to Mlflow API - [{response.status_code}]: {response.text}")

    def run(self):
        self._enable_endpoint()


class ModelDeploymentTask(Task):

    def launch(self):
        self.logger.info("Launching bootstrap job")
        host = f"https://{self.spark.conf.get('spark.databricks.workspaceUrl')}"
        dbutils = DBUtils(self.spark)
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
        pipeline = ModelDeploymentPipeline(
            token = token,
            host = host,
            model_name = self.conf["model-name"]
        )
        pipeline.run()
        self.logger.info("Bootstrap job finished!")


# if you're using python_wheel_task, you'll need the entrypoint function to be used in setup.py
def entrypoint():  # pragma: no cover
    task = ModelDeploymentTask()
    task.launch()

# if you're using spark_python_task, you'll need the __main__ block to start the code execution
if __name__ == '__main__':
    entrypoint()
