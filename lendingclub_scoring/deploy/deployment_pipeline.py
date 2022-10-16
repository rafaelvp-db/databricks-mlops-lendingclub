import mlflow
from mlflow.tracking import MlflowClient
import requests
from pyspark.dbutils import DBUtils


class ModelDeploymentPipeline:
    def __init__(
        self,
        spark,
        db_name,
        model_name,
        experiment_name,
        run_name,
    ):
        self._model_name = model_name
        self._spark = spark
        self._db_name = db_name
        self._run_name = run_name
        self._experiment_path = self.conf['experiment-path']
        self._client = MlflowClient()
        self._host = self.spark.conf.get("spark.databricks.workspaceUrl")

    def _get_candidate(
        self,
        status="FINISHED",
        sort_by="metrics.test.f1",
        ascending=True
    ):

        experiment = mlflow.get_experiment_by_name(self._experiment_path)

        df = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"status = '{status}'"
        )

        best_run_id = df["run_id"].values[0]
        return best_run_id

    def _set_tags(
        self,
        run_id
    ):
        tags = {
            "db_table": self.conf['table']
        }

        for key, value in tags.items():
            self._client.set_tag(run_id, key=key, value=value)

    def _get_latest_version(self, stages = ["None"]):

        latest_versions = self._client.get_latest_versions(
            name=self._model_name, stages=stages
        )
        if len(latest_versions) == 0:
            return None
        
        model_version_info = latest_versions[0]
        return model_version_info

    def _promote_to_staging(
        self,
        best_run_id,
        model_description="This model if a loan will be delinquent or not.",
        version_description="This model version was built using XGBoost, \
            with the best hyperparameters set identified with HyperOpt.",
        stages = ["None", "Staging"]
    ):

        model_version_info = self._get_latest_version(stages = stages)
        if model_version_info is None:
            self.logger.info("No versions to be promoted")
            return None

        source_version = model_version_info.version
        current_stage = model_version_info.current_stage

        if best_run_id == model_version_info.run_id:

            # More details about the model
            self._client.update_registered_model(
                name=model_version_info.name, description=model_description
            )

            # Gives more details on this specific model version
            self._client.update_model_version(
                name=model_version_info.name,
                version=model_version_info.version,
                description=version_description,
            )

            target_version = model_version_info.version
            self._client.transition_model_version_stage(
                name=self._model_name,
                version=target_version,
                stage="Staging"
            )

            model_version_info.stage = "Staging"
            return model_version_info


    def _test_predictions(self, model_version_info):

        model = mlflow.sklearn.load_model(model_uri=model_version_info.source)
        valid_table = self.conf['valid_table']
        X_valid = self.spark.sql(f"select * from {valid_table}")
        pred = model.predict(X_valid.sample(10))

        if pred is None:
            raise ValueError(f"Model generated invalid predictions. Model: {model_version_info}")

        return True

    def _promote_to_production(
        self, best_run_id
    ):

        from_stage = ["Staging"]
        model_version_info = self._get_latest_version(stages = from_stage)
        target_version = None
        valid_predictions = self._test_predictions(model_version_info=model_version_info)

        if not valid_predictions:
            raise ValueError(f"Predictions from {self._model_name} in {from_stage} are invalid")

        if model_version_info is not None:
            if best_run_id != model_version_info.run_id:
                raise ValueError(f"Best run ID {best_run_id} mismatch with run_id from Staging: {model_version_info.run_id}")
            
            target_version = model_version_info.version
            self._client.transition_model_version_stage(
                name=self._model_name,
                version=target_version,
                stage="Production",
                archive_existing_versions=True,
            )
            print(
                f"""Transitioned model {self._model_name} to Production,
                    model version: {model_version_info.version}"""
            )
        else:
            print("No versions in Staging, existing...")


    def _enable_endpoint(self):

        dbutils = DBUtils(self._spark)
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
        auth_header = {"Authorization": "Bearer " + token}
        endpoint_path = "/mlflow/endpoints-v2/enable"
        payload = {"registered_model_name": self._model_name}
        full_url = f"{self._host}{endpoint_path}"
        response = requests.post(url=full_url, json=payload, headers=auth_header)

        if response.status_code != 200:
            raise ValueError(f"Error making POST request to Mlflow API: {response.text}")

    def run(self):

        best_run_id = self._get_candidate()
        self.set_tags(best_run_id)
        self._promote_to_staging(best_run_id=best_run_id)
        self._promote_to_production(best_run_id=best_run_id)
        self._enable_endpoint()
