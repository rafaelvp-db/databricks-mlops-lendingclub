from lendingclub_scoring.common import Task
from lendingclub_scoring.config.ConfigProvider import setup_mlflow_config
import mlflow
import mlflow.sklearn
from pyspark.sql import DataFrame
from pyspark.sql.functions import lit, col

from lendingclub_scoring.data.data_provider import LendingClubDataProvider

FEATURES = [
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


class ABTestPipeline:
    def __init__(
        self,
        spark,
        input_path,
        output_path,
        model_name,
        prod_version,
        test_version,
        limit=None,
    ):
        self.spark = spark
        self.input_path = input_path
        self.output_path = output_path
        self.model_name = model_name
        self.limit = limit
        self.prod_version = prod_version
        self.test_version = test_version
        self.data_provider = LendingClubDataProvider(spark, input_path, limit)

    def run(self):
        df = self.data_provider.load_and_transform_data_consumer()
        a_df, b_df = df.randomSplit([0.8, 0.2])

        res_df = self.score_model(a_df, self.prod_version).union(
            self.score_model(b_df, self.test_version)
        )

        res_df.write.format("delta").mode("overwrite").save(self.output_path)

    def score_model(self, df: DataFrame, version: str) -> DataFrame:
        model_uri = f"models:/{self.model_name}/{version}"
        udf = mlflow.pyfunc.spark_udf(self.spark, model_uri)
        df = df.withColumn("prediction", udf(*[col(c) for c in FEATURES])).withColumn(
            "model_version", lit(version)
        )
        return df


class ABTestTask(Task):
    def init_adapter(self):
        pass

    def launch(self):
        self.logger.info("Launching bootstrap job")

        setup_mlflow_config(self.conf)
        p = ABTestPipeline(
            self.spark,
            self.conf["data-path"],
            self.conf["output-path"],
            self.conf["model-name"],
            self.conf["prod_version"],
            self.conf["test_version"],
        )
        p.run()

        self.spark.read.load(self.conf["output-path"]).show(10, False)
        self.logger.info("Bootstrap job finished!")


# if you're using python_wheel_task, you'll need the entrypoint function to be used in setup.py
def entrypoint():  # pragma: no cover
    task = ABTestTask()
    task.launch()

# if you're using spark_python_task, you'll need the __main__ block to start the code execution
if __name__ == '__main__':
    entrypoint()
