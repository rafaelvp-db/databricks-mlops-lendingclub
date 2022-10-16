from lendingclub_scoring.common import Task
from lendingclub_scoring.config.ConfigProvider import setup_mlflow_config
import mlflow
import mlflow.sklearn

from lendingclub_scoring.data.DataProvider import LendingClubDataProvider


class ScorePipeline:
    def __init__(self, spark, input_path, output_path, model_name, stage, limit=None):
        self.spark = spark
        self.input_path = input_path
        self.output_path = output_path
        self.model_name = model_name
        self.limit = limit
        self.stage = stage
        self.data_provider = LendingClubDataProvider(spark, input_path, limit)

    def run(self):
        df = self.data_provider.load_and_transform_data_consumer()
        df.createOrReplaceTempView("loans")

        self.spark.udf.register(
            "model",
            mlflow.pyfunc.spark_udf(
                self.spark,
                model_uri=f"models:/{self.model_name}/{self.stage}",
            ),
        )

        self.spark.sql(
            """
        select *, model(term, home_ownership, purpose, addr_state, verification_status, application_type, loan_amnt, 
        emp_length,  annual_inc,dti, delinq_2yrs, revol_util, total_acc, credit_length_in_years, int_rate, net,
         issue_year) as prediction 
        from loans
        """
        ).write.format("delta").mode("overwrite").save(self.output_path)



class ScoreTask(Task):
    def init_adapter(self):
        pass

    def launch(self):
        self.logger.info("Launching bootstrap job")

        setup_mlflow_config(self.conf)
        p = ScorePipeline(
            self.spark,
            self.conf["data-path"],
            self.conf["output-path"],
            self.conf["model-name"],
            self.conf["consumer_stage"],
        )
        p.run()
        self.spark.read.load(self.conf["output-path"]).show(10, False)
        self.logger.info("Bootstrap job finished!")


# if you're using python_wheel_task, you'll need the entrypoint function to be used in setup.py
def entrypoint():  # pragma: no cover
    task = ScoreTask()
    task.launch()

# if you're using spark_python_task, you'll need the __main__ block to start the code execution
if __name__ == '__main__':
    entrypoint()