from lendingclub_scoring.common import Task
from lendingclub_scoring.config.ConfigProvider import setup_mlflow_config
from lendingclub_scoring.pipelines.LendingClubTrainingPipeline import (
    LendingClubTrainingPipeline,
)


class TrainTask(Task):
    def init_adapter(self):
        self.experiment_id = setup_mlflow_config(self.conf)

    def launch(self):
        self.logger.info("Launching bootstrap job")

        p = LendingClubTrainingPipeline(self.spark, self.conf)
        p.run()

        self.logger.info("Bootstrap job finished!")


# if you're using python_wheel_task, you'll need the entrypoint function to be used in setup.py
def entrypoint():  # pragma: no cover
    task = TrainTask()
    task.launch()

# if you're using spark_python_task, you'll need the __main__ block to start the code execution
if __name__ == '__main__':
    entrypoint()
