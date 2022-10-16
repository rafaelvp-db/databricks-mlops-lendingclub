from lendingclub_scoring.common import Task
from lendingclub_scoring.config.ConfigProvider import setup_mlflow_config
from lendingclub_scoring.pipelines.LendingClubModelEvaluationPipeline import (
    LendingClubModelEvaluationPipeline,
)


class ModelEvalJob(Task):
    def init_adapter(self):
        pass

    def launch(self):
        self.logger.info("Launching bootstrap job")

        experiment_id = setup_mlflow_config(self.conf)
        p = LendingClubModelEvaluationPipeline(self.spark, experiment_id, self.conf)
        p.run()

        self.logger.info("Bootstrap job finished!")


if __name__ == "__main__":
    job = ModelEvalJob()
    job.launch()
