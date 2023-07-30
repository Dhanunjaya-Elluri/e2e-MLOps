__author__ = "Dhanunjaya Elluri"
__email__ = "dhanunjaya@elluri.net"

import logging

from azureml.core import RunConfiguration
from azureml.pipeline.core import Pipeline, PipelineData, PipelineParameter

from util.aml_helpers import PipelineHelper
from util.env_vars import EnvVars
from util.aml_environment_util import create_or_get_environment
from util.aml_compute_target import create_compute_target

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PipelineBuilder:
    """Class for building the pipeline."""

    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.env = EnvVars()
        self.pipeline_helper = PipelineHelper()

    def submit_pipeline(self, steps: list) -> Pipeline:
        """Submit the pipeline.
        Args:
            steps (list): The list of steps in the pipeline.
        Returns:
            Pipeline: The Azure ML pipeline run.
        """
        pipeline = Pipeline(self.workspace, steps=steps)
        pipeline._set_experiment_name
        pipeline.validate()
        published_pipeline = pipeline.publish(
            self.env.pipeline_name,
            description=self.env.pipeline_description,
            version=self.env.pipeline_version,
        )
        return published_pipeline

    def build_and_run_pipeline(self):
        """Build and run the pipeline."""
        workspace = self.pipeline_helper.get_workspace()
        dataset = self.pipeline_helper.get_dataset()

        # Create a compute target for model training
        compute_target = create_compute_target(
            workspace, self.env.compute_target_name, self.env.vm_size
        )

        # Define the Python environment for model training
        environment = create_or_get_environment(
            workspace,
            self.env.aml_environment_name,
            self.env.aml_train_conda_dependencies_file,
            create_new=self.env.rebuild_environment,
        )

        run_config = RunConfiguration()
        run_config.environment = environment

        run_config.environment.environment_variables[
            "DATASTORE_NAME"
        ] = self.pipeline_helper.get_datastore_name()

        # PipelineParameters
        # Create a PipelineParameter for the model name
        model_name = PipelineParameter(
            name="model_name", default_value=self.env.model_name
        )

        # Create a PipelineParameter for the model version
        model_version = PipelineParameter(
            name="model_version", default_value=self.env.model_version
        )

        # Create a PipelineParameter for caller_run_id
        caller_run_id = PipelineParameter(name="caller_run_id", default_value=None)

        # Create a PipelineParameter for data_file_path
        data_file_path = PipelineParameter(name="data_file_path", default_value=None)

        if self.env.dataset_name not in workspace.datasets:
            # Upload the dataset to the Datastore
            datastore = self.pipeline_helper.upload_dataset_to_datastore()
            # Register the dataset
            dataset = self.pipeline_helper.register_dataset(
                datastore, self.env.dataset_name
            )

        # Create a PipelineData to pass data between steps
        pipeline_data = PipelineData(
            "pipeline_data", datastore=workspace.get_default_datastore()
        )

        # Create training step
        train_step = self.pipeline_helper.create_train_step(
            compute_target=compute_target,
            environment=environment,
            run_config=run_config,
            dataset=dataset,
            model_name=model_name,
            model_version=model_version,
            caller_run_id=caller_run_id,
            data_file_path=data_file_path,
            pipeline_data=pipeline_data,
        )
        logger.info("Created training step")

        # Create evaluation step
        evaluation_step = self.pipeline_helper.create_evaluation_step(
            compute_target=compute_target,
            run_config=run_config,
            model_name_param=model_name,
        )
        logger.info("Created evaluation step")

        # Create register step
        register_step = self.pipeline_helper.create_register_step(
            compute_target=compute_target,
            pipeline_data=pipeline_data,
            run_config=run_config,
            model_name_param=model_name,
        )
        logger.info("Created register step")

        # Check if run_evaluation is set to True
        if self.env.run_evaluation.lower() == "true":
            # Add evaluation step to the pipeline
            logger.info("Adding evaluation step to the pipeline")
            evaluation_step.run_after(train_step)
            register_step.run_after(evaluation_step)
            steps = [train_step, evaluation_step, register_step]
        else:
            # Add register step to the pipeline
            logger.info("Adding register step to the pipeline")
            register_step.run_after(train_step)
            steps = [train_step, register_step]

        # Submit the pipeline to run
        published_pipeline = self.submit_pipeline(steps)
        logger.info(f"Published pipeline: {published_pipeline.name}")
        logger.info(f"for build {published_pipeline.version}")


if __name__ == "__main__":
    experiment_name = "Your_Pipeline_Experiment_Name"
    pipeline_builder = PipelineBuilder(experiment_name)
    pipeline_builder.build_and_run_pipeline()
