__author__ = "Dhanunjaya Elluri"
__email__ = "dhanunjaya@elluri.net"

import logging

from util.aml_helpers import (
    create_evaluation_step,
    create_train_step,
    register_dataset,
    upload_dataset_to_datastore,
    get_dataset,
    get_workspace,
    get_datastore_name,
)
from util.env_vars import EnvVars
from util.aml_environment_util import create_or_get_environment
from util.aml_compute_target import create_compute_target

from azureml.core import Experiment
from azureml.core.runconfig import RunConfiguration
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.core.graph import PipelineParameter


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

env = EnvVars()


def submit_pipeline(workspace, pipeline, experiment_name):
    experiment = Experiment(workspace, experiment_name)
    pipeline_run = experiment.submit(pipeline)
    pipeline_run.wait_for_completion(show_output=True)
    return pipeline_run


def build_and_run_pipeline(experiment_name):
    workspace = get_workspace()
    dataset = get_dataset()

    # Create a compute target for model training
    compute_target = create_compute_target(
        workspace, env.compute_target_name, env.vm_size
    )

    # Define the Python environment for model training
    environment = create_or_get_environment(
        workspace,
        env.aml_environment_name,
        env.aml_train_conda_dependencies_file,
        create_new=env.rebuild_environment,
    )

    run_config = RunConfiguration()
    run_config.environment = environment

    run_config.environment.environment_variables[
        "DATASTORE_NAME"
    ] = get_datastore_name()

    # PipelineParameters
    # Create a PipelineParameter for the model name
    model_name = PipelineParameter(name="model_name", default_value=env.model_name)

    # Create a PipelineParameter for the model version
    model_version = PipelineParameter(
        name="model_version", default_value=env.model_version
    )

    # Create a PipelineParameter for caller_run_id
    caller_run_id = PipelineParameter(name="caller_run_id", default_value=None)

    # Create a PipelineParameter for data_file_path
    data_file_path = PipelineParameter(name="data_file_path", default_value=None)

    if env.dataset_name not in workspace.datasets:
        # Upload the dataset to the Datastore
        datastore = upload_dataset_to_datastore()
        # Register the dataset
        dataset = register_dataset(datastore, env.dataset_name)

    # Create a PipelineData to pass data between steps
    pipeline_data = PipelineData(
        "pipeline_data", datastore=workspace.get_default_datastore()
    )

    # Create training step
    train_step = create_train_step(
        compute_target=compute_target,
        environment=environment,
        dataset=dataset,
        model_name=model_name,
        model_version=model_version,
        caller_run_id=caller_run_id,
        data_file_path=data_file_path,
        pipeline_data=pipeline_data,
    )
    logger.info("Created training step")

    # Create evaluation step
    evaluation_step = create_evaluation_step(
        compute_target=compute_target,
        run_config=run_config,
        model_name_param=model_name,
    )
    logger.info("Created evaluation step")

    # Create a PipelineData to store the model artifacts
    # model_output = PipelineData("model_output", datastore=datastore)

    # Build the pipeline
    pipeline = Pipeline(workspace=workspace, steps=[train_step, evaluation_step])

    # Submit the pipeline to run
    submit_pipeline(workspace, pipeline, experiment_name)


if __name__ == "__main__":
    experiment_name = "Your_Pipeline_Experiment_Name"

    build_and_run_pipeline(
        experiment_name,
    )
