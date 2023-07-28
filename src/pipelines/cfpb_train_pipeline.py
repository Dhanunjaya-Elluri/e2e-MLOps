__author__ = "Dhanunjaya Elluri"
__email__ = "dhanunjaya@elluri.net"

import os
import logging
from src.util.aml_helpers import register_dataset, upload_dataset_to_datastore

from util.env_vars import EnvVars
from util.aml_helpers import get_dataset, get_workspace, get_datastore_name
from util.aml_environment_util import create_or_get_environment
from util.aml_compute_target import create_compute_target

from azureml.core import Experiment
from azureml.core.runconfig import RunConfiguration
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.core.graph import PipelineParameter
from azureml.pipeline.steps import PythonScriptStep


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

env = EnvVars()


def create_training_step(
    dataset, model_output, compute_target, script_path, script_name, environment
):
    return PythonScriptStep(
        name="Train Model",
        script_name=script_name,
        arguments=[
            "--input-data",
            dataset.as_named_input("input_data"),
            "--output-dir",
            model_output,
        ],
        inputs=[dataset.as_named_input("input_data")],
        outputs=[model_output],
        compute_target=compute_target,
        source_directory=script_path,
        runconfig=environment,
    )


def submit_pipeline(workspace, pipeline, experiment_name):
    experiment = Experiment(workspace, experiment_name)
    pipeline_run = experiment.submit(pipeline)
    pipeline_run.wait_for_completion(show_output=True)
    return pipeline_run


def build_and_run_pipeline(
    script_path,
    script_name,
    experiment_name,
):
    workspace = get_workspace()
    dataset = get_dataset()

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
    pipeline_data = PipelineData("pipeline_data", datastore=datastore)

    # Create a PipelineData to store the model artifacts
    model_output = PipelineData("model_output", datastore=datastore)

    # Create a compute target for model training
    compute_target = create_compute_target(
        workspace, env.compute_target_name, env.vm_size
    )

    # Create a model training step
    train_step = create_training_step(
        dataset, model_output, compute_target, script_path, script_name, environment
    )

    # Build the pipeline
    pipeline = Pipeline(workspace=workspace, steps=[train_step])

    # Submit the pipeline to run
    submit_pipeline(workspace, pipeline, experiment_name)


if __name__ == "__main__":
    env = EnvVars()
    subscription_id = env.subscription_id
    resource_group = env.resource_group
    workspace_name = env.workspace_name
    dataset_name = env.dataset_name
    datastore_name = env.datastore_name
    zip_file_path = env.zip_file_path
    compute_target_name = "YOUR_COMPUTE_TARGET_NAME"
    vm_size = "Standard_NC6s_v3"
    environment_name = "your_environment_name"
    script_path = "scripts"
    script_name = "train_model.py"
    experiment_name = "Your_Pipeline_Experiment_Name"

    build_and_run_pipeline(
        subscription_id,
        resource_group,
        workspace_name,
        dataset_name,
        datastore_name,
        zip_file_path,
        compute_target_name,
        vm_size,
        environment_name,
        script_path,
        script_name,
        experiment_name,
    )
