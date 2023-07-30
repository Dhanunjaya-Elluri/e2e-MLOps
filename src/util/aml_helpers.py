__author__ = "Dhanunjaya Elluri"
__email__ = "dhanunjaya@elluri.net"

import logging

from azureml.core import Workspace, Dataset, Datastore
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.compute import ComputeTarget
from azureml.core.runconfig import RunConfiguration
from azureml.pipeline.core import PipelineData, PipelineParameter

from util.env_vars import EnvVars


class PipelineHelper:
    """Helper class for creating the pipeline."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        self.env = EnvVars()
        self.workspace = self.get_workspace()

    def get_workspace(self) -> Workspace:
        """Get the Azure ML workspace.
        Returns:
            Workspace: The Azure ML workspace.
        """
        return Workspace(
            self.env.subscription_id, self.env.resource_group, self.env.workspace_name
        )

    def upload_dataset_to_datastore(self) -> Datastore:
        """Upload the dataset to the datastore.
        Returns:
            Datastore: The Azure ML datastore.
        """
        datastore = Datastore.get(self.workspace, self.get_datastore_name())
        self.logger.info(
            f"Uploading the zip file {self.env.zip_file_path} to the",
            "Datastore {datastore.name}...",
        )
        datastore.upload_files(
            files=[self.env.zip_file_path],
            target_path=self.env.dataset_name,
            overwrite=True,
            show_progress=True,
        )
        self.logger.info("Zip file uploaded successfully.")
        return datastore

    def register_dataset(self, datastore: Datastore, path_on_datastore: str) -> Dataset:
        """Register the dataset.
        Args:
            datastore (Datastore): The Azure ML datastore.
            path_on_datastore (str): The path to the dataset on the datastore.
        Returns:
            Dataset: The registered dataset.
        """
        dataset = Dataset.Tabular.from_delimited_files(
            path=[(datastore, path_on_datastore)]
        )
        self.logger.info("Creating the Dataset from the uploaded zip file...")
        dataset = dataset.register(
            workspace=self.workspace,
            name=self.env.dataset_name,
            description="Consumer Complaints Dataset",
            tags={"format": "CSV"},
            create_new_version=True,
        )
        self.logger.info("Dataset created successfully.")
        return dataset

    def get_datastore_name(self) -> str:
        """Get the datastore name from the environment variables or use the default
        datastore.
        Returns:
            str: The datastore name.
        """
        if self.env.datastore_name:
            datastore_name = self.env.datastore_name
        else:
            datastore_name = self.workspace.get_default_datastore().name
        return datastore_name

    def create_train_step(
        self,
        compute_target: ComputeTarget,
        pipeline_data: PipelineData,
        run_config: RunConfiguration,
        model_name_param: PipelineParameter,
        dataset_name: str,
        dataset_version_param: PipelineParameter,
        data_file_path_param: PipelineParameter,
        caller_run_id_param: PipelineParameter,
    ) -> PythonScriptStep:
        """Create the training step.
        Args:
            compute_target (ComputeTarget): The compute target.
            pipeline_data (PipelineData): The pipeline data.
            model_name_param (PipelineParameter): The model name parameter.
            dataset_name (str): The dataset name.
            dataset_version_param (PipelineParameter): The dataset version parameter.
            data_file_path_param (PipelineParameter): The data file path parameter.
            caller_run_id_param (PipelineParameter): The caller run ID parameter.
        Returns:
            PythonScriptStep: The PythonScriptStep.
        """

        return PythonScriptStep(
            name="Train Complaints Model",
            script_name=self.env.train_script_path,
            arguments=[
                "--model_name",
                model_name_param,
                "--step-output",
                pipeline_data,
                "--output-dir",
                self.env.model_output,
                "--dataset-name",
                dataset_name,
                "--dataset_version",
                dataset_version_param,
                "--data_file_path",
                data_file_path_param,
                "--caller_run_id",
                caller_run_id_param,
            ],
            inputs=[self.env.dataset.as_named_input("input_data")],
            outputs=[pipeline_data],
            compute_target=compute_target,
            source_directory=self.env.source_directory,
            runconfig=run_config,
            allow_reuse=True,
        )

    def create_evaluation_step(
        self,
        compute_target: ComputeTarget,
        model_name_param: PipelineParameter,
        run_config: RunConfiguration,
    ) -> PythonScriptStep:
        """Create the evaluation step.
        Args:
            compute_target (ComputeTarget): The compute target.
            model_name_param (PipelineParameter): The model name parameter.
            run_config (RunConfiguration): The run configuration.
        Returns:
            PythonScriptStep: The PythonScriptStep.
        """

        return PythonScriptStep(
            name="Evaluate Complaints Model",
            script_name=self.env.evaluate_script_path,
            arguments=[
                "--model_name",
                model_name_param,
                "--allow_run_cancel",
                self.env.allow_run_cancel,
            ],
            compute_target=compute_target,
            source_directory=self.env.source_directory,
            runconfig=run_config,
            allow_reuse=True,
        )

    def create_register_step(
        self,
        compute_target: ComputeTarget,
        pipeline_data: PipelineData,
        run_config: RunConfiguration,
        model_name_param: PipelineParameter,
    ) -> PythonScriptStep:
        """Create the register step.
        Args:
            compute_target (ComputeTarget): The compute target.
            pipeline_data (PipelineData): The pipeline data.
            run_config (RunConfiguration): The run configuration.
            model_name_param (PipelineParameter): The model name parameter.
        Returns:
            PythonScriptStep: The PythonScriptStep.
        """

        return PythonScriptStep(
            name="Register Model ",
            script_name=self.env.register_script_path,
            compute_target=compute_target,
            source_directory=self.env.source_directory,
            inputs=[pipeline_data],
            arguments=[
                "--model_name",
                model_name_param,
                "--step_input",
                pipeline_data,
            ],
            runconfig=run_config,
            allow_reuse=False,
        )
