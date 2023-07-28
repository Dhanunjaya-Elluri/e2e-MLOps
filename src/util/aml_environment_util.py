__author__ = "Dhanunjaya Elluri"
__email__ = "dhanunjaya@elluri.net"

import os
import logging
import traceback
from env_vars import EnvVars

from azureml.core import Environment, Workspace
from azureml.core.environment import DEFAULT_CPU_IMAGE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

env = EnvVars()


def create_or_get_environment(
    workspace: Workspace,
    environment_name: str,
    conda_dependencies_file: str,
    create_new: bool = False,
    enable_docker: bool = None,
) -> Environment:
    """
    Creates an Azure ML Environment if it doesn't exist or retrieves the existing environment.

    Args:
        workspace (azureml.core.Workspace): The Azure ML workspace object.
        environment_name (str): The name of the environment.
        conda_dependencies_file (str): The path to the conda dependencies file.
        create_new (bool): Create a new environment or not.
        enable_docker (bool): Enable docker or not.

    Returns:
        azureml.core.Environment: The Azure ML Environment.
    """
    try:
        # Get the list of existing environments in the workspace
        environments = Environment.list(workspace=workspace)
        restored_environment = environments.get(environment_name)

        # If the environment doesn't exist or create_new is True, create a new environment
        if restored_environment is None or create_new:
            conda_dependencies_path = os.path.join(
                env.deployment_folder, conda_dependencies_file
            )
            new_env = Environment.from_conda_specification(
                environment_name,
                os.path.join(env.sources_directory_train, conda_dependencies_path),
            )
            restored_environment = new_env

            # Enable Docker and set the base image for the environment if enable_docker is provided
            if enable_docker is not None:
                restored_environment.docker.enabled = enable_docker
                restored_environment.docker.base_image = DEFAULT_CPU_IMAGE

            # Register the environment in the workspace
            restored_environment.register(workspace)

        # Print the restored environment information (optional)
        if restored_environment is not None:
            logger.info(f"Restored environment: {restored_environment.name}")

        return restored_environment

    except Exception as ex:
        traceback.print_exc()
        exit(1)
