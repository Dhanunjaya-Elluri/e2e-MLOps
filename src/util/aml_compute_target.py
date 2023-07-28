__author__ = "Dhanunjaya Elluri"
__email__ = "dhanunjaya@elluri.net"

import logging
import traceback

from env_vars import EnvVars
from azureml.core import Workspace
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.exceptions import ComputeTargetException


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

env = EnvVars()


def create_compute_target(
    workspace: Workspace, compute_target_name: str, vm_size: str
) -> ComputeTarget:
    """Creates an Azure ML Compute Target.
    Args:
        workspace (azureml.core.Workspace): The Azure ML workspace object.
        compute_target_name (str): The name of the Compute Target.
        vm_size (str): The VM size to use for the Compute Target.
        max_nodes (int): The maximum number of nodes to use for the Compute Target.
    Returns:
        azureml.core.compute.ComputeTarget: The Azure ML Compute Target.
    """
    try:
        if compute_target_name in workspace.compute_targets:
            compute_target = workspace.compute_targets[compute_target_name]
            if compute_target and type(compute_target) is AmlCompute:
                print(f"Found existing compute target {compute_target_name}...")
                print(f"Using existing compute target {compute_target_name}...")

        else:
            compute_config = AmlCompute.provisioning_configuration(
                vm_size=vm_size, max_nodes=env.max_nodes
            )
            compute_target = ComputeTarget.create(
                workspace, compute_target_name, compute_config
            )
            compute_target.wait_for_completion(
                show_output=True, min_node_count=None, timeout_in_minutes=20
            )
        return compute_target

    except ComputeTargetException:
        logger.error("Error occurred while creating compute target.")
        traceback.print_exc()
        exit(1)
