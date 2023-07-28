__author__ = "Dhanunjaya Elluri"
__email__ = "dhanunjaya@elluri.net"

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv


@dataclass(frozen=True)
class EnvVars:
    """A class to store environment variables."""

    load_dotenv()

    # Azure ML workspace variables
    workspace_name: Optional[str] = os.environ.get("WORKSPACE_NAME")
    resource_group: Optional[str] = os.environ.get("RESOURCE_GROUP")
    subscription_id: Optional[str] = os.environ.get("SUBSCRIPTION_ID")
    datastore_name: Optional[str] = os.environ.get("DATASTORE_NAME")
    dataset_name: Optional[str] = os.environ.get("DATASET_NAME")
    dataset_version: Optional[str] = os.environ.get("DATASET_VERSION")
    zip_file_path: Optional[str] = os.environ.get("ZIP_FILE_PATH")

    # AML Pipeline variables
    source_directory: Optional[str] = os.environ.get("SOURCE_DIR")
    deployment_folder: Optional[str] = os.environ.get("SOURCES_DEP_FILE_TRAIN")
    aml_environment_name: Optional[str] = os.environ.get("AML_ENV_NAME")
    aml_train_conda_dependencies_file: Optional[str] = os.environ.get(
        "AML_ENV_TRAIN_CONDA_DEP_FILE"
    )
    rebuild_environment: Optional[bool] = (
        os.environ.get("AML_REBUILD_ENVIRONMENT", "false").lower().strip() == "true"
    )

    # Compute variables
    compute_target_name: Optional[str] = os.environ.get("AML_COMPUTE_NAME")
    vm_size: Optional[str] = os.environ.get("AML_VM_SIZE")

    # Model variables
    model_name: Optional[str] = os.environ.get("MODEL_NAME")
    model_version: Optional[str] = os.environ.get("MODEL_VERSION")

    # Train Config variables
    train_script_path: Optional[str] = os.environ.get("TRAIN_SCRIPT_PATH")
    evaluate_script_path: Optional[str] = os.environ.get("EVALUATE_SCRIPT_PATH")
    register_script_path: Optional[str] = os.environ.get("REGISTER_SCRIPT_PATH")
    score_script_path: Optional[str] = os.environ.get("SCORE_SCRIPT_PATH")
