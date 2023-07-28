__author__ = "Dhanunjaya Elluri"
__email__ = "dhanunjaya@elluri.net"

from util.env_vars import EnvVars
import logging

from azureml.core import Workspace, Dataset, Datastore


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

env = EnvVars()


def get_workspace() -> Workspace:
    """Get the Azure ML workspace.
    Returns:
        Workspace: The Azure ML workspace.
    """
    return Workspace(env.subscription_id, env.resource_group, env.workspace_name)


def upload_dataset_to_datastore() -> Datastore:
    """Upload the dataset to the datastore.
    Returns:
        Datastore: The Azure ML datastore.
    """
    datastore = Datastore.get(get_workspace(), get_datastore_name())
    logger.info(
        f"Uploading the zip file {env.zip_file_path} to the Datastore {datastore.name}..."
    )
    datastore.upload_files(
        files=[env.zip_file_path],
        target_path=env.dataset_name,
        overwrite=True,
        show_progress=True,
    )
    logger.info("Zip file uploaded successfully.")
    return datastore


def register_dataset(datastore: Datastore, path_on_datastore: str) -> Dataset:
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
    logger.info("Creating the Dataset from the uploaded zip file...")
    dataset = dataset.register(
        workspace=get_workspace(),
        name=env.dataset_name,
        description="Consumer Complaints Dataset",
        tags={"format": "CSV"},
        create_new_version=True,
    )
    logger.info("Dataset created successfully.")
    return dataset


def get_datastore_name() -> str:
    """Get the datastore name from the environment variables or use the default datastore.
    Returns:
        str: The datastore name.
    """
    if env.datastore_name:
        datastore_name = env.datastore_name
    else:
        datastore_name = env.workspace.get_default_datastore().name
    return datastore_name
