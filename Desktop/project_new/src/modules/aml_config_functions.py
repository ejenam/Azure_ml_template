import os
import sys

from azure.identity import AzureCliCredential
from azureml.core.compute import AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core import Workspace
import json
from azure.ai.ml import MLClient


#def create_ml_client(subscription_id: str, resource_group: str, workspace_name: str, tenant_id: str = None):
def create_ml_client():
    """
    Create an Azure Machine Learning workspace client.

    This function attempts to create an Azure Machine Learning workspace client using the provided parameters. If it fails
    to create a client, it generates a new configuration file with the provided parameters and tries again.

    Parameters:
        subscription_id (str): Azure subscription ID.
        resource_group (str): Azure resource group name.
        workspace_name (str): Azure Machine Learning workspace name.
        tenant_id (str, optional): Azure Active Directory tenant ID. Default is None.

    Returns:
        azureml.core.Workspace: An Azure Machine Learning workspace client.
    """
    # Create an Azure CLI credential
    credentials = AzureCliCredential(tenant_id='6aa8da55-4c6f-496e-8fc1-de0f7819b03b')
    
    try:
        # Try to create the Azure Machine Learning workspace client using provided parameters
        ml_client = Workspace.from_config(auth=credentials)
    except Exception as ex:
        print("An error occurred while creating the AML client:", str(ex))
        print("Creating a new configuration file...")

        # Define the workspace configuration based on the provided parameters
        client_config = {
            "subscription_id": "1ebe1808-a398-4ab0-b17c-1e3649ea39d5",
            "resource_group": "practice_resource",
            "workspace_name": "practice_workspace",
        }

        # Write the configuration to a JSON file
        config_path = "../config.json"
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as fo:
            json.dump(client_config, fo)
        
        # Try to create the Azure Machine Learning workspace client again
        ml_client = MLClient.from_config(credential=credentials, path=config_path)
        # Try to create the Azure Machine Learning workspace client again
        #ml_client = Workspace.from_config(path=config_path)
    return ml_client
   


def get_compute(ml_client, compute_name:str, vm_size:str, min_instance:int, max_instances:int):
    ml_client = create_ml_client()
    # specify aml compute name.
    cpu_compute_target = compute_name
    
    try:
        cpu_cluster = ml_client.compute.get(cpu_compute_target)
        print(f'Using existing compute target: {cpu_compute_target}')
    except KeyError:
        print(f"Creating a new cpu compute target: {cpu_compute_target}...")
        cpu_cluster = AmlCompute(
            name = cpu_compute_target,
            size=vm_size,
            min_nodes=min_instance,
            max_nodes=max_instances
        )
        ml_client.compute.begin_create_or_update(compute).result()
        
    return cpu_compute_target, cpu_cluster   
