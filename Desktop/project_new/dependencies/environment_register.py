from azure.ai.ml.entities import Environment
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', 'modules')
sys.path.append(src_dir)


import aml_config as aml 

custom_env_name = "general_environment"

ml_client = aml.create_ml_client()
env_docker_conda = Environment(
    name=custom_env_name,
    description="Custom environment for classification and regression tasks",
    conda_file="conda.yaml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
    version="0.4.0",
)
ml_client.environments.create_or_update(env_docker_conda)
