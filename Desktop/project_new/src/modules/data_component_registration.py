
from azure.ai.ml import load_component
import os
import sys

# Get the current working directory
current_directory = os.getcwd()

# Specify the relative paths to your directories and files
aml_config_relative_path = 'modules'  # Adjust this to your aml_config directory
#components_relative_path = 'src'  # Adjust this to your components directory
data_prep_yaml_file = 'data_prep.yaml'  # Adjust this to your data_prep.yaml file

data_prep_yaml_path = os.path.abspath(os.path.join(current_directory, data_prep_yaml_file))

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', 'modules')
sys.path.append(src_dir)

import aml_config as aml

# Loading the component from the yaml file
loaded_component_prep = load_component(source=data_prep_yaml_path)

ml_client = aml.create_ml_client()

# Now we register the component to the workspace
data_prep_component = ml_client.create_or_update(loaded_component_prep)

# Create (register) the component in your workspace
print(
    f"Component {data_prep_component.name} with Version {data_prep_component.version} is registered"
)
