import argparse
from pathlib import Path
from typing_extensions import Concatenate
from uuid import uuid4
from datetime import datetime
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from pathlib import Path
from azureml.core import Workspace, Experiment, Run, Dataset, Datastore
import pandas as pd
import pickle
import sys
import mlflow

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, './.', 'modules')
sys.path.append(src_dir)

sys.path.insert(0, "./src/modules")
# Now you can import modules from function_dir
import tune_train_test as tt
import aml_config_functions as acf

def select_first_file(path):
    """Selects first file in folder, use under assumption there is only one file in folder
    Args:
        path (str): path to directory or file to choose
    Returns:
        str: full path of selected file
    """
    files = os.listdir(path)
    return os.path.join(path, files[0])



parser = argparse.ArgumentParser("train")
#parser.add_argument("--input_data", type=str, help="Path to raw data")
parser.add_argument("--model_output", type=str, help="name of trained data")
parser.add_argument("--train_data", type=str, help="Path of train data")
#parser.add_argument("--test_output", type=str, help="Path of test data")

args = parser.parse_args()

lines = [
    f"Model path: {args.model_output}",
    f"Test data path: {args.train_data}",
]

for line in lines:
    print(line)

# Load and split the test data

print("mounted_path files: ")
arr = os.listdir(args.train_data)

print(arr)

df_list = []
for filename in arr:
    print("reading file: %s ..." % filename)
    with open(os.path.join(args.train_data, filename), "r", encoding='latin-1') as handle:
        # print (handle.read())
        train_data = pd.read_csv((Path(args.train_data) / filename), encoding='latin-1')
        
#sys.path.insert(1, '/Users/ejenamvictor/Desktop/project_new/modules')
import data_prep_functions as dp

dp.check_missing_values(train_data, artifact_save_dir='artefacts')
df_dropped = dp.drop_high_cardinality_features(train_data, max_unique_threshold=0.9)
df_missing_dropped = dp.replace_missing_values(df_dropped, ms_threshold=10)
train_data = dp.drop_highly_correlated_features(df_missing_dropped, corr_threshold=0.8, plot_heatmaps=True)

# Assuming 'test_data' contains both features and the target column
X_train = train_data.drop(columns=['Machine failure'])
y_train = train_data['Machine failure']



best_params, hyperparameters_filename= tt.hyperparameter_tuning(X_train, y_train, 'train', bayesian_search=True, n_iter=30, random_seed=42)
trained_model = tt.train_model(X_train, y_train, 'HGBR.pkl', hyperparam_filename=hyperparameters_filename)


mlflow.sklearn.save_model(trained_model, args.model_output)
    
# Registering the model to the workspace
print("Registering the model via MLFlow")
mlflow.sklearn.log_model(
    sk_model=trained_model,
    registered_model_name='HGBC',
    artifact_path='HGBC',
)

# Saving the model to a file
mlflow.sklearn.save_model(
    sk_model=trained_model,
    path=os.path.join(args.model_output, 'trained_model'),
)
    
