import argparse
from pathlib import Path
from typing_extensions import Concatenate
from uuid import uuid4
from datetime import datetime
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle
import sys
import mlflow


sys.path.insert(0, "./src/modules")

import tune_train_test as tt
import data_prep_functions as dp
import evaluation as ev

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
parser.add_argument("--test_data", default = 'test_output', type=str, help="Path to raw data")
parser.add_argument("--model_input", type=str, help="trained model")
parser.add_argument("--inference_df", type=str, default='inference_df', help="Path to store inference data")


args = parser.parse_args()

#test_df = pd.read_csv(select_first_file(args.test_data))

#test_df = pd.read_csv(args.test_data)
arr = os.listdir(args.test_data)

print(arr)

print(arr)

df_list = []
for filename in arr:
    print("reading file: %s ..." % filename)
    with open(os.path.join(args.test_data, filename), "r", encoding='latin-1') as handle:
        # print (handle.read())
        test_data = pd.read_csv((Path(args.test_data) / filename), encoding='latin-1')
        
        

print(test_data.shape)
print(test_data.columns)


dp.check_missing_values(test_data, artifact_save_dir='artefacts')
df_dropped = dp.drop_high_cardinality_features(test_data, max_unique_threshold=0.9)
df_missing_dropped = dp.replace_missing_values(df_dropped, ms_threshold=10)
test_df = dp.drop_highly_correlated_features(df_missing_dropped, corr_threshold=0.8, plot_heatmaps=True)


# Assuming 'test_data' contains both features and the target column
X_test = test_df.drop(columns=['Machine failure'])
y_test = test_df['Machine failure']

# load mlflow model
model = mlflow.sklearn.load_model(args.model_input)

inference_df, inference_col_name, predictions = tt.predict_model(model, X_test, 'predictions')
classes = [0, 1]
ev.plot_confusion_matrix(y_test, predictions, classes, model_name='HGBR', artifact_save_dir='confussion_matrix', normalize=False, title='Confussion_matrix_machine_failure_classification')
plot_name, shap_values, top_n_feature_names, top_n_feature_importances = ev.shap_feature_importance(X_test, model, model_name='HGBR', n_features=5, artifact_save_dir='shap_importance')
ev.shap_dependence_plots(X_test, top_n_feature_names, model=model, model_name=None, shap_values=shap_values, artifact_save_dir='shap_dependence_plots')
true_labels = y_test.to_list()
target_names=['Class 0', 'Class 1']
evaluation_results = ev.evaluate_classification_models(model_name='HGBR', predictions=predictions, true_labels=true_labels, target_names=target_names, plot_classification_report=True, artifact_save_dir='classification_report')

inference_dir = Path(args.inference_df)
inference_dir.mkdir(parents=True, exist_ok=True)
inference_df.to_csv(inference_dir / "inference_df.csv", index=False)
