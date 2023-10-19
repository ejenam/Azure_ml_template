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
#current_dir = os.path.dirname(os.path.abspath(__file__))
#src_dir = os.path.join(current_dir, '../', 'modules')
#sys.path.append(src_dir)

#from modules import data_prep_functions

#from aml_config_functions import *
sys.path.insert(1, '/Users/ejenamvictor/Desktop/project_new/modules')
import data_prep_functions as dp

parser = argparse.ArgumentParser("prep")
parser.add_argument("--raw_data", type=str, help="Path to raw data")
parser.add_argument("--prep_data", type=str, help="Path of prepped data")

args = parser.parse_args()

print("hello training world...")

lines = [f"Raw data path: {args.raw_data}", f"Data output path: {args.prep_data}"]

for line in lines:
    print(line)

print("mounted_path files: ")
arr = os.listdir(args.raw_data)
print(arr)


df_list = []
for filename in arr:
    print("reading file: %s ..." % filename)
    with open(os.path.join(args.raw_data, filename), "r") as handle:
        # print (handle.read())
        input_df = pd.read_csv((Path(args.raw_data) / filename))
        df_list.append(input_df)

prep_data = df_list[0]
print(prep_data.columns)



# Directory to save the plots
plot_save_dir = 'plots'



dp.check_missing_values(input_df, artifact_save_dir='artefacts')
df_dropped = dp.drop_high_cardinality_features(input_df, max_unique_threshold=0.9)
df_missing_dropped = dp.replace_missing_values(df_dropped, ms_threshold=10)
df_clean = dp.drop_highly_correlated_features(df_missing_dropped, corr_threshold=0.8, plot_heatmaps=True)
#train_df, test_df = dp.custom_train_test_split(df_clean, test_size=0.2, random_state=101, time_series=False)


    
clean_data = df_clean.to_csv((Path(args.prep_data) / "prep_data.csv"), index=False)
#train_df.to_csv((Path(args.train_data) / "training_data.csv"))
#train_df.to_csv((Path(args.test_data) / "testing_data.csv"))

#credit_train_df.to_csv(os.path.join(args.train_data, "data.csv"), index=False)
#X_test = X_test.to_csv((Path(args.prep_data) / "X_test.csv"))
#y_train = y_train.to_csv((Path(args.prep_data) / "y_train.csv"))
#y_test = y_test.to_csv((Path(args.prep_data) / "y_test.csv"))



