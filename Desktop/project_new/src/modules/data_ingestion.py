from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azureml.core import Workspace, Experiment, Run, Dataset, Datastore
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

from azure.identity import DefaultAzureCredential
from azureml.core import Workspace


def load_data():
    run = Run.get_context()
    
    try:
        # pipeline run
        ws = run.experiment.workspace
    except:
        ws = Workspace.from_config()
    # Make sure that 'ds' is a valid datastore name
    #if ds is None:
        #raise ValueError("The 'ds' parameter cannot be None.")
        
    
    #def_ds = Datastore.get(ws, ds)
    def_ds = Datastore.get(ws, 'workspaceblobstore')
    #load_data('workspaceblobstore', 'ai4i2020.csv')
    raw_df = Dataset.Tabular.from_delimited_files([(def_ds,'ai4i2020.csv')]).to_pandas_dataframe()
    log.info('Data Loaded')
    
    return raw_df

def set_cwd_path(path: str):
    os.chdir(path)
    log.info(f'Current Directory set to: {os.getcwd()}')
