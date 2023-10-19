# the dsl decorator tells the sdk that we are defining an Azure Machine Learning pipeline
from azure.ai.ml import dsl, Input, Output, load_component
import os
import mlflow
import sys
import pandas as pd
import logging
# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

#aml_config_dir = os.path.abspath(os.path.join(current_directory, aml_config_relative_path))

#current_directory = os.getcwd()
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', 'modules')
sys.path.append(src_dir)

from aml_config import *

ml_client = create_ml_client()


cpu_compute_target, cpu_cluster = get_compute(ml_client, compute_name="cpu-cluster", vm_size="STANDARD_E16S_V3", min_instance=0, max_instances=4)

parent_directory = '../modules/'  # Adjust this to your components directory

data_prep = load_component(source=parent_directory + 'data_prep.yaml')

@dsl.pipeline(
    compute=cpu_compute_target
    if (cpu_cluster)
    else "serverless",  # "serverless" value runs pipeline on serverless compute
    description="first pipeline",
)
def classification_pipeline(
    input_data,
    ms_threshold,
    corr_threshold,
    plot_heatmaps,
    max_unique_threshold,
    output_data,
):
    # using data_prep_function like a python call with its own inputs
    data_prep_job = data_prep(
        input_data = input_data,
        ms_threshold = ms_threshold,
        corr_threshold = corr_threshold,
        plot_heatmaps = plot_heatmaps,
        max_unique_threshold = max_unique_threshold
    )
    
    #data_prep_job.outputs.output_data = Output(type='uri_folder', path=output_data, mode='rw_mount')
    # keys will code for the pipeline output identifier
    return {
        "pipeline_job_train_data": data_prep_job.outputs.output_data,
    }

parent_dir = "."

pipeline = classification_pipeline(
    input_data=Input(type="uri_folder", path= parent_dir + "/data/"),
    output_data=Output(type="uri_folder", path="processed_data"),
    ms_threshold = 10,
    corr_threshold = 0.8,
    plot_heatmaps = True,
    max_unique_threshold = 0.9,
)

pipeline_job = ml_client.jobs.create_or_update(
    pipeline,
    # Project's name
    experiment_name="data_prep_component",
)
ml_client.jobs.stream(pipeline_job.name)
