from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import logging
import os
import json
import joblib
import mlflow
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Directory to save the plots
plot_save_dir = 'plots'

def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


    
def plot_confusion_matrix(y_true, y_pred, classes, model_name, artifact_save_dir, normalize=False, title=None):
    """
    This function plots the confusion matrix.

    :param y_true: True labels of the data.
    :param y_pred: Predicted labels of the data.
    :param classes: List of class labels (e.g., ['Class 0', 'Class 1']).
    :param model_name: Name of the model for plot naming.
    :param artifact_save_dir: Directory where artifacts including plots will be saved.
    :param normalize: If True, normalize the confusion matrix.
    :param title: Title of the plot.
    """
    #mlflow.end_run()
    if not title:
        if normalize:
            title = 'Normalized Confusion Matrix'
        else:
            title = 'Confusion Matrix'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cmap=plt.cm.Blues
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    # Create figure and axis
    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    # Plot the confusion matrix
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap,
                xticklabels=classes, yticklabels=classes)
    
    # Customize plot labels and appearance
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(title)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    # Display the plot
    plt.tight_layout()
    
    # Artifact save directory for plots
    if artifact_save_dir and not os.path.exists(artifact_save_dir):
        os.makedirs(artifact_save_dir)
        
    # Create the artifact_save_dir directory if it doesn't exist
    #if not os.path.exists(artifact_save_dir):
        #os.makedirs(artifact_save_dir)
        
    # Save the heatmap plot as an image
    if artifact_save_dir:
        after_plot_path = os.path.join(artifact_save_dir, f"{model_name}_confusion_matrix.png")
        plt.savefig(after_plot_path)
        log.info(f"{model_name} confusion matrix saved: %s", after_plot_path)
        
        # Log the saved plot as an artifact using MLflow
        #with mlflow.start_run():
        mlflow.log_artifact(after_plot_path, artifact_path="confussion_matrix.png")
          

    # Log the heatmap plot as an artifact using MLflow
    #mlflow.log_artifact(plot_save_path, artifact_path=f'{plot_name}.png')

    plt.show()
    


def shap_feature_importance(X_test, model, model_name:str, n_features, artifact_save_dir: str):
    model_prefix = os.path.splitext(model_name)[0]
    
    if not os.path.exists(artifact_save_dir):
        log.info(f'Creating {artifact_save_dir} directory in {os.getcwd()}')
        os.makedirs(artifact_save_dir)
    
    #log.info(f'Loading {model_name}')
    #model = joblib.load(model_name)
    
    model

    log.info('Generating SHAP Values')

    # Convert X_test to a pandas DataFrame
    X_test_df = pd.DataFrame(X_test)  # Assuming X_test is a 2D array
    
    # Identify categorical columns
    categorical_features = list(X_test_df.select_dtypes(include=['category', 'object']).columns)
    
    # Create a ColumnTransformer to apply one-hot encoding to categorical columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), categorical_features)
        ],
        remainder='passthrough'  # Keep non-categorical columns as-is
    )
    
    # Fit and transform the test data
    X_test_transformed = preprocessor.fit_transform(X_test_df)
    
    # Generate SHAP values using the transformed data
    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values(X_test_transformed)
    
    log.info('Saving SHAP Plot to Artifact Directory')
    plt.clf()
    feature_names = (
        list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)) +
        list(X_test_df.select_dtypes(exclude=['category', 'object']).columns)
    )
    shap.summary_plot(shap_values, X_test_transformed, feature_names=feature_names, show=False)
    fig = plt.gcf()

    # Artifact save directory for plots
    plot_save_dir = os.path.join(artifact_save_dir, 'shap_summary_plots')
    if not os.path.exists(plot_save_dir):
        os.makedirs(plot_save_dir)

    plot_name = os.path.join(plot_save_dir, f'{model_prefix}_SHAP_summary.png')
    plt.savefig(plot_name)
    log.info(f'{plot_name} saved.')

    # Log SHAP summary plot as an MLflow artifact
    #with mlflow.start_run():
    #mlflow.shap.log_summary(fig, feature_names=feature_names)

    log.info(f'Extracting Top {n_features} Important features for the model')

    # Calculate feature importances and sort them
    feature_importances = np.abs(shap_values).mean(axis=0)
    feature_importances_dict = dict(zip(feature_names, feature_importances))
    sorted_features = sorted(feature_importances_dict.items(), key=lambda x: x[1], reverse=True)
    
    log.info(f'Top {n_features} Important features are:')
    top_n_feature = sorted_features[:n_features]
    top_n_feature_names = [feature[0] for feature in top_n_feature]  # Extract feature names
    top_n_feature_importances = [feature[1] for feature in top_n_feature]  # Extract importances
    
    log.info(', '.join([f'{feature}: {importance}' for feature, importance in top_n_feature]))
    
    # Log SHAP summary plot as an MLflow artifact
    #with mlflow.start_run():
    mlflow.log_params({'n_features': n_features})
    mlflow.log_params({'top_n_feature_names': top_n_feature_names})
    mlflow.log_params({'top_n_feature_importances': top_n_feature_importances})
    
    # Save the SHAP summary plot as an image and log it as an artifact
    with tempfile.TemporaryDirectory() as tempdir:
        plot_name = os.path.join(tempdir, f'{model_prefix}_SHAP_summary.png')
        plt.savefig(plot_name)
        mlflow.log_artifact(plot_name, artifact_path='shap_summary_plots')
        
    
    # Log feature importances as MLflow parameters
    #mlflow.log_params({'n_features': n_features})
    #mlflow.log_params({'top_n_feature_names': top_n_feature_names})
    #mlflow.log_params({'top_n_feature_importances': top_n_feature_importances})

    return plot_name, shap_values, top_n_feature_names, top_n_feature_importances


def shap_dependence_plots(X_test: pd.DataFrame, features: list, shap_values: np.array, artifact_save_dir: str, model=None, model_name=None):
    # Start an MLflow run
    #mlflow.start_run()

    model_prefix = os.path.splitext(model_name)[0] if model_name else None
    if not os.path.exists(artifact_save_dir):
        mlflow.log_params({'artifact_save_dir': artifact_save_dir})
        os.makedirs(artifact_save_dir)
    
    # Load the model if 'model' is not provided
    if model is None:
        if model_name is None:
            raise ValueError("You must provide either 'model' or 'model_name' to load the model.")
        log.info(f'Loading {model_name}')
        model = joblib.load(model_name)
        mlflow.sklearn.log_model(model, artifact_path="model")
    
    log.info('Generating SHAP Values')
    
    # Convert X_test to a pandas DataFrame
    X_test_df = pd.DataFrame(X_test)
    
    # Identify categorical columns
    categorical_features = list(X_test_df.select_dtypes(include=['category', 'object']).columns)
    
    # Create a ColumnTransformer to apply one-hot encoding to categorical columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), categorical_features)
        ],
        remainder='passthrough'  # Keep non-categorical columns as-is
    )
    
    # Fit and transform the test data
    X_test_transformed = preprocessor.fit_transform(X_test_df)
    
    # Generate feature names
    feature_names = (
        list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)) +
        list(X_test_df.select_dtypes(exclude=['category', 'object']).columns)
    )

    # Artifact save directory for plots
    plot_save_dir = os.path.join(artifact_save_dir, 'shap_dependence_plots')
    if not os.path.exists(plot_save_dir):
        os.makedirs(plot_save_dir)

    for index, feature_name in enumerate(features):
        # Find the corresponding integer index for the feature name
        try:
            feature_index = feature_names.index(feature_name)
        except ValueError:
            log.warning(f'Feature {feature_name} not found in feature_names. Skipping.')
            continue
            
        plot_name = os.path.join(plot_save_dir, f'{model_prefix}_feature{index}_SHAP_dependence.png') 
        
        # Clear the previous plot
        plt.clf()
        shap.dependence_plot(feature_index, shap_values, X_test_transformed, 
                             feature_names=feature_names, show=False)  # Pass feature names for labeling
        
        # Save the SHAP dependence plot as an image
        fig = plt.gcf()
        plt.savefig(plot_name, dpi=150, bbox_inches='tight')
        log.info(f'{plot_name} saved.')
        mlflow.log_artifact(plot_name, artifact_path="shap_dependence_plots")
        
    log.info(f'All SHAP dependence plots saved in {plot_save_dir}')

    
def evaluate_classification_models(model_name, predictions, true_labels, target_names, plot_classification_report=False, artifact_save_dir: str = 'artifacts'):
    """
    Evaluate a classification model and generate a classification report.
    
    Parameters:
        model_name (str): Name of the model for plot naming.
        predictions (array-like): Model predictions (predicted class labels).
        true_labels (array-like): True class labels.
        target_names (list): e.g ['Class 0', 'Class 1']
        plot_classification_report (bool): Whether to plot the classification report (default: False).
        artifact_save_dir (str): Directory where artifacts including plots will be saved.

    Returns:
        dict: A dictionary containing the computed metrics.
    """
    evaluation_results = {}
    
    accuracy = accuracy_score(true_labels, predictions)
    mlflow.log_metrics({"accuracy": accuracy})
    evaluation_results['accuracy'] = accuracy
    
    precision = precision_score(true_labels, predictions)
    mlflow.log_metrics({"precision": precision})
    evaluation_results['precision'] = precision

    recall = recall_score(true_labels, predictions)
    mlflow.log_metrics({"recall": recall})
    evaluation_results['recall'] = recall

    f1 = f1_score(true_labels, predictions)
    mlflow.log_metrics({"f1": f1})
    evaluation_results['f1'] = f1

    try:
        roc_auc = roc_auc_score(true_labels, predictions)
        mlflow.log_metrics({"roc_auc": roc_auc})
        evaluation_results['roc_auc'] = roc_auc
    except ValueError:
        # roc_auc_score may not work for multiclass classification
        evaluation_results['roc_auc'] = None

    confusion = confusion_matrix(true_labels, predictions)
    evaluation_results['confusion_matrix'] = confusion

    if plot_classification_report:
        # Generate the classification report
        report = classification_report(
            true_labels, predictions, target_names=target_names, output_dict=True
        )

        # Convert the classification report to a DataFrame for easy plotting
        report_df = pd.DataFrame(report).transpose()

        # Plot the classification report as a heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(report_df.iloc[:-3, :-1], annot=True, cmap='Blues', fmt=".2f", cbar=False)
        plt.title('Classification Report')
        plt.xlabel('Classes')
        plt.ylabel('Metrics')
        
        fig = plt.gcf()
        model_prefix = os.path.splitext(model_name)[0]
        if not os.path.exists(artifact_save_dir):
            os.makedirs(artifact_save_dir)

        #plot_name = os.path.join(plot_save_dir, f'{model_prefix}_SHAP_summary.png')
        plot_name = os.path.join(artifact_save_dir, f'{model_prefix}_Classification_Report.png')
        
        # Artifact save directory for plots
        #if not os.path.exists(artifact_save_dir):
            #os.makedirs(artifact_save_dir)
        
        plt.savefig(plot_name)
        log.info(f'{plot_name} saved.')

        # Log the classification report plot as an artifact in MLflow
        mlflow.log_artifact(plot_name)

        plt.show()

        # Add classification report metrics to the evaluation results
        for metric, values in report.items():
            if isinstance(values, dict):
                for class_name, value in values.items():
                    metric_name = f"{metric}_{class_name}"
                    evaluation_results[metric_name] = value

    # Log the evaluation_results dictionary to MLflow
    mlflow.log_params(evaluation_results)
    log.info(evaluation_results)
    
    return evaluation_results

