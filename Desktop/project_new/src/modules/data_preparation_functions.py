import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
import logging
import os
import mlflow

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


import matplotlib.pyplot as plt
import seaborn as sns

# Directory to save the plots
plot_save_dir = 'plots'

def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def check_missing_values(df, artifact_save_dir='artefacts'):
    """
    Check and visualize missing values in a DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame to check for missing values.
        artifact_save_dir (str): Directory to save the heatmap plot and other artifacts.

    Returns:
        None
    """
    log = logging.getLogger(__name__)

    # Check for missing values and compute the count of missing values in each column
    missing_values = df.isnull().sum()

    # Plot a heatmap of missing values
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.xlabel('Columns')
    plt.ylabel('Rows')

    # List the number of missing values in each column
    log.info("Number of missing values in each column:")
    for column, count in missing_values.items():
        if count > 0:
            log.info(f"Column '{column}' had {count} missing values.")

    # Create the artifact_save_dir directory if it doesn't exist
    if not os.path.exists(artifact_save_dir):
        os.makedirs(artifact_save_dir)

    # Save the heatmap plot as an image in the specified directory
    plot_name = 'missing_values_heatmap'
    plot_save_path = os.path.join(artifact_save_dir, f"{plot_name}.png")
    plt.savefig(plot_save_path)
    log.info(f'{plot_name} saved at: {plot_save_path}')

    # Log the heatmap plot as an artifact using MLflow
    mlflow.log_artifact(plot_save_path, artifact_path=f'{plot_name}.png')

    plt.show()

    

def replace_missing_values(df, ms_threshold: int, artifact_save_dir='artefacts'):
    """
    Replace missing values in a DataFrame using interpolation and iterative imputation.

    Parameters:
        df (pd.DataFrame): The DataFrame containing missing values.
        ms_threshold (int): Threshold to switch between interpolation and iterative imputer.
        artifact_save_dir (str, optional): Directory to save artifacts (e.g., logs) (default: None).

    Returns:
        pd.DataFrame: DataFrame with missing values replaced.
    """
    # Create a logger
    log = logging.getLogger(__name__)

    # If an artifact_save_dir is specified, configure the logger to save logs to that directory
    if artifact_save_dir:
        log_filename = 'replace_missing_values.log'
        log_filepath = os.path.join(artifact_save_dir, log_filename)

        # Configure the logger
        logging.basicConfig(filename=log_filepath, level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Threshold to switch between interpolation and iterative imputer
    interpolation_threshold = ms_threshold

    # Count the missing values in each column
    missing_values = df.isnull().sum()

    # List to store column names that need imputation
    columns_to_impute = []

    # Identify columns where the gap between missing values is less than the threshold
    for column, count in missing_values.items():
        if count > 0:
            indices = df[column].index[df[column].isnull()]
            differences = np.diff(indices)
            if all(diff <= interpolation_threshold for diff in differences):
                columns_to_impute.append(column)

    # Separate columns for interpolation and iterative imputer
    columns_to_interpolate = [col for col in columns_to_impute if col not in columns_to_impute]
    columns_to_iterative_impute = [col for col in columns_to_impute if col in columns_to_impute]

    # Replace missing values with interpolation
    if len(columns_to_interpolate) > 0:
        imputer = SimpleImputer(strategy='nearest')
        df[columns_to_interpolate] = imputer.fit_transform(df[columns_to_interpolate])
        for column in columns_to_interpolate:
            log.info(f"Imputed '{column}' using 'nearest' strategy.")

    # Replace missing values with iterative imputer
    if len(columns_to_iterative_impute) > 0:
        imputer = IterativeImputer()
        df[columns_to_iterative_impute] = imputer.fit_transform(df[columns_to_iterative_impute])
        for column in columns_to_iterative_impute:
            log.info(f"Imputed '{column}' using 'iterative' strategy.")

    return df

    
def drop_highly_correlated_features(df, corr_threshold=0.8, plot_heatmaps=True, artifact_save_dir='artefacts'):
    """
    Perform feature selection based on Spearman correlation coefficient.

    Parameters:
    - df: pandas DataFrame containing the dataset.
    - corr_threshold: The threshold for correlation above which features will be dropped (default is 0.8).
    - plot_heatmaps: Whether to plot heatmaps before and after dropping (default is True).
    - artifact_save_dir: Directory to save the correlation heatmap plots (default is None).

    Returns:
    - A DataFrame with the highly correlated features dropped.
    """
    # Create a logger
    log = logging.getLogger(__name__)

    if artifact_save_dir and not os.path.exists(artifact_save_dir):
        os.makedirs(artifact_save_dir)
    
    # Calculate the correlation matrix (Spearman by default in pandas)
    corr_matrix = df.corr(method='spearman')
    
    if plot_heatmaps:
        # Plot the correlation heatmap before dropping
        fig_before = plt.figure(figsize=(8, 6))
        plt.title("Correlation Heatmap (Before Dropping)")
        sns_plot_before = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        
        # Save the plot as an image file
        if artifact_save_dir:
            before_plot_path = os.path.join(artifact_save_dir, "correlation_heatmap_before.png")
            plt.savefig(before_plot_path)
            log.info("Correlation heatmap (Before Dropping): %s", before_plot_path)
            
        mlflow.log_artifact(before_plot_path, artifact_path="correlation_heatmap_before.png")
        plt.show()
    
    # Create a set to store the columns to drop
    columns_to_drop = set()
    
    # Create a list to store the names of the dropped columns
    dropped_columns = []
    
    # Iterate through the columns and identify highly correlated features
    for col1 in corr_matrix.columns:
        for col2 in corr_matrix.columns:
            if col1 != col2 and abs(corr_matrix.loc[col1, col2]) >= corr_threshold:
                # Check if col1 or col2 should be dropped based on their mean correlation
                mean_corr_col1 = corr_matrix.loc[col1, :].drop(col1).abs().mean()
                mean_corr_col2 = corr_matrix.loc[col2, :].drop(col2).abs().mean()
                
                if mean_corr_col1 > mean_corr_col2:
                    columns_to_drop.add(col1)
                    dropped_columns.append(col1)
                else:
                    columns_to_drop.add(col2)
                    dropped_columns.append(col2)
    
    # Drop the highly correlated features from the DataFrame
    df = df.drop(columns=columns_to_drop)
    
    if plot_heatmaps:
        # Calculate the correlation matrix after dropping
        corr_matrix_after_drop = df.corr(method='spearman')
        
        # Plot the correlation heatmap after dropping
        fig_after = plt.figure(figsize=(8, 6))
        plt.title("Correlation Heatmap (After Dropping)")
        sns_plot_after = sns.heatmap(corr_matrix_after_drop, annot=True, cmap='coolwarm', fmt=".2f")
        
        # Save the plot as an image file
        if artifact_save_dir:
            after_plot_path = os.path.join(artifact_save_dir, "correlation_heatmap_after.png")
            plt.savefig(after_plot_path)
            log.info("Correlation heatmap (After Dropping): %s", after_plot_path)
            
        mlflow.log_artifact(after_plot_path, artifact_path="correlation_heatmap_after.png")
        plt.show()
           
    # Log the names of the dropped columns
    log.info("Dropped columns: %s", dropped_columns)

    return df

def drop_high_cardinality_features(df, max_unique_threshold=0.9):
    """
    Drop high cardinality features (columns) from a DataFrame based on a threshold.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        max_unique_threshold (float): The maximum allowed fraction of unique values in a column (default is 0.9).

    Returns:
        pd.DataFrame: The DataFrame with high cardinality columns dropped.
    """
    if df is None:
        raise ValueError("Input DataFrame 'df' cannot be None.")
        
    # Calculate the maximum number of allowed unique values for each column
    max_unique_values = len(df) * max_unique_threshold
    
    # Identify and drop columns with unique values exceeding the threshold
    high_cardinality_columns = [col for col in df.columns if df[col].nunique() > max_unique_values]
    
    # Log the names of the dropped columns using MLflow
    if high_cardinality_columns:
        mlflow.log_param("HighCardinalityColumns", ', '.join(high_cardinality_columns))
    
    df_dropped = df.drop(columns=high_cardinality_columns)
    
    return df_dropped

def select_categorical_columns(data):
    """
    Select categorical columns from a DataFrame.

    Parameters:
    - data: pandas DataFrame containing the dataset.

    Returns:
    - A list of column names that are categorical.
    """
    categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
    return categorical_columns




if __name__ == "__main__":
    main()