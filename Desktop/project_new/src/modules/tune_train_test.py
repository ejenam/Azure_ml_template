from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import logging
import os
import json
import joblib
import mlflow

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)



def custom_train_test_split(data, target_column, test_size=0.2, random_state=101, time_series=False):
    """
    Split the dataset into training and testing sets.

    Parameters:
    - data: pandas DataFrame containing the dataset.
    - target_column: Name of the target column.
    - test_size: Proportion of the dataset to include in the test split (default is 0.2).
    - random_state: Seed for random number generation (optional).
    - time_series: Set to True if the data is time series data (default is False).

    Returns:
    - X_train, X_test, y_train, y_test: The split datasets.
    """
    if time_series:
        # For time series data, split by a specific time point
        data = data.sort_index()  # Sort by time index if not already sorted
        n = len(data)
        split_index = int((1 - test_size) * n)
        X_train, X_test = data.iloc[:split_index, :-1], data.iloc[split_index:, :-1]
        y_train, y_test = data.iloc[:split_index][target_column], data.iloc[split_index:][target_column]
    else:
        # For regular (cross-sectional) data, use train_test_split
        X = data.drop(columns=[target_column])
        y = data[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test



def hyperparameter_tuning(X_train, y_train, model_prefix:str, param_grid=None, random_search=False, bayesian_search=False, n_iter=10, random_seed=101):
    """
    Train a Histogram Gradient Boosting Classifier and tune its hyperparameters.

    Parameters:
    - X_train, y_train: Training data and labels.
    - X_test, y_test: Testing data and labels.
    - model_prefix: Prefix for model artifacts.
    - param_grid: Hyperparameter grid to search (default is None).
    - random_search: Whether to use random search instead of grid search (default is False).
    - bayesian_search: Whether to use Bayesian hyperparameter search (default is False).
    - n_iter: Number of parameter settings that are sampled (only for random_search or bayesian_search).

    Returns:
    - Trained model, best hyperparameters, and test accuracy.
    """
    # Identify categorical columns
    categorical_features = list(X_train.select_dtypes(include=['category', 'object']).columns)
    # Create a ColumnTransformer to apply one-hot encoding to categorical columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), categorical_features)
        ],
        remainder='passthrough'  # Keep non-categorical columns as-is
    )
    # Create a Histogram Gradient Boosting Classifier
    clf = HistGradientBoostingClassifier(random_state=42)
    
    # Combine preprocessing and classifier into a single pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])

    if not bayesian_search:
        # Define hyperparameters for grid search or random search
        hyperparameters = {
            'classifier__max_iter': [100, 200, 300],  # Adjust the values as needed
            'classifier__learning_rate': [0.001, 0.01, 0.1],  # Adjust the values as needed
            'classifier__max_depth': [3, 4, 5],  # Adjust the values as needed
            'classifier__l2_regularization': [0.0, 0.1, 0.2]  # Adjust the values as needed
        }

        if random_search:
            # Use RandomizedSearchCV
            search = RandomizedSearchCV(pipeline, param_distributions=hyperparameters, n_iter=n_iter, scoring='accuracy', n_jobs=-1, random_state=random_seed)
        else:
            # Use GridSearchCV
            search = GridSearchCV(pipeline, param_grid=hyperparameters, scoring='accuracy', n_jobs=-1, random_state=random_seed)
    else:
        # Use Bayesian hyperparameter search with BayesSearchCV
        param_grid = {
            'classifier__max_iter': (100, 300),
            'classifier__learning_rate': (0.001, 0.1),
            'classifier__max_depth': (3, 5),
            'classifier__l2_regularization': (0.0, 0.2)
        }

        search = BayesSearchCV(pipeline, param_grid, n_iter=n_iter, cv=TimeSeriesSplit(n_splits=3), scoring='accuracy', n_jobs=-1, random_state=random_seed)

    # Fit the search to the training data
    search.fit(X_train, y_train)

    # Get the best hyperparameters and the best estimator (trained model)
    best_params = search.best_params_
    best_estimator = search.best_estimator_
    
    log.info('Parameters chosen are:')
    log.info(best_params)
    
    log.info('The best estimator is:')
    log.info(best_estimator)
    
    # Evaluate the best model on the test data
   # y_pred = best_estimator.predict(X_test)
   # test_accuracy = accuracy_score(y_test, y_pred)
   # log.info(f'Test Accuracy: {test_accuracy:.2f}')
    
    # Save the best model to a file
    model_filename = f'{model_prefix}_best_model.joblib'
    joblib.dump(best_estimator, model_filename)
    
    # Save best hyperparameters to a JSON file
    hyperparameters_filename = f'{model_prefix}_hyperparameters.json'
    log.info(f'Saving best hyperparameters for {model_prefix} as {hyperparameters_filename}')
    with open(hyperparameters_filename, 'w') as f:
        json.dump(best_params, f)
    mlflow.log_params(best_params)  # Log best hyperparameters to MLflow

    # Save the best model to a file
    model_filename = f'{model_prefix}_best_model.joblib'
    joblib.dump(best_estimator, model_filename)
        
    # Log the model artifact to MLflow
    mlflow.sklearn.log_model(best_estimator, artifact_path=model_prefix)
        
    return best_params, hyperparameters_filename



def train_model(X_train, y_train, model_name:str, hyperparam: dict=None, hyperparam_filename: str=None):
    # Identify categorical columns
    categorical_features = list(X_train.select_dtypes(include=['category', 'object']).columns)
    # Create a ColumnTransformer to apply one-hot encoding to categorical columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), categorical_features)
        ],
        remainder='passthrough'  # Keep non-categorical columns as-is
    )
    X_train_transformed = preprocessor.fit_transform(X_train)
    if hyperparam_filename is not None:
        log.info(f'Loading in hyperparameters: {hyperparam_filename}')
        with open(hyperparam_filename, 'r') as f:
            best_params = json.load(f)
    elif hyperparam is not None:
        best_params = hyperparam
    else:
        raise ValueError('Either hyperparam or hyperparam_filename must be assigned')
    
    # Create and train the model with the specified hyperparameters
    log.info('Training Model')
    trained_model = HistGradientBoostingClassifier(class_weight='balanced',
        max_iter=best_params['classifier__max_iter'],
        learning_rate=best_params['classifier__learning_rate'],
        max_depth=best_params['classifier__max_depth'],
        l2_regularization=best_params['classifier__l2_regularization'],
        random_state=10
    )
    trained_model.fit(X_train_transformed, y_train)
    
    # Save the trained model to a file
    log.info(f'Saving {model_name}')
    joblib.dump(trained_model, model_name)

    # Stop Logging
    mlflow.end_run()
    
    return trained_model

def predict_model(trained_model, X_test, inference_col_name):
    """
    Predict using a trained machine learning model.

    Parameters:
    - trained_model: The trained machine learning model.
    - X_test: The test dataset on which to make predictions.
    - inference_col_name: The name of the column to store predictions in the inference DataFrame.

    Returns:
    - inference_df: The DataFrame containing predictions.
    - inference_col_name: The name of the column where predictions are stored.
    - predictions: The predictions made by the model.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    import pandas as pd
    
    # Identify categorical columns
    categorical_features = list(X_test.select_dtypes(include=['category', 'object']).columns)
    
    # Create a ColumnTransformer to apply one-hot encoding to categorical columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), categorical_features)
        ],
        remainder='passthrough'  # Keep non-categorical columns as-is
    )
    
    # Fit and transform the test data
    X_test_transformed = preprocessor.fit_transform(X_test)
    
    # Get the one-hot encoded feature names
    ohe = preprocessor.named_transformers_['cat']
    cat_feature_names = list(ohe.get_feature_names_out(input_features=categorical_features))
    
    # Combine the one-hot encoded feature names and non-categorical column names
    all_column_names = cat_feature_names + list(X_test.select_dtypes(exclude=['category', 'object']).columns)
    
    # Convert X_test_transformed to a DataFrame with appropriate column names
    inference_df = pd.DataFrame(X_test_transformed, columns=all_column_names)
    
    # Make predictions using the trained model
    predictions = trained_model.predict(X_test_transformed)
    
    # Add predictions to the DataFrame with the specified column name
    inference_df[inference_col_name] = predictions
    
    return inference_df, inference_col_name, predictions


if __name__ == "__main__":
    main()
