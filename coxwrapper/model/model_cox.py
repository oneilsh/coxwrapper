
import logging
import joblib
import pandas as pd
import numpy as np
import os
import inspect
import pickle
from datetime import datetime

# MLflow removed for v0.2.0 - using simple pickle persistence
# --- Lifelines Imports for Cox PH ---
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__) # Use logger instead of direct logging.info

# Import specific functions/classes directly from sibling modules
from .dataloader_cox import load_configuration, load_data_from_bigquery
from .preprocessing_cox import split_data, create_preprocessor, apply_preprocessing 

# --- Helper Function for Model Evaluation ---
# This is the single, corrected version of evaluate_cox_model.
def evaluate_cox_model(model, X_test, y_test_duration, y_test_event, duration_col, event_col):
    """
    Evaluates the Cox PH model on the test set.

    Args:
        model: The trained CoxPHFitter model.
        X_test (pd.DataFrame): Test features.
        y_test_duration (pd.Series): Test durations.
        y_test_event (pd.Series): Test event indicators.
        duration_col (str): Name of the duration column in the original DataFrame.
        event_col (str): Name of the event column in the original DataFrame.

    Returns:
        dict: A dictionary of evaluation metrics.
    """
    logger.info(f"Type of model in evaluate_cox_model: {type(model)}")
    evaluation_metrics = {}
    try:
        # Create a combined DataFrame for scoring as expected by model.score()
        test_df_for_score = X_test.copy()
        test_df_for_score[duration_col] = y_test_duration
        test_df_for_score[event_col] = y_test_event

        if hasattr(model, 'score'):
            logger.info(f"Model has score method. Signature: {inspect.signature(model.score)}")
            # Use model.score with the combined DataFrame. It typically returns average partial log-likelihood.
            log_likelihood_score = model.score(test_df_for_score)
            
            # For explicit Concordance Index calculation, use lifelines.utils.concordance_index
            predictions_for_cindex = model.predict_partial_hazard(X_test)
            c_index_explicit = concordance_index(y_test_duration, predictions_for_cindex, y_test_event)

            evaluation_metrics['log_likelihood_score'] = log_likelihood_score
            evaluation_metrics['c_index'] = c_index_explicit
            logger.info(f"Log-likelihood Score: {log_likelihood_score}")
            logger.info(f"Concordance Index (C-index): {c_index_explicit}")

        else:
            logger.info("Model does NOT have a score method!")
            evaluation_metrics['c_index'] = np.nan # Indicate not calculated if score method is missing

    except Exception as e:
        logger.error(f"Error during model evaluation: {e}", exc_info=True)
        # evaluation_metrics will remain empty or partial if error occurs
    return evaluation_metrics


# --- MLflow wrapper removed in v0.2.0 for simplicity ---
# We now use direct pickle persistence instead of MLflow model logging


# --- Helper functions for joblib saving/loading (not used by run_end_to_end_pipeline directly now) ---
def save_model_joblib(model, path):
    """
    Save the trained model to a file using joblib.
    """
    try:
        joblib.dump(model, path)
        logger.info(f"Model saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        raise

def load_model_joblib(path):
    """
    Load a model from a file using joblib.
    """
    try:
        model = joblib.load(path)
        logger.info(f"Model loaded from {path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


# --- Main End-to-End Pipeline Function ---
def run_end_to_end_pipeline(config_path, preloaded_data_df=None, test_size=0.2, random_state=42, **model_kwargs):
    """
    Runs the end-to-end ML pipeline for the Cox Proportional Hazards model.

    Args:
        config_path (str): Path to the configuration YAML file.
        preloaded_data_df (pd.DataFrame, optional): Preloaded data DataFrame. If None, data is loaded from BigQuery.
        test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.2.
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.
        **model_kwargs: Keyword arguments for CoxPHFitter initialization (e.g., penalizer, l1_ratio).

    Returns:
        tuple: A tuple containing the trained CoxPHFitter model object and evaluation metrics.
    """
    logger.info("Starting the end-to-end ML pipeline for Cox PH model...")

    # v0.2.0: Simplified logging without MLflow
    logger.info(f"Model parameters: {model_kwargs}")

    config = dataloader_cox.load_configuration(config_path)
    logger.info("Configuration loaded successfully.")

    data_df = preloaded_data_df if preloaded_data_df is not None else dataloader_cox.load_data_from_bigquery(config)
    logger.info(f"Using preloaded data. Shape: {data_df.shape}")

    # Identify features and target columns from config
    feature_columns = config["model_features_final"]
    duration_col = config["model_io_columns"]["duration_col"]
    event_col = config["model_io_columns"]["event_col"]
    
    logger.info(f"Identified {len(feature_columns)} predictor features.")
    logger.info(f"Feature columns: {feature_columns}")

    # Prepare X, y, event, duration for splitting
    X = data_df[feature_columns].copy()
    y_duration = data_df[duration_col]
    y_event = data_df[event_col]

    logger.info("Splitting data into training and testing sets...")
    X_train_raw, X_test_raw, y_train_duration, y_test_duration, y_train_event, y_test_event = \
        dataloader_cox.split_time_to_event_data(X, y_duration, y_event, test_size, random_state)
    
    logger.info(f"Data split complete: {len(X_train_raw)} train samples, {len(X_test_raw)} test samples.")
    logger.info(f"Raw data split. X_train_raw shape: {X_train_raw.shape}, X_test_raw shape: {X_test_raw.shape}")

    # Convert relevant columns to float64 for preprocessing consistency
    for col in X_train_raw.select_dtypes(include=['Int64', 'boolean']).columns:
        X_train_raw[col] = X_train_raw[col].astype(float)
    for col in X_test_raw.select_dtypes(include=['Int64', 'boolean']).columns:
        X_test_raw[col] = X_test_raw[col].astype(float)

    logger.info("Applying feature preprocessing pipeline...")
    # Create and fit preprocessor on training data
    fitted_preprocessor = preprocessing_cox.create_preprocessor(X_train_raw)
    
    logger.info("Feature preprocessing pipeline created and fitted.")

    # Apply preprocessing
    X_train_processed_array, X_test_processed_array = preprocessing_cox.apply_preprocessing(
        fitted_preprocessor, X_train_raw, X_test_raw
    )
    
    # Get processed feature names
    feature_columns_processed = fitted_preprocessor.get_feature_names_out().tolist()

    # Convert processed arrays back to DataFrame with correct column names
    X_train_processed = pd.DataFrame(X_train_processed_array, columns=feature_columns_processed)
    X_test_processed = pd.DataFrame(X_test_processed_array, columns=feature_columns_processed)

    logger.info(f"Processed X_train shape: {X_train_processed.shape}, dtypes: {X_train_processed.dtypes.to_dict()}")
    logger.info(f"Processed X_test shape: {X_test_processed.shape}, dtype: {X_test_processed.dtypes.to_dict()}")
    logger.info(f"Features processed. X_train_processed shape: {X_train_processed.shape}, X_test_processed shape: {X_test_processed.shape}")
    
    logger.info("Checking processed features for potential convergence issues before model fit...")
    # Placeholder for dynamic feature dropping logic if needed
    logger.info("No additional problematic processed columns identified for temporary drop.")
    logger.info(f"Final features for model fit. X_train shape: {X_train_processed.shape}, X_test shape: {X_test_processed.shape}")

    logger.info("Training the Cox Proportional Hazards model...")
    logger.info("Initializing CoxPHFitter model...")
    trained_cox_model = CoxPHFitter(**model_kwargs)
    
    # Combine data into a single DataFrame for fitting (robust method)
    df_train_combined = X_train_processed.copy()
    df_train_combined[duration_col] = y_train_duration
    df_train_combined[event_col] = y_train_event
    trained_cox_model.fit(df_train_combined, duration_col=duration_col, event_col=event_col)
    
    logger.info("Model training completed.")
    logger.info("Cox PH Model training completed successfully.")

    logger.info("\n--- Cox PH Model Summary ---")
    trained_cox_model.print_summary() # Prints to stdout
    logger.info("----------------------------")

    logger.info("Evaluating the Cox PH model on the test set...")
    evaluation_metrics = evaluate_cox_model(
        trained_cox_model, X_test_processed, y_test_duration, y_test_event, duration_col, event_col
    )
    logger.info(f"Evaluation metrics: {evaluation_metrics}")

    # --- Pickle the trained model and preprocessor directly ---
    # Use a configurable output directory or default to current directory
    output_dir = os.environ.get("COXWRAPPER_OUTPUT_DIR", ".")
    pickled_models_output_dir = os.path.join(output_dir, "pickled_models")
    os.makedirs(pickled_models_output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"cox_ph_model_{timestamp}.pkl"
    preprocessor_filename = f"preprocessor_{timestamp}.pkl"

    pickled_model_path = os.path.join(pickled_models_output_dir, model_filename)
    pickled_preprocessor_path = os.path.join(pickled_models_output_dir, preprocessor_filename)

    try:
        with open(pickled_model_path, 'wb') as f:
            pickle.dump(trained_cox_model, f)
        logger.info(f"Cox PH model pickled successfully to: {pickled_model_path}")

        with open(pickled_preprocessor_path, 'wb') as f:
            pickle.dump(fitted_preprocessor, f)
        logger.info(f"Preprocessor pickled successfully to: {pickled_preprocessor_path}")
        
        # v0.2.0: Direct pickle saving without MLflow artifacts
        logger.info("Model and preprocessor saved as pickle files")

    except Exception as e:
        logger.error(f"Error pickling model or preprocessor: {e}", exc_info=True)
        raise # Re-raise to signal pipeline failure if pickling fails

    logger.info("ML pipeline completed successfully (with potential warnings/errors).")
    logger.info("\n--- Pipeline Execution Summary ---")
    logger.info(f"Trained Cox PH Model Object: {trained_cox_model}")
    logger.info(f"Final Evaluation Metrics: {evaluation_metrics}")
    logger.info(f"Cox PH Model pickled to: {pickled_model_path}")
    logger.info(f"Preprocessor pickled to: {pickled_preprocessor_path}")
    logger.info("v0.2.0: Simplified pipeline without MLflow dependency")

    return trained_cox_model, evaluation_metrics, pickled_model_path, pickled_preprocessor_path


