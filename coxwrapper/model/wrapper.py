"""
CoxModelWrapper - Simplified API for Cox regression modeling

This wrapper provides a clean, simplified interface around the existing
twinsight_model functionality while maintaining backward compatibility.
"""

import os
import pickle
import yaml
import requests
from typing import Dict, List, Union, Any, Optional
import pandas as pd
import numpy as np
import logging

# Import existing functionality
from .dataloader_cox import load_configuration, load_data_from_bigquery
from .preprocessing_cox import create_preprocessor, apply_preprocessing
from .model_cox import CoxPHFitter, evaluate_cox_model, run_end_to_end_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CoxModelWrapper:
    """
    Simplified wrapper for Cox proportional hazards modeling on All of Us data.
    
    This class provides a clean API that wraps the existing twinsight_model
    functionality, making it easier to use while maintaining full compatibility.
    """
    
    def __init__(self, config: Union[str, Dict]):
        """
        Initialize the Cox model wrapper.
        
        Args:
            config: Either:
                - Path to a local YAML config file (str)  
                - URL to a YAML config file (str starting with http:// or https://)
                - Config dictionary (dict)
        """
        logger.info("Initializing CoxModelWrapper v0.2.0")
        
        
        # Load configuration
        if isinstance(config, str):
            if config.startswith(('http://', 'https://')):
                # Load from URL
                self.config = self._load_config_from_url(config)
                self.config_source = f"URL: {config}"
            else:
                # Load from local file
                if not os.path.exists(config):
                    raise FileNotFoundError(f"Configuration file not found: {config}")
                self.config = load_configuration(config)
                self.config_source = f"file: {config}"
        elif isinstance(config, dict):
            self.config = config.copy()
            self.config_source = "dictionary"
        else:
            raise TypeError("config must be a file path (str), URL (str), or dictionary")
        
        # Initialize state
        self.data = None
        self.model = None
        self.preprocessor = None
        self.training_stats = None
        self.data_summary = None
        
        # Extract key config info
        self.outcome_name = self.config.get('outcome', {}).get('name', 'unknown')
        self.raw_feature_names = self.config.get('model_features_final', [])
        self.feature_names = []  # Will be set after preprocessing
        self.duration_col = self.config.get('model_io_columns', {}).get('duration_col', 'time_to_event_days')
        self.event_col = self.config.get('model_io_columns', {}).get('event_col', 'event_observed')
        
        logger.info(f"Configuration loaded from {self.config_source}")
        logger.info(f"Outcome: {self.outcome_name}")
        logger.info(f"Features: {len(self.feature_names)} features configured")
    
    def _load_config_from_url(self, url: str) -> Dict:
        """
        Load configuration from a URL.
        
        Args:
            url: URL to the YAML configuration file
            
        Returns:
            Dict containing the configuration
        """
        try:
            logger.info(f"Downloading configuration from: {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()  # Raises an HTTPError for bad responses
            
            # Parse YAML content
            config = yaml.safe_load(response.text)
            logger.info("Configuration downloaded and parsed successfully")
            return config
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download configuration from URL: {e}")
            raise RuntimeError(f"Failed to download configuration from URL: {e}")
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML from URL: {e}")
            raise RuntimeError(f"Failed to parse YAML from URL: {e}")
    
    def load_data(self, use_mock: bool = False, n_patients: int = 1000):
        """
        Load and prepare data according to the configuration.
        
        Args:
            use_mock: If True, generate synthetic data instead of loading from BigQuery
            n_patients: Number of patients to generate for mock data
        
        This wraps the existing dataloader_cox.load_data_from_bigquery functionality.
        """
        if use_mock:
            logger.info(f"Generating mock data with {n_patients} patients...")
            self.data = self._generate_mock_data(n_patients)
            logger.info(f"Mock data generated successfully: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
        else:
            logger.info("Loading data from BigQuery...")
            try:
                self.data = load_data_from_bigquery(self.config)
                logger.info(f"Data loaded successfully: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
            except Exception as e:
                logger.error(f"Failed to load data: {e}")
                raise
        
        # Generate basic data summary (for later use)
        self._generate_data_summary()
        
        return True
    
    def _generate_mock_data(self, n_patients: int) -> pd.DataFrame:
        """
        Generate synthetic data matching the preprocessing pipeline expectations.
        
        Args:
            n_patients: Number of patients to generate
            
        Returns:
            DataFrame with synthetic patient data
        """
        np.random.seed(42)  # For reproducible results
        
        data = {
            'person_id': range(n_patients),
        }
        
        # Generate demographics based on realistic distributions
        data['ethnicity'] = np.random.choice(
            ['Not Hispanic or Latino', 'Hispanic or Latino', 'Unknown'],
            n_patients, p=[0.7, 0.2, 0.1]
        )
        
        data['sex_at_birth'] = np.random.choice(
            ['Male', 'Female'], n_patients, p=[0.48, 0.52]
        )
        
        # Generate age (realistic for COPD study)
        data['age_at_time_0'] = np.random.normal(65, 12, n_patients).clip(40, 90).astype(float)
        
        # Generate BMI
        data['bmi'] = np.random.normal(28, 5, n_patients).clip(15, 50).astype(float)
        
        # Generate binary comorbidities with realistic prevalence
        data['obesity'] = np.random.choice([0, 1], n_patients, p=[0.7, 0.3]).astype(float)
        data['diabetes'] = np.random.choice([0, 1], n_patients, p=[0.8, 0.2]).astype(float)
        data['cardiovascular_disease'] = np.random.choice([0, 1], n_patients, p=[0.75, 0.25]).astype(float)
        
        # Generate smoking status (categorical)
        data['smoking_status'] = np.random.choice(
            ['Never', 'Former', 'Current'], n_patients, p=[0.4, 0.4, 0.2]
        )
        
        # Generate alcohol use
        data['alcohol_use'] = np.random.choice([0, 1], n_patients, p=[0.6, 0.4]).astype(float)
        
        # Generate survival data based on covariates (simplified Cox model)
        # Higher risk for: older age, smoking, diabetes, CVD
        risk_score = (
            (data['age_at_time_0'] - 65) * 0.03 +  # Age effect
            (data['smoking_status'] == 'Current').astype(float) * 0.8 +  # Current smoking
            (data['smoking_status'] == 'Former').astype(float) * 0.4 +   # Former smoking  
            data['diabetes'] * 0.5 +  # Diabetes
            data['cardiovascular_disease'] * 0.6 +  # CVD
            data['obesity'] * 0.3  # Obesity
        )
        
        # Generate time to event (exponential with varying hazard)
        baseline_hazard = 0.002  # Daily hazard rate
        hazard_ratios = np.exp(risk_score)
        
        # Generate survival times (in days)
        survival_times = np.random.exponential(1 / (baseline_hazard * hazard_ratios))
        
        # Cap at 5 years (1825 days) for study period
        study_end = 1825
        data[self.duration_col] = np.minimum(survival_times, study_end).astype(float)
        
        # Event observed (1 if event occurred before study end, 0 if censored)
        data[self.event_col] = (survival_times < study_end).astype(float)
        
        df = pd.DataFrame(data)
        
        # Convert categorical columns
        df['ethnicity'] = df['ethnicity'].astype('category')
        df['sex_at_birth'] = df['sex_at_birth'].astype('category')
        df['smoking_status'] = df['smoking_status'].astype('category')
        
        logger.info(f"Mock data generated: {df[self.event_col].sum():.0f} events out of {n_patients} patients ({df[self.event_col].mean():.1%} event rate)")
        
        return df
    
    def _mask_small_counts(self, count: int, threshold: int = 20) -> Union[int, str]:
        """
        Apply privacy masking to small counts per AoU requirements.
        
        Args:
            count: The count to potentially mask
            threshold: Counts between 1 and threshold (inclusive) are masked
            
        Returns:
            Original count if >= threshold or == 0, otherwise "<threshold"
        """
        if 1 <= count <= threshold:
            return f"<{threshold + 1}"  # Per AoU: mask 1-20 as "<21"
        return count
    
    def _generate_data_summary(self):
        """
        Generate a summary of the loaded data (internal method).
        """
        if self.data is None:
            logger.warning("No data loaded yet")
            return
        
        summary = {
            'total_patients': len(self.data),
            'total_features': len(self.feature_names),
            'outcome_name': self.outcome_name,
            'data_shape': self.data.shape,
            'config_source': self.config_source
        }
        
        # Add basic outcome statistics if available (with privacy masking)
        if self.event_col in self.data.columns:
            events_observed = int(self.data[self.event_col].sum())
            events_censored = int(len(self.data) - self.data[self.event_col].sum())
            
            # Apply privacy masking for small counts
            summary['events_observed'] = self._mask_small_counts(events_observed)
            summary['events_censored'] = self._mask_small_counts(events_censored)
        
        # Add demographic breakdowns (with privacy masking)
        if 'sex_at_birth' in self.data.columns:
            sex_counts = self.data['sex_at_birth'].value_counts().to_dict()
            summary['demographics'] = {}
            for sex, count in sex_counts.items():
                summary['demographics'][f'sex_{sex}'] = self._mask_small_counts(count)
        
        self.data_summary = summary
        events_str = summary.get('events_observed', 'unknown')
        logger.info(f"Data summary generated: {summary['total_patients']} patients, {events_str} events")
    
    def train_data_summary(self) -> Dict:
        """
        Return a summary of the training data.
        
        Returns:
            Dict containing data summary information
        """
        if self.data_summary is None:
            if self.data is not None:
                self._generate_data_summary()
            else:
                raise RuntimeError("No data loaded. Call load_data() first.")
        
        return self.data_summary.copy()
    
    def train(self, split: float = 0.8, random_state: int = 42, **model_kwargs):
        """
        Train the Cox proportional hazards model.
        
        Args:
            split: Train/test split ratio (default 0.8)
            random_state: Random seed for reproducibility
            **model_kwargs: Additional arguments for CoxPHFitter (e.g., penalizer, l1_ratio)
        
        Returns:
            bool: True if training successful
        """
        if self.data is None:
            raise RuntimeError("No data loaded. Call load_data() first.")
        
        logger.info(f"Training Cox PH model with {split:.1%} train split...")
        logger.info(f"Model parameters: {model_kwargs}")
        
        try:
            # Prepare data for training - use raw feature names initially
            X = self.data[self.raw_feature_names].copy()
            y_duration = self.data[self.duration_col]
            y_event = self.data[self.event_col]
            
            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train_duration, y_test_duration, y_train_event, y_test_event = \
                self._split_survival_data(X, y_duration, y_event, split, random_state)
            
            # Preprocess features
            X_train_processed, X_test_processed, self.preprocessor = \
                self._preprocess_features(X_train, X_test)
            
            # Set the actual feature names after preprocessing
            self.feature_names = list(X_train_processed.columns)
            logger.info(f"Preprocessed feature names: {self.feature_names}")
            
            # Train Cox model
            self.model = self._train_cox_model(
                X_train_processed, y_train_duration, y_train_event, **model_kwargs
            )
            
            # Evaluate model
            self.training_stats = self._evaluate_model(
                X_test_processed, y_test_duration, y_test_event
            )
            
            logger.info("Model training completed successfully")
            logger.info(f"Training stats: {self.training_stats}")
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def _split_survival_data(self, X, y_duration, y_event, split, random_state):
        """Split survival data maintaining time-to-event structure."""
        from sklearn.model_selection import train_test_split
        
        # Combine duration and event for splitting
        y_combined = np.column_stack([y_duration, y_event])
        
        # Split maintaining the relationship between duration and event
        X_train, X_test, y_combined_train, y_combined_test = train_test_split(
            X, y_combined, test_size=(1.0 - split), random_state=random_state
        )
        
        # Unpack the combined arrays
        y_train_duration = pd.Series(y_combined_train[:, 0], index=X_train.index, name=y_duration.name)
        y_train_event = pd.Series(y_combined_train[:, 1], index=X_train.index, name=y_event.name)
        y_test_duration = pd.Series(y_combined_test[:, 0], index=X_test.index, name=y_duration.name)
        y_test_event = pd.Series(y_combined_test[:, 1], index=X_test.index, name=y_event.name)
        
        return X_train, X_test, y_train_duration, y_test_duration, y_train_event, y_test_event
    
    def _preprocess_features(self, X_train, X_test):
        """Preprocess features using the existing preprocessing pipeline."""
        from .preprocessing_cox import create_preprocessor, apply_preprocessing
        
        # Create and fit preprocessor
        preprocessor = create_preprocessor(X_train)
        
        # Apply preprocessing
        X_train_processed_array, X_test_processed_array = apply_preprocessing(
            preprocessor, X_train, X_test
        )
        
        # Convert back to DataFrames with feature names
        feature_names = preprocessor.get_feature_names_out()
        X_train_processed = pd.DataFrame(X_train_processed_array, columns=feature_names, index=X_train.index)
        X_test_processed = pd.DataFrame(X_test_processed_array, columns=feature_names, index=X_test.index)
        
        return X_train_processed, X_test_processed, preprocessor
    
    def _train_cox_model(self, X_train, y_duration, y_event, **model_kwargs):
        """Train the Cox proportional hazards model."""
        from lifelines import CoxPHFitter
        
        # Combine data for training
        df_train = X_train.copy()
        df_train[self.duration_col] = y_duration
        df_train[self.event_col] = y_event
        
        # Initialize and train model
        model = CoxPHFitter(**model_kwargs)
        model.fit(df_train, duration_col=self.duration_col, event_col=self.event_col)
        
        return model
    
    def _evaluate_model(self, X_test, y_duration, y_event):
        """Evaluate the trained model."""
        from .model_cox import evaluate_cox_model
        
        # Get evaluation metrics
        metrics = evaluate_cox_model(
            self.model, X_test, y_duration, y_event, 
            self.duration_col, self.event_col
        )
        
        # Add basic model info
        metrics['model_type'] = 'Cox Proportional Hazards'
        metrics['n_features'] = len(self.feature_names)
        metrics['n_train_samples'] = len(self.data) - len(X_test)
        metrics['n_test_samples'] = len(X_test)
        
        return metrics
    
    def get_prediction(self, pt_data: List[Dict]) -> List[Dict]:
        """
        Get predictions for patient data.
        
        Args:
            pt_data: List of patient data dictionaries
            
        Returns:
            List of prediction dictionaries with survival probabilities and risk scores
        """
        if self.model is None:
            raise RuntimeError("No trained model. Call train() first.")
        
        if self.preprocessor is None:
            raise RuntimeError("No preprocessor available. Call train() first.")
        
        logger.info(f"Getting predictions for {len(pt_data)} patients...")
        logger.info(f"Expected raw features: {self.raw_feature_names}")
        logger.info(f"Provided features: {list(pt_data[0].keys())}")
        
        try:
            # Convert patient data to DataFrame
            df_patients = pd.DataFrame(pt_data)
            
            # Validate against raw feature names (what users should provide)
            missing_features = set(self.raw_feature_names) - set(df_patients.columns)
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
            
            # Select only required raw features in correct order
            X = df_patients[self.raw_feature_names].copy()
            
            # Preprocess features using fitted preprocessor
            X_processed_array = self.preprocessor.transform(X)
            X_processed = pd.DataFrame(
                X_processed_array, 
                columns=self.preprocessor.get_feature_names_out(),
                index=X.index
            )
            
            # Get baseline survival function
            baseline_survival = self.model.baseline_survival_
            
            # Calculate predictions for each patient
            predictions = []
            for i, (idx, patient_features) in enumerate(X_processed.iterrows()):
                try:
                    # Get patient's risk score (log hazard ratio)
                    risk_score = self.model.predict_partial_hazard(patient_features)
                    
                    # Calculate survival probabilities at different time points
                    # Use median follow-up time from training data as reference
                    median_time = self.data[self.duration_col].median()
                    
                    # Get survival probability at median time
                    survival_prob = self.model.predict_survival_function(
                        patient_features, times=[median_time]
                    ).iloc[0, 0]
                    
                    # Calculate relative risk (compared to baseline)
                    baseline_surv_at_time = baseline_survival.loc[median_time].iloc[0] if median_time in baseline_survival.index else baseline_survival.iloc[0, 0]
                    relative_risk = baseline_surv_at_time / survival_prob if survival_prob > 0 else float('inf')
                    
                    # Risk categories based on survival probability
                    if survival_prob >= 0.8:
                        risk_category = "Low"
                    elif survival_prob >= 0.6:
                        risk_category = "Medium"
                    else:
                        risk_category = "High"
                    
                    predictions.append({
                        'patient_id': i,
                        'risk_score': float(risk_score.iloc[0]) if hasattr(risk_score, 'iloc') else float(risk_score),
                        'survival_probability': float(survival_prob),
                        'relative_risk': float(relative_risk) if relative_risk != float('inf') else "Very High",
                        'risk_category': risk_category,
                        'prediction_time': float(median_time),
                        'confidence': "Model trained on synthetic data - validate with real data"
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to predict for patient {i}: {e}")
                    predictions.append({
                        'patient_id': i,
                        'error': str(e),
                        'note': 'Prediction failed for this patient'
                    })
            
            logger.info(f"Successfully generated predictions for {len(predictions)} patients")
            logger.info(f"Sample prediction: {predictions[0] if predictions else 'No predictions'}")
            return predictions
            
        except Exception as e:
            logger.error(f"Prediction generation failed: {e}")
            raise
    
    def predict_single(self, patient_data: Dict) -> Dict:
        """
        Get prediction for a single patient.
        
        Args:
            patient_data: Dictionary with patient features
            
        Returns:
            Prediction dictionary for the patient
        """
        prediction = self.get_prediction([patient_data])[0]
        logger.info(f"Single prediction result: {prediction}")
        return prediction
    
    def get_feature_importance(self) -> Dict:
        """
        Get feature importance from the trained Cox model.
        
        Returns:
            Dictionary with feature names and their coefficients
        """
        if self.model is None:
            raise RuntimeError("No trained model. Call train() first.")
        
        try:
            # Get coefficients from the fitted model
            coefficients = self.model.params_
            p_values = self.model.summary.p
            
            # Create feature importance dictionary
            importance = {}
            for feature, coef in coefficients.items():
                importance[feature] = {
                    'coefficient': float(coef),
                    'hazard_ratio': float(np.exp(coef)),
                    'p_value': float(p_values[feature]),
                    'interpretation': self._interpret_coefficient(coef, p_values[feature])
                }
            
            return importance
            
        except Exception as e:
            logger.error(f"Failed to get feature importance: {e}")
            return {}
    
    def _interpret_coefficient(self, coef: float, p_value: float) -> str:
        """Interpret a coefficient's clinical meaning."""
        if p_value > 0.05:
            return "Not statistically significant"
        
        if coef > 0:
            return "Increases risk (hazard)"
        elif coef < 0:
            return "Decreases risk (hazard)"
        else:
            return "No effect on risk"
    
    def save_pickle(self, filepath: str):
        """
        Save the trained model to a pickle file.
        
        Args:
            filepath: Path where to save the model
            
        Note: This is a placeholder implementation for now
        """
        if self.model is None:
            raise RuntimeError("No trained model. Call train() first.")
        
        logger.info(f"Saving model to {filepath}...")
        
        # TODO: Implement privacy-compliant saving
        # For now, just save the model object
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'config': self.config,
                'training_stats': self.training_stats,
                'feature_names': self.feature_names,
                'version': '0.2.0'
            }, f)
        
        logger.info(f"Model saved successfully to {filepath}")
    
    def get_train_stats(self) -> Dict:
        """
        Return training statistics.
        
        Returns:
            Dict containing training metrics
        """
        if self.training_stats is None:
            raise RuntimeError("No training stats available. Call train() first.")
        
        return self.training_stats.copy()
    
    def get_input_schema(self) -> Dict:
        """
        Return the required input schema for predictions.
        
        Returns:
            Dict describing expected input format
        """
        return {
            'required_features': self.feature_names,
            'outcome': self.outcome_name,
            'example_input': {feature: f"<{feature}_value>" for feature in self.feature_names},
            'note': 'Input schema generation not fully implemented yet'
        }
    
    def __repr__(self):
        """String representation of the wrapper."""
        status = []
        if self.data is not None:
            status.append(f"data_loaded({self.data.shape[0]} rows)")
        if self.model is not None:
            status.append("model_trained")
        
        status_str = ", ".join(status) if status else "empty"
        return f"CoxModelWrapper(outcome={self.outcome_name}, status={status_str})"
