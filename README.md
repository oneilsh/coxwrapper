# CoxWrapper

A Python package for Cox proportional hazards modeling on AllOfUs research platform data.

## Overview

CoxWrapper provides a clean, simplified interface for building and deploying Cox proportional hazards survival models. It's designed specifically for use in Jupyter environments within the AllOfUs research platform, but can also be used with synthetic data for testing and development.

## Features

- **Simplified API**: Clean, easy-to-use interface for Cox modeling
- **AllOfUs Integration**: Built-in support for AllOfUs data structures and BigQuery
- **Mock Data Support**: Generate synthetic data for testing and development
- **Comprehensive Preprocessing**: Automated feature preprocessing with outlier handling
- **Model Persistence**: Save and load trained models
- **Privacy-Compliant**: Built with AllOfUs privacy requirements in mind

## Installation

### From Source (Development)

```bash
# Clone the repository
git clone https://github.com/your-org/coxwrapper.git
cd coxwrapper

# Install with development dependencies
pip install -e ".[dev]"

# Or install just the package
pip install -e .
```

### In Jupyter Environments

For use in AllOfUs Workbench or other Jupyter environments:

```bash
# Install from PyPI (when available)
pip install coxwrapper

# Or install from local directory
pip install /path/to/coxwrapper
```

## Quick Start

### Basic Usage with Mock Data

```python
from coxwrapper import CoxModelWrapper

# Initialize with mock data for testing
wrapper = CoxModelWrapper()

# Load mock data (for development/testing)
wrapper.load_data(use_mock=True, n_patients=1000)

# Train the model
wrapper.train(split=0.8, random_state=42)

# Get predictions for new patients
patient_data = [
    {
        'age_at_time_0': 65.0,
        'bmi': 28.5,
        'diabetes': 1,
        'smoking_status': 'Former',
        'ethnicity': 'Not Hispanic or Latino',
        'sex_at_birth': 'Male'
    }
]

predictions = wrapper.get_prediction(patient_data)
print(predictions)
```

### Using with AllOfUs Data

```python
from coxwrapper import CoxModelWrapper

# Load configuration
config = CoxModelWrapper('path/to/your/config.yaml')

# Load data from BigQuery (requires proper environment setup)
config.load_data(use_mock=False)

# Train model
config.train(penalizer=0.01, l1_ratio=0.0)

# Save model
config.save_pickle('my_cox_model.pkl')
```

## Configuration

CoxWrapper uses YAML configuration files to define:

- **Outcome**: The event of interest (e.g., COPD diagnosis)
- **Features**: Predictor variables from various OMOP domains
- **Cohort Parameters**: Time windows and inclusion criteria
- **Model Settings**: Hyperparameters and preprocessing options

### Creating an Example Configuration

```bash
python -m coxwrapper create-config my_config.yaml
```

### Configuration Structure

```yaml
metadata:
  schema_version: 1.1.0
  description: "COPD risk prediction model"

cohort_parameters:
  min_lookback_days: 365
  min_followup_days: 1825

outcome:
  name: "copd"
  domain: "condition_occurrence"
  concepts_include: [255573, 321316]  # COPD concepts
  concepts_exclude: [4288734]         # Exclusions

features:
  - name: "age_at_time_0"
    domain: "person"
    type: "continuous"
    description: "Age at baseline"

  - name: "bmi"
    domain: "measurement"
    type: "continuous"
    concepts_include: [3038553]

  - name: "smoking_status"
    domain: "observation"
    type: "categorical"
    concepts_include: [1585856]

model_features_final:
  - "age_at_time_0"
  - "bmi"
  - "smoking_status"

model_io_columns:
  duration_col: "time_to_event_days"
  event_col: "event_observed"
```

## API Reference

### CoxModelWrapper

Main class for Cox proportional hazards modeling.

#### Methods

- `__init__(config)`: Initialize wrapper with configuration
- `load_data(use_mock=False, n_patients=1000)`: Load data from BigQuery or generate mock data
- `train(split=0.8, random_state=42, **model_kwargs)`: Train the Cox model
- `get_prediction(pt_data)`: Get predictions for patient data
- `predict_single(patient_data)`: Get prediction for single patient
- `get_feature_importance()`: Get feature importance from trained model
- `save_pickle(filepath)`: Save trained model to pickle file
- `get_train_stats()`: Get training statistics
- `get_input_schema()`: Get expected input schema

### Direct Model Functions

For more advanced usage, you can use the underlying functions directly:

```python
from coxwrapper.model import (
    run_end_to_end_pipeline,
    evaluate_cox_model,
    load_configuration,
    load_data_from_bigquery
)

# Run complete pipeline
model, metrics, model_path, preprocessor_path = run_end_to_end_pipeline(
    config_path='config.yaml',
    test_size=0.2,
    penalizer=0.01
)
```

## Environment Setup

### AllOfUs Workbench

For use in AllOfUs Workbench, ensure you have:

1. **Environment Variables**:
   ```bash
   export WORKSPACE_CDR="all-of-us-research-workbench-####.r2023q3_unzipped_data"
   export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
   ```

2. **Required Permissions**: Access to BigQuery datasets and necessary OMOP tables

3. **Python Environment**: Python 3.10+ with required packages

### Local Development

For local development without BigQuery access:

```python
# Use mock data
wrapper = CoxModelWrapper()
wrapper.load_data(use_mock=True, n_patients=1000)
```

## Model Persistence

### Saving Models

```python
# Save trained model
wrapper.save_pickle('cox_model.pkl')

# Or set custom output directory
import os
os.environ['COXWRAPPER_OUTPUT_DIR'] = '/path/to/models'
wrapper.save_pickle('cox_model.pkl')
```

### Loading Models

```python
import pickle

# Load saved model
with open('cox_model.pkl', 'rb') as f:
    saved_model = pickle.load(f)

# Use for predictions
# (Note: You'll need to recreate the wrapper with proper preprocessing)
```

## Data Privacy

CoxWrapper is designed with privacy in mind:

- **Small Counts Masking**: Automatically masks counts ≤20 as "<21"
- **No PHI in Logs**: Logging avoids printing sensitive patient data
- **Configurable Outputs**: Model outputs can be controlled per environment

## Development

### Running Tests

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=coxwrapper
```

### Code Quality

```bash
# Format code
black coxwrapper/

# Lint code
flake8 coxwrapper/

# Type checking
mypy coxwrapper/
```

## Troubleshooting

### Common Issues

1. **BigQuery Connection Issues**
   - Ensure `WORKSPACE_CDR` and `GOOGLE_CLOUD_PROJECT` are set
   - Check BigQuery permissions

2. **Missing Dependencies**
   ```bash
   pip install --upgrade lifelines pandas numpy scikit-learn
   ```

3. **Memory Issues**
   - Use smaller mock datasets for testing
   - Consider data sampling for large BigQuery queries

4. **Import Errors**
   - Ensure package is properly installed: `pip install -e .`
   - Check Python path includes the package directory

### Getting Help

- Check the [Issues](https://github.com/your-org/coxwrapper/issues) page
- Review the configuration examples
- Use `python -m coxwrapper info` to check your environment

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.
