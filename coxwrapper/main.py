"""
CoxWrapper - Main entry point for command line usage
"""

import sys
import argparse
import logging
from pathlib import Path

from coxwrapper import CoxModelWrapper, __version__

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_example_config(output_path: str):
    """Create an example configuration file."""
    example_config = """
# Example CoxWrapper Configuration
metadata:
  schema_version: 1.1.0
  description: "Example configuration for CoxWrapper"
  created_date: "2024-01-01"

cohort_parameters:
  min_lookback_days: 365
  min_followup_days: 1825

outcome:
  name: "copd"
  domain: "condition_occurrence"
  description: "COPD diagnosis"
  concepts_include: [255573]
  concepts_exclude: []

co_indicators:
  - name: "diabetes"
    domain: "condition_occurrence"
    description: "Diabetes diagnosis"
    concepts_include: [201826]

features:
  - name: "age_at_time_0"
    domain: "person"
    type: "continuous"
    description: "Age at baseline"

  - name: "bmi"
    domain: "measurement"
    type: "continuous"
    description: "Body Mass Index"
    concepts_include: [3038553]

model_io_columns:
  duration_col: "time_to_event_days"
  event_col: "event_observed"

model_features_final:
  - "age_at_time_0"
  - "bmi"
  - "diabetes"
"""

    with open(output_path, 'w') as f:
        f.write(example_config)
    logger.info(f"Example configuration created at: {output_path}")


def main():
    """Main entry point for CoxWrapper CLI."""
    parser = argparse.ArgumentParser(
        description="CoxWrapper - Cox proportional hazards modeling for AllOfUs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create an example configuration file
  python -m coxwrapper create-config example_config.yaml

  # Show package information
  python -m coxwrapper info
        """
    )

    parser.add_argument(
        '--version', action='version',
        version=f'CoxWrapper {__version__}'
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Create config command
    config_parser = subparsers.add_parser(
        'create-config',
        help='Create an example configuration file'
    )
    config_parser.add_argument(
        'output_path',
        help='Path where to save the example configuration'
    )

    # Info command
    subparsers.add_parser(
        'info',
        help='Show package information and environment details'
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == 'create-config':
        create_example_config(args.output_path)
    elif args.command == 'info':
        print(f"CoxWrapper v{__version__}")
        print("=" * 40)
        print(f"Package location: {Path(__file__).parent}")
        print("\nAvailable modules:")
        try:
            from coxwrapper.model import CoxModelWrapper
            print("✓ CoxModelWrapper available")
        except ImportError as e:
            print(f"✗ CoxModelWrapper not available: {e}")

        try:
            import lifelines
            print(f"✓ lifelines v{lifelines.__version__} available")
        except ImportError:
            print("✗ lifelines not available")

        try:
            import pandas
            print(f"✓ pandas v{pandas.__version__} available")
        except ImportError:
            print("✗ pandas not available")

        try:
            import numpy
            print(f"✓ numpy v{numpy.__version__} available")
        except ImportError:
            print("✗ numpy not available")


if __name__ == "__main__":
    main()
