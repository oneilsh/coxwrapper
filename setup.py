"""
Setup script for CoxWrapper package.
"""

from setuptools import setup, find_packages

setup(
    name="coxwrapper",
    version="0.2.0",
    packages=["coxwrapper", "coxwrapper.model"],
    package_dir={"coxwrapper": "coxwrapper"},
    include_package_data=True,
    install_requires=[
        "lifelines>=0.27.5",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "google-cloud-bigquery>=3.0.0",
        "pyyaml>=6.0",
        "requests>=2.31.0",
        "joblib>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "coxwrapper=coxwrapper.main:main",
        ]
    },
    # Package data
    package_data={
        "coxwrapper": ["*.yaml"],
    },
)
