"""
CoxWrapper - A package for Cox proportional hazards modeling on AllOfUs data.

This package provides tools for:
- Loading and preprocessing AllOfUs data
- Training Cox proportional hazards models
- Making predictions with trained models
- Working in Jupyter environments within AllOfUs platform
"""

__version__ = "0.2.0"
__author__ = "CoxWrapper Team"
__description__ = "Cox proportional hazards modeling for AllOfUs research platform"

from .model.wrapper import CoxModelWrapper

__all__ = ["CoxModelWrapper"]
