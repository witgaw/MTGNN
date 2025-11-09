"""
MTGNN: Multivariate Time Series Forecasting with Graph Neural Networks

This package provides a PyTorch implementation of the MTGNN model for time series forecasting.
"""

__version__ = "0.1.0"

from .net import gtnet, MTGNNModel
from .trainer import Trainer, Optim, create_data_loader_from_arrays
from .train_multi_step import train_injected, TrainingConfig
from .layer import *
from .util import *

__all__ = [
    "train_injected",
    "TrainingConfig",
    "MTGNNModel",
    "create_data_loader_from_arrays",
    "gtnet",
    "Trainer",
    "Optim",
]