"""Module containing submodules for each dataset.

Each dataset is defined as a class in that submodule.
"""
from .util import BaseDataset
from .base_data_module import BaseDataModule

from .chess_positions_module import ChessDataModule
