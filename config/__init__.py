"""
Configuration package for 3R-Fold project.
"""

from .paths import (
    get_base_path,
    set_base_path,
    get_data_path,
    get_results_path,
    get_models_path,
    get_plots_path,
    BASE_PATH
)

__all__ = [
    'get_base_path',
    'set_base_path', 
    'get_data_path',
    'get_results_path',
    'get_models_path',
    'get_plots_path',
    'BASE_PATH'
]
