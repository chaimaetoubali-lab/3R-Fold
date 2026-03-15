"""
Global configuration settings for 3R-Fold project.
"""

import os

# Define your base path here - change this to your actual path
#BASE_PATH = "/content/drive/MyDrive/Foldv"
BASE_PATH = "./"

def get_base_path():
    """Get the base path for the project."""
    return BASE_PATH

def set_base_path(path):
    """Set the base path for the project."""
    global BASE_PATH
    BASE_PATH = path

# Commonly used subdirectories that you might need
def get_data_path():
    """Get the data directory path."""
    return os.path.join(BASE_PATH, "data")

def get_results_path():
    """Get the results directory path."""
    return os.path.join(BASE_PATH, "results")

def get_models_path():
    """Get the models directory path."""
    return os.path.join(BASE_PATH, "models")

def get_plots_path():
    """Get the plots directory path."""
    return os.path.join(BASE_PATH, "results", "plots")
