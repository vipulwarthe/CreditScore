import yaml
import joblib

config_dir = '../config/config.yaml'

def config_load():
    """
    Load configuration data from a YAML file.
    
    Returns
    -------
    dict: Configuration data loaded from the YAML file.
    """
    with open(config_dir, 'r') as file:
        config = yaml.safe_load(file)    
    return config

def pickle_load(file_path):
    """
    Load data from a binary pickle file.

    Args
    ----
    file_path (str): The path to the pickle file.

    Returns
    -------
    Any: Data loaded from the pickle file.
    """
    return joblib.load(file_path)

def pickle_dump(data, file_path):
    """
    Dump data into a binary pickle file.

    Args
    ----
    data (Any): The data to be saved in the pickle file.
    file_path (str): The path to the pickle file where data will be saved.
    """
    joblib.dump(data, file_path)