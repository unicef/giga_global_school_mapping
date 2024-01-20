"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import yaml
from easydict import EasyDict


def create_config(config_file, prefix=""):
    """Loads YAML config file as dictionary.

    Args:
    - config_file_exp (str): Path to config file.
    - prefix (str): Config file prefix.

    Returns:
    - dict: The config as a dictionary.
    """
    with open(config_file, "r") as stream:
        config = yaml.safe_load(stream)

    cfg = EasyDict()
    for k, v in config.items():
        cfg[k] = v
        if "DIR" in k:
            if len(prefix) > 0:
                cfg[k] = prefix + cfg[k]
    return cfg

def load_config(config_file_exp, prefix=""):
    """Loads configuration files with system configurations included.

    Args:
        config_file_exp (str): Path to the experiment-specific configuration file.
        prefix (str): Prefix to be added to file paths in the configuration.

    Returns:
        dict: The loaded configuration as a dictionary.
    """
    cwd = os.path.dirname(os.getcwd())
    sys_config_file = f"{cwd}/configs/config.yaml"
    sys_config = create_config(sys_config_file, prefix=prefix)
    config = create_config(config_file_exp, prefix=prefix)
    config['config_name'] = os.path.basename(config_file_exp).split('.')[0]
    remove = []
    for key, val in sys_config.items():
        if key in config:
            remove.append(key)
    for key in remove:
        sys_config.pop(key)
    config.update(sys_config)
    
    return config