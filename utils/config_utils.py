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
