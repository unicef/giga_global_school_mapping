import os
import pandas as pd
import geopandas as gpd

import sys

sys.path.insert(0, "../utils/")
import clean_utils
import config_utils

import warnings

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")


def main():
    cwd = os.path.dirname(os.getcwd())
    config_file = os.path.join(cwd, "configs/data_config.yaml")
    config = config_utils.create_config(config_file)

    schools = clean_utils.clean_data(config, category="school")
    nonschools = clean_utils.clean_data(config, category="non_school")
    clean_utils.augment_non_school_data(config, category="non_school")


if __name__ == "__main__":
    main()
