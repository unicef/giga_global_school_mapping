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

    clean_utils.generate_ghsl_per_country(config, layer="ghsl")
    schools = clean_utils.clean_data(config, category="school", gee=False)
    nonschools = clean_utils.clean_data(config, category="non_school", gee=False)


if __name__ == "__main__":
    main()
