import os
import pandas as pd
import geopandas as gpd

import sys
sys.path.insert(0, "../utils/")
import download_utils
import config_utils

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")


def main():
    # Load config file
    cwd = os.path.dirname(os.getcwd())
    config_file = os.path.join(cwd, "configs/data_config.yaml")
    config = config_utils.create_config(config_file)

    # Load UNICEF data
    unicef = download_utils.load_unicef(config)

    # Download Overture data
    overture_schools = download_utils.download_overture(config, category="school")
    overture_nonschools = download_utils.download_overture(config, category="non_school", exclude="school")

    # Download OSM data
    osm_schools = download_utils.download_osm(config, category="school")
    osm_nonschools = download_utils.download_osm(config, category="non_school")

    # Download Microsoft Building Footprints
    download_utils.download_ms(config, verbose=True)
    

if __name__ == "__main__":
    main()
