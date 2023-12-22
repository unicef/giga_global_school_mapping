import os
import time
import math
import random
import logging
import argparse
from tqdm import tqdm

import pandas as pd
import geopandas as gpd

from owslib.wms import WebMapService

import sys

sys.path.insert(0, "../utils/")
import data_utils
import config_utils

SEED = 42
logging.basicConfig(level=logging.INFO)


def download_sat_images(
    creds,
    config,
    category,
    iso=None,
    sample_size=None,
    src_crs="EPSG:4326",
    id_col="UID",
    name="clean",
    filename=None,
):
    """
    Downloads satellite images based on geographic data points.

    Args:
    - creds (dict): Credentials for accessing the satellite image service.
    - config (dict): Configuration settings.
    - category (str): Type of data category.
    - iso (str, optional): ISO code for the country. Defaults to None.
    - sample_size (int, optional): Number of samples to consider. Defaults to None.
    - src_crs (str, optional): Source coordinate reference system. Defaults to "EPSG:4326".
    - id_col (str, optional): Column name containing unique identifiers. Defaults to "UID".
    - name (str, optional): Name of the dataset. Defaults to "clean".
    - filename (str, optional): File name to load the data. Defaults to None.

    Returns:
    - None
    """
    
    cwd = os.path.dirname(os.getcwd())
    vectors_dir = config["VECTORS_DIR"]

    if not filename:
        filename = f"{iso}_{name}.geojson"
        filename = os.path.join(cwd, vectors_dir, category, name, filename)
    data = gpd.read_file(filename).reset_index(drop=True)

    if 'iso' in data.columns:
        data = data[data["iso"] == iso].reset_index(drop=True)
    if sample_size:
        data = data.iloc[:sample_size]
    data = data_utils._convert_crs(data, data.crs, config["SRS"])
    logging.info(f"Data dimensions: {data.shape}, CRS: {data.crs}")

    out_dir = os.path.join(cwd, config["RASTERS_DIR"], config["DIR"], iso, category)
    out_dir = data_utils._makedir(out_dir)

    url = f"{config['DIGITALGLOBE_URL']}connectid={creds['CONNECT_ID']}"
    wms = WebMapService(url, username=creds["USERNAME"], password=creds["PASSWORD"])

    bar_format = "{l_bar}{bar:20}{r_bar}{bar:-20b}"
    for index in tqdm(range(len(data)), bar_format=bar_format):
        image_file = os.path.join(out_dir, f"{data[id_col][index]}.tiff")
        if not os.path.exists(image_file):
            bbox = (
                data.lon[index] - config["SIZE"],
                data.lat[index] - config["SIZE"],
                data.lon[index] + config["SIZE"],
                data.lat[index] + config["SIZE"],
            )
            img = wms.getmap(
                bbox=bbox,
                layers=config["LAYERS"],
                srs=config["SRS"],
                size=(config["WIDTH"], config["HEIGHT"]),
                featureProfile=config["FEATUREPROFILE"],
                coverage_cql_filter=config["COVERAGE_CQL_FILTER"],
                exceptions=config["EXCEPTIONS"],
                transparent=config["TRANSPARENT"],
                format=config["FORMAT"],
            )
            with open(image_file, "wb") as file:
                file.write(img.read())


def main():
    # Parser
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Satellite Image Download")
    parser.add_argument("--config", help="Config file")
    parser.add_argument("--creds", help="Credentials file")
    parser.add_argument("--filename", help="Data file")
    args = parser.parse_args()

    # Load config
    config = config_utils.create_config(args.config)
    creds = config_utils.create_config(args.creds)

    # Download satellite images
    download_images(args.filename, creds, config)


if __name__ == "__main__":
    main()
