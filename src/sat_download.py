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
    category=None,
    iso=None,
    sample_size=None,
    src_crs="EPSG:4326",
    id_col="UID",
    name="clean",
    data=None,
    filename=None,
    out_dir=None,
    download_validated=False
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
    
    if data is None:
        if not filename:
            vectors_dir = config["vectors_dir"]
            filename = f"{iso}_{name}.geojson"
            filename = os.path.join(cwd, vectors_dir, category, name, filename)
        data = gpd.read_file(filename).reset_index(drop=True)
    
    if "clean" in data.columns:
        data = data[data["clean"] == 0]
    if "validated" in data.columns and not download_validated:
        data = data[data["validated"] == 0]
    if 'iso' in data.columns:
        data = data[data["iso"] == iso].reset_index(drop=True)
    if sample_size:
        data = data.iloc[:sample_size]

    data = data_utils._convert_crs(data, data.crs, config["srs"])
    logging.info(f"Data dimensions: {data.shape}, CRS: {data.crs}")

    if not out_dir:
        out_dir = os.path.join(cwd, config["rasters_dir"], config["maxar_dir"], iso, category)
    out_dir = data_utils._makedir(out_dir)

    url = f"{config['digitalglobe_url']}connectid={creds['connect_id']}"
    wms = WebMapService(url, username=creds["username"], password=creds["password"])

    bar_format = "{l_bar}{bar:20}{r_bar}{bar:-20b}"
    for index in tqdm(range(len(data)), bar_format=bar_format):
        image_file = os.path.join(out_dir, f"{data[id_col][index]}.tiff")
        while not os.path.exists(image_file):
            try:
                bbox = (
                    data.lon[index] - config["size"],
                    data.lat[index] - config["size"],
                    data.lon[index] + config["size"],
                    data.lat[index] + config["size"],
                )
                img = wms.getmap(
                    bbox=bbox,
                    layers=config["layers"],
                    srs=config["srs"],
                    size=(config["width"], config["height"]),
                    featureProfile=config["featureprofile"],
                    coverage_cql_filter=config["coverage_cql_filter"],
                    exceptions=config["exceptions"],
                    transparent=config["transparent"],
                    format=config["format"],
                )
                with open(image_file, "wb") as file:
                    file.write(img.read())
            except Exception as e: 
                #logging.info(e)
                pass


def main():
    # Parser
    parser = argparse.ArgumentParser(description="Satellite Image Download")
    parser.add_argument("--config", help="Config file")
    parser.add_argument("--creds", help="Credentials file")
    parser.add_argument("--category", help="Category (e.g. school or non_school)")
    parser.add_argument("--iso", help="ISO code")
    parser.add_argument("--filename", help="Data file", default=None)
    args = parser.parse_args()

    # Load config
    config = config_utils.load_config(args.config)
    creds = config_utils.create_config(args.creds)

    # Download satellite images
    download_images(creds, config, iso=args.iso, category=args.category, filename=args.filename)


if __name__ == "__main__":
    main()
