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


def download_images(
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
    cwd = os.path.dirname(os.getcwd())
    vectors_dir = config["VECTORS_DIR"]

    if not filename:
        filename = f"{iso}_{name}.geojson"
        filename = os.path.join(cwd, vectors_dir, category, name, filename)
    data = gpd.read_file(filename).reset_index(drop=True)

    if iso:
        data = data[data["iso"] == iso].reset_index(drop=True)
    if sample_size:
        data = data.iloc[:sample_size]
    data = data_utils._convert_to_crs(data, data.crs, config["SRS"])
    logging.info(f"Data dimensions: {data.shape}, CRS: {data.crs}")

    out_dir = os.path.join(cwd, config["RASTERS_DIR"], config["DIR"], iso, category)
    out_dir = data_utils._makedir(out_dir)

    url = f"{config['DIGITALGLOBE_URL']}connectid={creds['CONNECT_ID']}"
    wms = WebMapService(url, username=creds["USERNAME"], password=creds["PASSWORD"])

    for index in tqdm(range(len(data))):
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
