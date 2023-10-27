import os
import time
import math
import random
import logging
import argparse
from tqdm.notebook import tqdm

import pandas as pd
import geopandas as gpd

from owslib.wms import WebMapService
from utils import config_utils
from utils import data_utils


def download_images(
    filename, 
    creds, 
    config, 
    iso=None, 
    src_crs="EPSG:4326", 
    id_col="UID"
):
    url = f"https://evwhs.digitalglobe.com/mapservice/wmsaccess?connectid={creds['CONNECT_ID']}"
    wms = WebMapService(
        url, 
        username=creds['USERNAME'],
        password=creds['PASSWORD']
    )

    data = gpd.read_file(filename).reset_index(drop=True)
    if iso != None: 
        data = data[data['iso'] == iso].reset_index(drop=True)
    data = data_utils._convert_to_crs(data, data.crs, config["SRS"])
    print(f"Data dimensions: {data.shape}, CRS: {data.crs}")
    
    for i in tqdm(range(len(data))):
        bbox=(
            data.lon[i] - config['SIZE'], 
            data.lat[i] - config['SIZE'], 
            data.lon[i] + config['SIZE'], 
            data.lat[i] + config['SIZE'], 
        )
        img = wms.getmap(
            bbox=bbox,
            layers=config['LAYERS'],
            srs=config['SRS'],
            size=(config['WIDTH'], config['HEIGHT']),
            featureProfile=config['FEATUREPROFILE'],
            coverage_cql_filter=config['COVERAGE_CQL_FILTER'],
            exceptions=config['EXCEPTIONS'],
            transparent=config['TRANSPARENT'],
            format=config['FORMAT']          
        )
        image_name = data[id_col][i]
        image_name = f"{config['DIR']}{image_name}.tiff"
        with open(image_name, 'wb') as file:
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
