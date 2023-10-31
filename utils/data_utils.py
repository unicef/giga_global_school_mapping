import re
import os
import uuid
import requests

import geojson
import numpy as np
import pandas as pd
import geopandas as gpd

from tqdm.notebook import tqdm
from pyproj import Proj, Transformer

try:
    import config_utils
except:
    from utils import config_utils

pd.options.mode.chained_assignment = None


def _makedir(out_dir):
    """Creates a new directory in current working directory."""

    cwd = os.path.dirname(os.getcwd())
    out_dir = os.path.join(cwd, out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    return out_dir


def _load_data_config(filename="configs/data_config.yaml"):
    """Loads the data config, which contains the URL of data sources."""

    cwd = os.path.dirname(os.getcwd())
    data_config_file = os.path.join(cwd, filename)
    data_config = config_utils.create_config(data_config_file)
    return data_config


def _get_iso_regions(data, iso_code):
    """Adds country, region, and subregion to dataframe."""

    # Load ISO codes of countries and regions/subregions
    data_config = _load_data_config()
    url = data_config["ISO_REGIONAL_CODES"]
    codes = pd.read_csv(url)

    subcode = codes.query(f"`alpha-3` == '{iso_code}'")
    data["iso"] = iso_code
    data["country"] = subcode["name"].values[0]
    data["subregion"] = subcode["sub-region"].values[0]
    data["region"] = subcode["region"].values[0]

    return data


def _convert_to_crs(data, src_crs="EPSG:4326", target_crs="EPSG:3857"):
    """Converts GeoDataFrame to CRS."""

    # Convert lat long to the target CRS
    if ("lat" not in data.columns) or ("lon" not in data.columns):
        data["lon"], data["lat"] = data.geometry.x, data.geometry.y
    transformer = Transformer.from_crs(src_crs, target_crs, always_xy=True)
    data["lon"], data["lat"] = transformer.transform(
        data["lon"].values, data["lat"].values
    )
    geometry = gpd.GeoSeries.from_xy(data["lon"], data["lat"])

    # Convert the data to GeoDataFrame
    data = pd.DataFrame(data.drop("geometry", axis=1))
    data = gpd.GeoDataFrame(data, geometry=geometry, crs=target_crs)
    return data


def _concat_data(data, out_file, verbose=True):
    """Concatenates data and converts to GeoDataFrame"""

    data = pd.concat(data).reset_index(drop=True)
    data = gpd.GeoDataFrame(data, geometry=data["geometry"], crs="EPSG:4326")
    data.to_file(out_file, driver="GeoJSON")
    if verbose:
        print(f"Data dimensions: {data.shape}, CRS: {data.crs}")
        print(f"Generated {out_file}")

    return data


def _generate_uid(data, category):
    """Generates a unique id based on source, iso, and category."""

    data["index"] = data.index.to_series().apply(lambda x: str(x).zfill(8))
    data["category"] = category
    uids = data[["source", "iso", "category", "index"]].agg("-".join, axis=1)
    data = data.drop(["index", "category"], axis=1)
    data["UID"] = uids
    return data


def _prepare_data(data, iso_code, category, source, columns, out_file=None):
    """Prepare datasets to have uniform columnns"""

    if "name" not in data.columns:
        data["name"] = None
    data["source"] = source
    data = data.drop_duplicates("geometry", keep="first")
    data = _get_iso_regions(data, iso_code)
    data = _generate_uid(data, category)
    data = data[columns]
    if out_file:
        data.to_file(out_file, driver="GeoJSON")
    return data


def _get_geoboundaries(iso_code, out_dir="data/geoboundary", adm_level="ADM0"):
    """Fetches the geoboundary given an ISO code"""

    # Query geoBoundaries
    out_dir = _makedir(out_dir)
    data_config = _load_data_config()
    try:
        url = f"{data_config['GEOBOUNDARIES_URL']}{iso_code}/{adm_level}/"
        r = requests.get(url)
        download_path = r.json()["gjDownloadURL"]
    except:
        url = f"{data_config['GEOBOUNDARIES_URL']}{iso_code}/ADM0/"
        r = requests.get(url)
        download_path = r.json()["gjDownloadURL"]

    # Save the result as a GeoJSON
    filename = f"{iso_code}_geoboundary.geojson"
    out_file = os.path.join(out_dir, filename)
    geoboundary = requests.get(download_path).json()
    with open(out_file, "w") as file:
        geojson.dump(geoboundary, file)

    # Read data using GeoPandas
    geoboundary = gpd.read_file(out_file)
    return geoboundary