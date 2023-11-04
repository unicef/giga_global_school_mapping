import re
import os
import uuid
import requests
import logging

import geojson
import numpy as np
import pandas as pd
import geopandas as gpd

from tqdm import tqdm
from pyproj import Proj, Transformer
from scipy.sparse.csgraph import connected_components

pd.options.mode.chained_assignment = None
logging.basicConfig(level=logging.INFO)


def _clean_text(text):
    """Remove all non-word characters (everything except numbers and letters)"""
    if text:
        text = re.sub(r"[^\w\s]", " ", text)
        text = text.upper()
    else:
        text = ""
    return text


def _makedir(out_dir):
    """Creates a new directory in current working directory."""

    cwd = os.path.dirname(os.getcwd())
    out_dir = os.path.join(cwd, out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    return out_dir


def _get_iso_regions(config, data, iso_code):
    """Adds country, region, and subregion to dataframe."""

    # Load ISO codes of countries and regions/subregions
    codes = pd.read_csv(config["iso_regional_codes"])
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


def _concat_data(data, out_file=None, verbose=True):
    """Concatenates a list of datasets and converts it to a GeoDataFrame."""

    data = pd.concat(data).reset_index(drop=True)
    data = gpd.GeoDataFrame(data, geometry=data["geometry"], crs="EPSG:4326")
    
    if out_file:
        data.to_file(out_file, driver="GeoJSON")
        logging.info(f"Generated {out_file}")
    
    if verbose:
        logging.info(f"Data dimensions: {data.shape}, CRS: {data.crs}")
    
    return data


def _generate_uid(data, category):
    """Generates a unique id based on source, iso, and category."""

    data["index"] = data.index.to_series().apply(lambda x: str(x).zfill(8))
    data["category"] = category
    uids = data[["source", "iso", "category", "index"]].agg("-".join, axis=1)
    data = data.drop(["index", "category"], axis=1)
    data["UID"] = uids.str.upper()
    return data


def _prepare_data(config, data, iso_code, category, source, columns, out_file=None):
    """Prepare datasets to have uniform columnns"""

    if "name" not in data.columns:
        data["name"] = None
    data["source"] = source.upper()
    data = data.drop_duplicates("geometry", keep="first")
    data = _get_iso_regions(config, data, iso_code)
    data = _generate_uid(data, category)
    data = data[columns]
    if out_file:
        data.to_file(out_file, driver="GeoJSON")
    return data


def _get_geoboundaries(config, iso_code, out_dir=None, adm_level="ADM0"):
    """Fetches the geoboundary given an ISO code"""

    # Query geoBoundaries
    if not out_dir:
        cwd = os.path.dirname(os.getcwd())
        out_dir = os.path.join(cwd, config['rasters_dir'], "geoboundaries")
    if not os.path.exists(out_dir):
        out_dir = _makedir(out_dir)
        
    try:
        url = f"{config['geoboundaries_url']}{iso_code}/{adm_level}/"
        r = requests.get(url)
        download_path = r.json()["gjDownloadURL"]
    except:
        url = f"{config['geoboundaries_url']}{iso_code}/ADM0/"
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


def _read_data(data_dir, exclude=[]):
    """Reads and concatenates data from a directory of files."""

    data_dir = _makedir(data_dir)
    files = next(os.walk(data_dir), (None, None, []))[2]
    files = [file for file in files if file not in exclude]

    data = []
    bar_format = "{l_bar}{bar:20}{r_bar}{bar:-20b}"
    pbar = tqdm(files, total=len(files), bar_format=bar_format)
    for file in pbar:
        pbar.set_description(f"Reading {file}")
        filename = os.path.join(data_dir, file)
        subdata = gpd.read_file(filename)
        data.append(subdata)

    # Concatenate files in data_dir
    data = gpd.GeoDataFrame(pd.concat(data).copy(), crs="EPSG:4326")
    return data


def _drop_duplicates(data, priority):
    data["temp_source"] = pd.Categorical(
        data["source"], categories=priority, ordered=True
    )
    data = data.sort_values("temp_source", ascending=True).drop_duplicates(["group"])
    data = data.reset_index(drop=True)
    return data


def _connect_components(data, buffer_size):
    # Dissolve overlapping geometries based on: https://gis.stackexchange.com/a/271737

    temp = data.copy()
    if data.crs != "EPSG:3857":
        temp = _convert_to_crs(data, target_crs="EPSG:3857")
    geometry = temp["geometry"].buffer(buffer_size)
    overlap_matrix = geometry.apply(lambda x: geometry.overlaps(x)).values.astype(int)
    n, groups = connected_components(overlap_matrix, directed=False)
    data["group"] = groups
    return data
