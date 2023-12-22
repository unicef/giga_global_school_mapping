import re
import os
import time
import duckdb
import geojson
import overpass
import data_utils
import logging
import leafmap
import pyproj
import subprocess
import operator

from tqdm import tqdm
import pandas as pd
import geopandas as gpd
from rapidfuzz import fuzz
from country_bounding_boxes import country_subunits_by_iso_code

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
logging.basicConfig(level=logging.INFO)
os.environ['PROJ_LIB'] =pyproj.datadir.get_data_dir()


def _query_osm(iso_code, out_file, query):
    """
    Queries OpenStreetMap (OSM) for a given ISO code and saves the result in a GeoJSON file.

    Args:
    - iso_code (str): ISO code of the country.
    - out_file (str): File path to save the queried OSM data in GeoJSON format.
    - query (str): OSM query string.

    Returns:
    - GeoDataFrame: GeoDataFrame containing the queried OSM data.
    """

    api = overpass.API(timeout=1500)
    osm_query = f"""
        area["ISO3166-1"="{iso_code}"][admin_level=2];
        ({query});
        out center;
    """
    data = api.get(osm_query, verbosity="geom")
    with open(out_file, "w") as file:
        geojson.dump(data, file)

    data = gpd.read_file(out_file)
    return data


def download_osm(config, category, source="osm"):
    """
    Downloads OpenStreetMap (OSM) Points of Interest (POIs) based on a list of ISO codes.

    Args:
    - config (dict): Configuration settings.
        - "iso_codes" (list): List of ISO country codes.
        - "vectors_dir" (str): Directory to store vector data.
        - "columns" (list): List of columns for the dataset.
        - "iso_regional_codes" (str): URL or filepath to the ISO regional codes data CSV file.
    - category (str): Type of data category to download.
    - source (str, optional): Source identifier for the downloaded data. Defaults to "osm".

    Returns:
    - GeoDataFrame: Combined and processed GeoDataFrame containing OSM POIs.
    """

    out_dir = os.path.join(config["vectors_dir"], category, source)
    out_dir = data_utils._makedir(out_dir)
    osm_file = os.path.join(os.path.dirname(out_dir), f"{source}.geojson")
    iso_codes = config['iso_codes']

    url = config["iso_regional_codes"]
    codes = pd.read_csv(url)

    keywords = config[category]
    query = "".join(
        [
            f"""node["{key}"~"^({"|".join(values)})"](area);
        way["{key}"~"^({"|".join(values)})"](area);
        rel["{key}"~"^({"|".join(values)})"](area);
        """
            for key, values in keywords.items()
            if key != "translated"
        ]
    )

    data = []
    for iso_code in (pbar := data_utils._create_progress_bar(iso_codes)):
        pbar.set_description(f"Processing {iso_code}")
        filename = f"{iso_code}_{source}.geojson"
        out_subfile = os.path.join(out_dir, filename)

        if not os.path.exists(out_subfile):
            alpha_2 = codes.query(f"`alpha-3` == '{iso_code}'")["alpha-2"].values[0]
            _query_osm(alpha_2, out_subfile, query)

        subdata = gpd.read_file(out_subfile).reset_index(drop=True)
        if len(subdata) > 0:
            subdata = data_utils._prepare_data(
                config=config,
                data=subdata,
                iso_code=iso_code,
                category=category,
                source=source,
                columns=config["columns"],
                out_file=out_subfile,
            )
            data.append(subdata)

    # Combine data and save to file
    data = data_utils._concat_data(data, osm_file)
    return data


def _query_overture(config, iso_code, out_file, query):
    """
    Queries Overture Map for a given ISO code.

    Args:
    - config (dict): Configuration settings.
        - "overture_url" (str): URL to the Overture AWS S3 storage bucket.
    - iso_code (str): ISO code for a specific country.
    - out_file (str): Output file path to save the queried data.
    - query (str): Query string to filter data from Overture Map.

    Returns:
    - GeoDataFrame: Queried and processed data as a GeoDataFrame.
    """

    # Connect to DuckDB
    db = duckdb.connect()
    db.execute("INSTALL spatial")
    db.execute("INSTALL httpfs")
    db.execute(
        """
        LOAD spatial;
        LOAD httpfs;
        SET s3_region='us-west-2';
    """
    )

    # Fetch country bounding box and execute Overture query
    url = config["overture_url"]
    bbox = [c.bbox for c in country_subunits_by_iso_code(iso_code)][0]
    overture_query = f"""
        COPY(
            select
            JSON(names) AS names,
            ST_GeomFromWkb(geometry) AS geometry
        from
            read_parquet('{url}')
        where
            bbox.minx > {bbox[0]}
            and bbox.miny > {bbox[1]}
            and bbox.maxx < {bbox[2]}
            and bbox.maxy < {bbox[3]}
            and (
            {query}
            )
        ) TO '{out_file}'
        WITH (FORMAT GDAL, DRIVER 'GeoJSON')
    """
    db.execute(overture_query).fetchall()

    # Intersect data with the country's geoboundaries
    data = gpd.read_file(out_file)
    return data


def download_overture(config, category, exclude=None, source="overture"):
    """
    Downloads Overture Map points of interest (POIs) based on a list of ISO codes.

    Args:
    - config (dict): Configuration settings.
        - "iso_codes" (list): List of ISO country codes.
        - "columns" (list): List of columns for the dataset.
        - "vectors_dir" (str): Directory to store vector data.
    - category (str): Type of data category ('school' or 'non_school').
    - exclude (str or None, optional): Keywords to exclude Defaults to None.
    - source (str, optional): Source identifier for Overture Map dataset. Defaults to 'overture'.

    Returns:
    - GeoDataFrame: Combined GeoDataFrame containing downloaded Overture Map POIs.
    """

    # Generate output directory
    out_dir = os.path.join(config["vectors_dir"], category, source)
    out_dir = data_utils._makedir(out_dir)
    overture_file = os.path.join(os.path.dirname(out_dir), f"{source}.geojson")
    iso_codes = config['iso_codes']

    # Fetch school keywords and generate keywords query
    keywords = config[category] 
    query = " or ".join(
        [
            f"""UPPER(names) LIKE '%{keyword.replace('_', ' ').upper()}%'
        """
            for key, values in keywords.items()
            for keyword in values
        ]
    )

    if exclude:
        exclude_keywords = config[exclude]
        exclude_query = " and ".join(
            [
                f"""UPPER(names) NOT LIKE '%{keyword.replace('_', ' ').upper()}%'
            """
                for key, values in exclude_keywords.items()
                for keyword in values
            ]
        )
        query = f""" ({query}) and ({exclude_query})"""

    data = []
    for iso_code in (pbar := data_utils._create_progress_bar(iso_codes)):
        pbar.set_description(f"Processing {iso_code}")
        filename = f"{iso_code}_{source}.geojson"
        out_subfile = os.path.join(out_dir, filename)

        # Fetch Overture data
        if not os.path.exists(out_subfile):
            _query_overture(config, iso_code, out_subfile, query)

        # Post-process data
        subdata = gpd.read_file(out_subfile).reset_index(drop=True)
        if len(subdata) > 0:
            # Extract name from overture data
            if "names" in subdata.columns:
                geoboundary = data_utils._get_geoboundaries(config, iso_code)
                subdata["name"] = subdata["names"].apply(
                    lambda x: x["common"][0]["value"]
                )
                subdata = gpd.sjoin(subdata, geoboundary, predicate="within")

            subdata = data_utils._prepare_data(
                config=config,
                data=subdata,
                iso_code=iso_code,
                category=category,
                source=source,
                columns=config["columns"],
                out_file=out_subfile,
            )
            data.append(subdata)

    # Combine datasets
    data = data_utils._concat_data(data, overture_file)
    return data


def load_unicef(config, category="school", source="unicef"):
    """
    Combines the UNICEF/Giga datasets into a single GeoJSON file.

    Args:
    - config (dict): Configuration settings.
        - "columns" (list): List of columns for the dataset.
    - category (str, optional): Type of data category ('school' or 'non_school'). 
      Defaults to 'school'.
    - source (str, optional): Source identifier for UNICEF/Giga dataset. 
      Defaults to 'unicef'.

    Returns:
    - GeoDataFrame: Combined GeoDataFrame from UNICEF/Giga datasets.
    """

    # Generate data directory
    data_dir = os.path.join(config["vectors_dir"], category, source)
    data_dir = data_utils._makedir(data_dir)
    files = next(os.walk(data_dir), (None, None, []))[2]
    logging.info(f"Number of CSV files: {len(files)}")

    data = []
    for file in (pbar := data_utils._create_progress_bar(files)):
        iso_code = file.split("_")[0]
        pbar.set_description(f"Processing {iso_code}")

        filename = os.path.join(data_dir, file)
        subdata = pd.read_csv(filename).reset_index(drop=True)
        subdata["geometry"] = gpd.GeoSeries.from_xy(subdata["lon"], subdata["lat"])
        columns = config["columns"]

        subdata = data_utils._prepare_data(
            config=config,
            data=subdata,
            iso_code=iso_code,
            category=category,
            source=source,
            columns=columns,
        )
        data.append(subdata)

    # Combine unicef datasets
    out_dir = os.path.dirname(data_dir)
    out_file = os.path.join(out_dir, f"{source}.geojson")
    data = data_utils._concat_data(data, out_file)
    return data


def _get_ms_country(config):
    """
    Match ISO country codes to Microsoft (MS) Building Footprints country names 
    using fuzzy string matching.

    Parameters:
    - config (dict): A dictionary containing configuration settings.
        - "iso_codes" (list): List of ISO country codes.
        - "microsoft_url" (str): URL or file path to the MS country data CSV file.

    Returns:
    - matches (dict): A dictionary mapping ISO country codes to their respective best-matched 
    MS country names.
    """
    
    iso_codes = config["iso_codes"]   
    msf_links = pd.read_csv(config["microsoft_url"])
    matches = dict()
    for iso_code in iso_codes:
        country, _, _ = data_utils._get_iso_regions(config, iso_code)  
        max_score = 0
        for msf_country in msf_links.Location.unique():
            msf_country_ = re.sub(r"(\w)([A-Z])", r"\1 \2", msf_country)
            score = fuzz.partial_token_sort_ratio(country, msf_country_)
                          
            if score > max_score:
                max_score = score
                matches[iso_code] = msf_country
    return matches


def download_ms(config, source="ms", verbose=False):
    """
    Download and process Microsoft (MS) Building Footprints data for specified ISO country codes.

    Parameters:
    - config (dict): A dictionary containing configuration settings.
        - "iso_codes" (list): List of ISO country codes.
        - "vectors_dir" (str): Directory to store vector data.
        - "rasters_dir" (str): Directory to store raster data.

    Optional Parameters:
    - source (str): Source identifier for the data (default is "ms").
    - verbose (bool): If True, display detailed logs; otherwise, suppress output (default is False).

    Notes:
    This function uses leafmap and subprocess libraries for downloading MS buildings data,
    converting it to different formats, and storing it in specified directories.
    """

    if "ms_matches" in config:
        matches = config["ms_matches"]
    else:
        matches = _get_ms_country(config)
    iso_codes = config["iso_codes"]
    
    for iso_code in (pbar := data_utils._create_progress_bar(iso_codes)):
        pbar.set_description(f"Processing {iso_code}")
        out_dir = data_utils._makedir(os.path.join(config["vectors_dir"], f"{source}_buildings"))
        temp_dir = data_utils._makedir(os.path.join(out_dir, iso_code))
        country = matches[iso_code]
        
        out_file = str(os.path.join(out_dir, f"{iso_code}_{source}_EPSG4326.geojson"))
        if not os.path.exists(out_file):
            quiet = operator.not_(verbose)
            leafmap.download_ms_buildings(country, out_dir=temp_dir, merge_output=out_file, quiet=quiet)
    
        out_file_epsg3857 = str(os.path.join(out_dir, f"{iso_code}_{source}_EPSG3857.geojson"))
        tif_dir = data_utils._makedir(os.path.join(config["rasters_dir"], f"{source}_buildings"))
        tif_file = str(os.path.join(tif_dir, f"{iso_code}_{source}.tif"))
    
        if (not os.path.exists(out_file_epsg3857)) and (not os.path.exists(tif_file)):
            command1 = "ogr2ogr -s_srs EPSG:4326 -t_srs EPSG:3857 {} {}".format(
                out_file_epsg3857,
                out_file
            )
            command2 = "gdal_rasterize -burn 255 -tr 10 10 -a_nodata 0 -at -l {} {} {}".format(
                f'{iso_code}_{source}_EPSG4326',
                out_file_epsg3857,
                tif_file
            )
            subprocess.Popen(f"{command1} && {command2}", shell=True)


def download_ghsl(config):
    ghsl_folder = os.path.join(config["rasters_dir"], "ghsl")
    data_utils._makedir(ghsl_folder)
    
    ghsl_path = os.path.join(ghsl_folder, config["ghsl_file"])
    ghsl_zip = os.path.join(ghsl_folder, "ghsl.zip")
    if not os.path.exists(ghsl_path):
        command1 = f"wget {config['ghsl_url']} -O {ghsl_zip}"
        command3 = f"unzip {ghsl_zip} -d {ghsl_folder}"
        subprocess.Popen(f"{command1} && {command2}", shell=True)

    
