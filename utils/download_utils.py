import os
import duckdb
import geojson
import overpass
import data_utils
import logging

from tqdm import tqdm
import pandas as pd
import geopandas as gpd
from country_bounding_boxes import country_subunits_by_iso_code

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
logging.basicConfig(level=logging.INFO)


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


def download_overture(config, category, exclude="school", source="overture"):
    """
    Downloads Overture Map points of interest (POIs) based on a list of ISO codes.

    Args:
    - config (dict): Configuration settings.
    - category (str): Type of data category ('school' or 'non_school').
    - exclude (str or None, optional): Keywords to exclude ('school' or 'non_school'). 
      Defaults to 'school'.
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

    if exclude is not None:
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
