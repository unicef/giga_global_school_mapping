import os
import uuid
import requests
from tqdm.notebook import tqdm

import duckdb
import geojson
import overpass
import pandas as pd
import geopandas as gpd
from utils import config_utils

from country_bounding_boxes import country_subunits_by_iso_code
from pyproj import Proj, Transformer


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


def _convert_to_crs(data, src_crs="EPSG:4326", target_crs="EPSG:3857"):
    """Converts GeoDataFrame to CRS."""
    
    # Convert lat long to the target CRS
    if ('lat' not in data.columns) or ('lon' not in data.columns):
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


def _get_geoboundaries(iso_code, out_dir="data/geoboundary", adm_level="ADM0"):
    """Fetches the geoboundary given an ISO code"""
    
    # Query geoBoundaries
    out_dir = _makedir(out_dir)
    data_config = _load_data_config()
    url = f"{data_config['GEOBOUNDARIES_URL']}{iso_code}/{adm_level}/"
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


def _query_osm(iso_code, out_file, query):
    """Queries OSM for a given ISO code."""
    
    api = overpass.API(timeout=1500)
    data = api.get(
        f"""
        area["ISO3166-1"="{iso_code}"][admin_level=2];
        ({query});
        out center;
    """,
        verbosity="geom",
    )
    with open(out_file, "w") as file:
        geojson.dump(data, file)

    data = gpd.read_file(out_file)
    return data


def download_osm(iso_codes, out_dir, category="SCHOOL"):
    """Downloads OSM POIs based on a list of ISO codes."""
    
    out_dir = _makedir(out_dir)
    data_config = _load_data_config()
    url = data_config["ISO_REGIONAL_CODES"]
    codes = pd.read_csv(url)

    keywords = data_config[category]
    query = "".join(
        [f"""
        node["{key}"="{keyword}"](area);
        way["{key}"="{keyword}"](area);
        rel["{key}"="{keyword}"](area);
        """
            for key, values in keywords.items()
            for keyword in values
        ]
    )

    data = []
    pbar = tqdm(enumerate(iso_codes), total=len(iso_codes))
    for iter, iso_code in pbar:
        pbar.set_description(f"Processing {iso_code}")
        filename = f"{iso_code}_osm.geojson"
        out_file = os.path.join(out_dir, filename)

        if not os.path.exists(out_file):
            alpha_2 = codes.query(f"`alpha-3` == '{iso_code}'")["alpha-2"].values[0]
            _query_osm(alpha_2, out_file, query)

        subdata = gpd.read_file(out_file)
        if len(subdata) > 0:
            subdata["iso"] = iso_code
            subdata["source"] = "OSM"
            subdata = subdata[['iso', 'source', 'name', 'geometry']]
            subdata.to_file(out_file, driver="GeoJSON")
            data.append(subdata)

    # Combine data and save to file
    data = pd.concat(data)
    data = gpd.GeoDataFrame(data).reset_index(drop=True)
    #data["lon"], data["lat"] = data.geometry.x, data.geometry.y
    uids = data.index.to_series().apply(lambda x: f"osm-{x}")
    data.insert(loc=0, column='UID', value=uids)
    print(f"Data dimensions: {data.shape} CRS: {data.crs}")

    # Save dataset
    out_dir = os.path.dirname(out_dir)
    out_file = os.path.join(out_dir, "osm.geojson")
    data.to_file(out_file, driver="GeoJSON")
    print(f"Generated {out_file}")
    return data


def _query_overture(iso_code, out_file, query):
    """Queries Overture Map for a given ISO code."""
    
    # Connect to DuckDB
    db = duckdb.connect()
    db.execute("INSTALL spatial")
    db.execute("INSTALL httpfs")
    db.execute("""
        LOAD spatial;
        LOAD httpfs;
        SET s3_region='us-west-2';
    """
    )

    # Fetch country bounding box and execute Overture query
    data_config = _load_data_config()
    url = data_config["OVERTURE_URL"]
    bbox = [c.bbox for c in country_subunits_by_iso_code(iso_code)][0]
    db.execute(f"""
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
    """).fetchall()

    # Intersect data with the country's geoboundaries
    data = gpd.read_file(out_file)
    return data


def download_overture(iso_codes, out_dir, category="SCHOOL"):
    """Downloads Overture Map POIs based on a list of ISO codes."""

    # Fetch school keywords and generate keywords query
    out_dir = _makedir(out_dir)
    data_config = _load_data_config()
    keywords = data_config[category]
    query = " or ".join(
        [
            f"UPPER(names) LIKE '%{keyword.upper()}%'"
            for key, values in keywords.items()
            for keyword in values
        ]
    )

    data = []
    pbar = tqdm(enumerate(iso_codes), total=len(iso_codes))
    for iter, iso_code in pbar:
        pbar.set_description(f"Processing {iso_code}")
        filename = f"{iso_code}_overture.geojson"
        out_file = os.path.join(out_dir, filename)

        # Fetch Overture data
        if not os.path.exists(out_file):
            _query_overture(iso_code, out_file, query)

        # Post-process data
        subdata = gpd.read_file(out_file)
        if len(subdata) > 0:
            subdata["iso"] = iso_code
            subdata["source"] = "OVERTURE"
            
            if 'names' in subdata.columns:
                geoboundary = _get_geoboundaries(iso_code)
                subdata["name"] = subdata["names"].apply(lambda x: x["common"][0]["value"])
                subdata = gpd.sjoin(subdata, geoboundary, predicate="within")
            
            subdata = subdata[['iso', 'source', 'name', 'geometry']]
            subdata.to_file(out_file, driver="GeoJSON")
            data.append(subdata)

    # Combine datasets
    data = pd.concat(data)
    data = gpd.GeoDataFrame(data).reset_index(drop=True)
    #data["lon"], data["lat"] = data.geometry.x, data.geometry.y
    uids = data.index.to_series().apply(lambda x: f"overture-{x}")
    data.insert(loc=0, column='UID', value=uids)
    print(f"Data dimensions: {data.shape} CRS: {data.crs}")

    # Save dataset
    out_dir = os.path.dirname(out_dir)
    out_file = os.path.join(out_dir, "overture.geojson")
    data.to_file(out_file, driver="GeoJSON")
    print(f"Generated {out_file}")
    return data


def load_files(data_dir, out_file):
    """Combines the UNICEF/Giga datasets into a single CSV file."""

    # Load school data files
    data_dir = _makedir(data_dir)
    files = next(os.walk(data_dir), (None, None, []))[2]
    print(f"Number of CSV files: {len(files)}")

    # Load ISO codes of countries and regions/subregional codes
    data_config = _load_data_config()
    url = data_config["ISO_REGIONAL_CODES"]
    codes = pd.read_csv(url)

    data = []
    pbar = tqdm(enumerate(files), total=len(files))
    for iter, file in pbar:
        # Read lat-lon file
        filename = os.path.join(data_dir, file)
        subdata = pd.read_csv(filename, low_memory=False).reset_index(drop=True)

        # Get iso, country name, region, and sub-region
        iso_code = file.split("_")[0]  # Get ISO code
        subcode = codes.query(f"`alpha-3` == '{iso_code}'")
        pbar.set_description(f"Processing {iso_code}")

        # Add iso, country, region, and subregion
        columns = {
            "source": "UNICEF", 
            "iso": iso_code,
            "country": subcode["name"].values[0],
            "subregion": subcode["sub-region"].values[0], 
            "region": subcode["region"].values[0]
        }
        for key, value in columns.items():
            subdata[key] = value

        # Append to data
        columns = [key for key, value in columns.items()] + ['geometry']
        subdata["geometry"] = gpd.GeoSeries.from_xy(subdata["lon"], subdata["lat"])
        subdata = subdata[columns]
        data.append(subdata)

    # Combine dataset
    data = pd.concat(data).reset_index(drop=True)
    uids = data.index.to_series().apply(lambda x: f"unicef-{x}")
    data.insert(loc=0, column='UID', value=uids)
    data = gpd.GeoDataFrame(data, geometry=data["geometry"], crs="EPSG:4326")
    print(f"Data dimensions: {data.shape}, CRS: {data.crs}")

    # Save dataset
    out_dir = os.path.dirname(data_dir)
    out_file = os.path.join(out_dir, out_file)
    data.to_file(out_file, driver="GeoJSON")
    print(f"Generated {out_file}")
    return data
