import re
import os
import uuid
import requests

import duckdb
import geojson
import overpass
import swifter

import numpy as np
import pandas as pd
import geopandas as gpd

import folium
import rasterio as rio
from rasterio.plot import show
import matplotlib.pyplot as plt
from IPython.display import display

from tqdm.notebook import tqdm
from pyproj import Proj, Transformer
from scipy.sparse.csgraph import connected_components
from country_bounding_boxes import country_subunits_by_iso_code

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


def _concat_data(data, out_file, verbose=True):
    data = pd.concat(data).reset_index(drop=True)
    data = gpd.GeoDataFrame(data, geometry=data["geometry"], crs="EPSG:4326")
    data.to_file(out_file, driver="GeoJSON")
    
    if verbose:
        print(f"Data dimensions: {data.shape}, CRS: {data.crs}")
        print(f"Generated {out_file}")
    
    return data


def _generate_uid(data, category):
    """Generates a unique id based on source, iso, and category. """
    
    data['index'] = data.index.to_series().apply(lambda x: str(x).zfill(8))
    data['category'] = category
    uids = data[['source', 'iso', 'category', 'index']].agg('-'.join, axis=1)
    data = data.drop(['index', 'category'], axis=1)
    data["UID"] = uids
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


def map_coords(filename, index, zoom_start=18, max_zoom=20):
    data = gpd.read_file(filename)
    coords = data.iloc[index].geometry.y, data.iloc[index].geometry.x
    map = folium.Map(
        location=coords, 
        zoom_start=zoom_start, 
        max_zoom=max_zoom
    )
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
        attr='Google',
        name='Google Satellite',
        overlay=True,
        control=True
    ).add_to(map)
    folium.Marker(
        location=coords, fill_color='#43d9de', radius=8
    ).add_to(map)
    display(map)


def inspect_images(
    filename,
    image_dir,
    iso,
    n_rows=8,
    n_cols=4,
    index=0,
    figsize=(15, 35),
    category="SCHOOL"
):
    data = gpd.read_file(filename)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    samples = data.iloc[index : index + (n_rows * n_cols)]
    row_index, col_index = 0, 0

    increment = 1
    for idx, item in samples.iterrows():
        class_dir = os.path.join(image_dir, f"{iso}/{category.lower()}")
        filepath = os.path.join(class_dir, f"{item.UID}.tiff")

        image = rio.open(filepath)
        show(image, ax=axes[row_index, col_index])
        axes[row_index, col_index].tick_params(
            left=False, bottom=False, labelleft=False, labelbottom=False
        )
        axes[row_index, col_index].set_axis_off()
        axes[row_index, col_index].set_title(
            f"Index: {idx}\n{item.UID}\n{item['name']}", 
            fontdict={"fontsize": 9}
        )

        col_index += 1
        if col_index >= n_cols:
            row_index += 1
            col_index = 0  
        if row_index >= n_rows:
            break


def _connect_components(data, buffer_size, optimize=False):
    # Dissolve overlapping geometries
    # based on: https://gis.stackexchange.com/a/271737
    data_config = _load_data_config()
    temp = _convert_to_crs(data, target_crs="EPSG:3857")
    geometry = temp['geometry'].buffer(buffer_size)
    overlap_matrix = geometry.apply(lambda x: geometry.overlaps(x)).values.astype(int)
    n, groups = connected_components(overlap_matrix, directed=False)
    data['group'] = groups
    
    # Prioritize by: OVERTURE, OSM, UNICEF
    data['temp_source'] = pd.Categorical(
        data['source'], 
        categories=['OVERTURE','OSM','UNICEF'], 
        ordered=True
    )
    data = data.sort_values('temp_source', ascending=True).drop_duplicates(['group'])
    data = data.reset_index(drop=True)
    columns = data_config["COLUMNS"]
    if 'giga_id_school' in data.columns:
        columns = columns + ['giga_id_school']
    data = data[columns]
    return data


def deduplicate_data(
    data_dir, 
    out_dir, 
    out_file="clean.geojson",
    iso_codes=None, 
    buffer_size=30
):
    data_dir = _makedir(data_dir)
    out_dir = _makedir(out_dir)
    files = next(os.walk(data_dir), (None, None, []))[2]
    files = [file for file in files if file != out_file]
        
    data = []
    for file in files:
        filename = os.path.join(data_dir, file)
        subdata = gpd.read_file(filename)
        data.append(subdata)
    
    data = gpd.GeoDataFrame(pd.concat(data).copy(), crs="EPSG:4326") 
    data = data.drop_duplicates('geometry', keep="first")
    if not iso_codes:
        iso_codes = data.iso.unique()
    
    out_data = []
    pbar1 = tqdm(enumerate(iso_codes), total=len(iso_codes))
    for iter, iso_code in pbar1:
        pbar1.set_description(f"Processing {iso_code}")
        subdata = data[data["iso"] == iso_code].reset_index(drop=True)
        
        # Fetch geoboundaries with admin level 1
        columns = ['shapeName', 'geometry']
        geoboundaries = _get_geoboundaries(iso_code, adm_level="ADM1")[columns]
        geoboundaries = geoboundaries.dropna(subset=['shapeName'])
        subdata = subdata.sjoin(geoboundaries, how="left", predicate="within")
        
        # Split the data into smaller admin boundaries
        out_subdata = []
        for shape_name in subdata.shapeName.unique():   
            pbar1.set_description(f"Processing {iso_code} {shape_name}")
            subsubdata = subdata[subdata["shapeName"] == shape_name]
            subsubdata = subsubdata.drop(["index_right", "shapeName"], axis=1)
            subsubdata = subsubdata.reset_index(drop=True)
            if len(subsubdata) > 0:
                subsubdata = _connect_components(subsubdata, buffer_size)    
                out_subdata.append(subsubdata)        
        
        # Save cleaned file
        filename = f"{iso_code}_clean.geojson"
        out_subfile = os.path.join(out_dir, filename)
        out_subdata = _concat_data(out_subdata, out_subfile, verbose=False)
        out_data.append(out_subdata)
    
    # Save combined dataset
    out_dir = os.path.dirname(out_dir)
    out_file = os.path.join(out_dir, out_file)
    data = _concat_data(out_data, out_file)
    return data


def _query_osm(iso_code, out_file, query):
    """Queries OSM for a given ISO code."""
    
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


def download_osm(iso_codes, out_dir, out_file="osm.geojson", category="SCHOOL"):
    """Downloads OSM POIs based on a list of ISO codes."""
    
    out_dir = _makedir(out_dir)
    data_config = _load_data_config()
    url = data_config["ISO_REGIONAL_CODES"]
    codes = pd.read_csv(url)

    keywords = data_config[category]
    query = "".join(
        [f"""
        node["{key}"~"^({"|".join(values)})"](area);
        way["{key}"~"^({"|".join(values)})"](area);
        rel["{key}"~"^({"|".join(values)})"](area);
        """
        for key, values in keywords.items()
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

        subdata = gpd.read_file(out_file).reset_index(drop=True)
        if (len(subdata) > 0) and ('name' in subdata.columns):   
            subdata["source"] = "OSM"
            subdata = subdata.drop_duplicates('geometry', keep="first")
            subdata = _get_iso_regions(subdata, iso_code)
            subdata = _generate_uid(subdata, category)
            subdata = subdata[data_config["COLUMNS"]]
            subdata.to_file(out_file, driver="GeoJSON")
            data.append(subdata)

    # Combine data and save to file
    out_dir = os.path.dirname(out_dir)
    out_file = os.path.join(out_dir, out_file)
    data = _concat_data(data, out_file)
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


def download_overture(
    iso_codes, 
    out_dir, 
    out_file="overture.geojson", 
    category="SCHOOL",
    exclude=None
):
    """Downloads Overture Map POIs based on a list of ISO codes."""

    # Fetch school keywords and generate keywords query
    out_dir = _makedir(out_dir)
    data_config = _load_data_config()
    keywords = data_config[category]
        
    query = " or ".join([
        f"""UPPER(names) LIKE '%{keyword.replace('_', ' ').upper()}%'
        """
        for key, values in keywords.items()
        for keyword in values
    ])
    
    if exclude is not None:
        exclude_keywords = data_config[exclude]
        exclude_query = " and ".join([
            f"""UPPER(names) NOT LIKE '%{keyword.replace('_', ' ').upper()}%'
            """
            for key, values in exclude_keywords.items()
            for keyword in values
        ]) 
        query = f""" ({query}) and ({exclude_query})"""

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
        subdata = gpd.read_file(out_file).reset_index(drop=True)
        if len(subdata) > 0:
            if 'names' in subdata.columns:
                geoboundary = _get_geoboundaries(iso_code)
                subdata["name"] = subdata["names"].apply(lambda x: x["common"][0]["value"])
                subdata = gpd.sjoin(subdata, geoboundary, predicate="within")
            
            subdata["source"] = "OVERTURE"
            subdata = subdata.drop_duplicates('geometry', keep="first")
            subdata = _get_iso_regions(subdata, iso_code)
            subdata = _generate_uid(subdata, category)
            subdata = subdata[data_config["COLUMNS"]]
            subdata.to_file(out_file, driver="GeoJSON")
            data.append(subdata)

    # Combine datasets
    out_dir = os.path.dirname(out_dir)
    out_file = os.path.join(out_dir, out_file)
    data = _concat_data(data, out_file)
    return data


def load_files(data_dir, out_file, category='SCHOOL'):
    """Combines the UNICEF/Giga datasets into a single CSV file."""

    # Load school data files
    data_dir = _makedir(data_dir)
    data_config = _load_data_config()
    files = next(os.walk(data_dir), (None, None, []))[2]
    print(f"Number of CSV files: {len(files)}")

    data = []
    pbar = tqdm(enumerate(files), total=len(files))
    for iter, file in pbar:
        # Read lat-lon file
        filename = os.path.join(data_dir, file)
        subdata = pd.read_csv(filename, low_memory=False).reset_index(drop=True)

        # Get iso, country name, region, and sub-region
        iso_code = file.split("_")[0]  # Get ISO code
        pbar.set_description(f"Processing {iso_code}")

        subdata['source'] = 'UNICEF'
        subdata["geometry"] = gpd.GeoSeries.from_xy(subdata["lon"], subdata["lat"])
        subdata = subdata.drop_duplicates('geometry', keep="first")
        subdata = _get_iso_regions(subdata, iso_code)
        subdata = _generate_uid(subdata, category)
        subdata = subdata[data_config["COLUMNS"] + ["giga_id_school"]]
        
        # Append to data
        data.append(subdata)

    # Combine dataset
    out_dir = os.path.dirname(data_dir)
    out_file = os.path.join(out_dir, out_file)
    data = _concat_data(data, out_file)
    return data
