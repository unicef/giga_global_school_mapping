import os
import geojson
import pandas as pd
import geopandas as gpd
import data_utils

from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

import folium
import rasterio as rio
from rasterio.plot import show
from IPython.display import display
from scipy.sparse.csgraph import connected_components


def map_coords(filename, index, zoom_start=18, max_zoom=20):
    data = gpd.read_file(filename)
    coords = data.iloc[index].geometry.y, data.iloc[index].geometry.x
    map = folium.Map(location=coords, zoom_start=zoom_start, max_zoom=max_zoom)
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
        attr="Google",
        name="Google Satellite",
        overlay=True,
        control=True,
    ).add_to(map)
    folium.Marker(location=coords, fill_color="#43d9de", radius=8).add_to(map)
    display(map)


def inspect_images(
    filename,
    image_dir,
    iso,
    n_rows=8,
    n_cols=4,
    index=0,
    figsize=(15, 35),
    category="SCHOOL",
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
            f"Index: {idx}\n{item.UID}\n{item['name']}", fontdict={"fontsize": 9}
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
    data_config = data_utils._load_data_config()
    temp = data_utils._convert_to_crs(data, target_crs="EPSG:3857")
    geometry = temp["geometry"].buffer(buffer_size)
    overlap_matrix = geometry.apply(lambda x: geometry.overlaps(x)).values.astype(int)
    n, groups = connected_components(overlap_matrix, directed=False)
    data["group"] = groups

    # Prioritize by: OVERTURE, OSM, UNICEF
    data["temp_source"] = pd.Categorical(
        data["source"], categories=["OVERTURE", "OSM", "UNICEF"], ordered=True
    )
    data = data.sort_values("temp_source", ascending=True).drop_duplicates(["group"])
    data = data.reset_index(drop=True)
    columns = data_config["COLUMNS"]
    if "giga_id_school" in data.columns:
        columns = columns + ["giga_id_school"]
    data = data[columns]
    return data


def deduplicate_data(
    data_dir, out_dir, out_file="clean.geojson", iso_codes=None, buffer_size=30
):
    # Read files in data_dir --> should contain Overture, OSM, and UNICEF files
    data_dir = data_utils._makedir(data_dir)
    out_dir = data_utils._makedir(out_dir)
    files = next(os.walk(data_dir), (None, None, []))[2]
    files = [file for file in files if file != out_file]

    data = []
    for file in (pbar:= tqdm(files, total=len(files))):
        pbar.set_description(f"Reading {file}")
        filename = os.path.join(data_dir, file)
        subdata = gpd.read_file(filename)
        data.append(subdata)
    
    # Concatenate files in data_dir
    data = gpd.GeoDataFrame(pd.concat(data).copy(), crs="EPSG:4326")
    data = data.drop_duplicates("geometry", keep="first")
    if not iso_codes:
        iso_codes = data.iso.unique()

    out_data = []
    pbar = tqdm(enumerate(iso_codes), total=len(iso_codes))
    for _, iso_code in pbar:
        pbar.set_description(f"Processing {iso_code}")
        filename = f"{iso_code}_clean.geojson"
        out_subfile = os.path.join(out_dir, filename)
        
        if not os.path.exists(out_subfile):
            subdata = data[data["iso"] == iso_code].reset_index(drop=True)
            # Fetch geoboundaries with admin level 1
            columns = ["shapeName", "geometry"]
            geoboundaries = data_utils._get_geoboundaries(iso_code, adm_level="ADM1")[columns]
            geoboundaries = geoboundaries.dropna(subset=["shapeName"])
            subdata = subdata.sjoin(geoboundaries, how="left", predicate="within")

            # Split the data into smaller admin boundaries
            out_subdata = []
            for shape_name in subdata.shapeName.unique():
                pbar.set_description(f"Processing {iso_code} {shape_name}")
                subsubdata = subdata[subdata["shapeName"] == shape_name]
                subsubdata = subsubdata.drop(["index_right", "shapeName"], axis=1)
                subsubdata = subsubdata.reset_index(drop=True)
                if len(subsubdata) > 0:
                    subsubdata = _connect_components(subsubdata, buffer_size)
                    out_subdata.append(subsubdata)
            # Save cleaned file
            out_subdata = data_utils._concat_data(out_subdata, out_subfile, verbose=False)
        
        out_subdata = gpd.read_file(out_subfile).reset_index(drop=True)
        out_data.append(out_subdata)

    # Save combined dataset
    out_dir = os.path.dirname(out_dir)
    out_file = os.path.join(out_dir, out_file)
    data = data_utils._concat_data(out_data, out_file)
    return data