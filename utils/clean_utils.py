import re
import os
import geojson
import itertools
import logging

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
from rasterio.mask import mask
import translators as ts

import networkx as nx
from rapidfuzz import fuzz
from tqdm import tqdm

import data_utils
import warnings

SEED = 42
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
logging.basicConfig(level=logging.INFO)
  

def _sample_points(
    config, 
    iso_code, 
    buffer_size,
    spacing,
    sname="clean"
):
    """
    Generates additional negative data points based on given configurations and spatial parameters.

    Args:
    - config (dict): Configuration settings.
    - iso_code (str): ISO code for a specific location.
    - buffer_size (float): Buffer size for points.
    - spacing (float): Spacing between points.
    - sname (str, optional): Name identifier (default is "clean").

    Returns:
    - GeoDataFrame: GeoDataFrame containing generated negative samples.
    """
    # Get current working directory
    cwd = os.path.dirname(os.getcwd())
    points = data_utils.generate_samples(config, iso_code, buffer_size, spacing)

    # Read positive data and perform buffer operation on geometries
    filename = f"{iso_code}_{sname}.geojson"
    vector_dir = os.path.join(cwd, config["vectors_dir"])
    pos_file = os.path.join(vector_dir, config["pos_class"], sname, filename)
    pos_df = gpd.read_file(pos_file).to_crs("EPSG:3857")
    
    pos_df["geometry"] = pos_df["geometry"].buffer(buffer_size, cap_style=3)
    points["geometry"] = points["geometry"].buffer(buffer_size, cap_style=3)

     # Identify intersecting points and remove them
    points["index"] = points.index
    intersecting = pos_df.sjoin(points, how="inner")["index"]
    points = points[~points["index"].isin(intersecting)]
    points["geometry"] = points["geometry"].centroid

    # Sample points from the microsoft raster
    points = points.to_crs("EPSG:3857")
    ms_coord_list = [(x, y) for x, y in zip(points["geometry"].x, points["geometry"].y)]
    
    points = points.to_crs("ESRI:54009")
    ghsl_coord_list = [(x, y) for x, y in zip(points["geometry"].x, points["geometry"].y)]
    
    raster_dir = os.path.join(cwd, config["rasters_dir"])
    ms_path = os.path.join(raster_dir, "ms_buildings", f"{iso_code}_ms.tif")

    # Filter points with pixel value greater than 0 and convert back to EPSG:4326
    if os.path.exists(ms_path):
        with rio.open(ms_path) as src:
            points['ms_val'] = [x[0] for x in src.sample(ms_coord_list)]

    ghsl_path = os.path.join(raster_dir, "ghsl", config["ghsl_built_c_file"])
    with rio.open(ghsl_path) as src:
        col_val = 'ghsl_val'  if 'ms_val' in points.columns else 'pixel_val'
        points[col_val]  = [x[0] for x in src.sample(ghsl_coord_list)]

    if 'ms_val' in points.columns:
        points['pixel_val'] = points[['ms_val', 'ghsl_val']].max(axis=1)
    points = points[points['pixel_val'] > 0]
        
    points = points.to_crs("EPSG:4326")
    return points


def _augment_negative_samples(config, sname="clean"):
    """
    Augments negative data points by generating additional points and combines datasets 
    based on provided configurations.

    Args:
    - config (dict): Configuration settings.
    - category (str, optional): Category of data (default is "non_school").
    - name (str, optional): Name identifier (default is "clean").

    Returns:
    - None: The function saves the combined dataset as a GeoJSON file.

    Raises:
    - FileNotFoundError: If the specified file or directory does not exist.
    - Exception: If there is an issue during data processing or concatenation.
    """
    
    cwd = os.path.dirname(os.getcwd())

    data = []
    counts = data_utils.get_counts(config, column='iso')

    items = list(counts.index[::-1]) 
    for iso_code in (pbar := data_utils._create_progress_bar(items)):
        pbar.set_description(f"Processing {iso_code}")
        subdata = counts[counts.index == iso_code]
        
        neg_file = os.path.join(
            cwd, 
            config["vectors_dir"], 
            config["neg_class"], 
            sname, 
            f"{iso_code}_{sname}.geojson"
        )
        negatives = gpd.read_file(neg_file)

        n_pos = subdata[config["pos_class"]].iloc[0]
        n_neg = subdata[config["neg_class"]].iloc[0]
        
        if n_pos > n_neg:
            buffer_size = config["object_proximity"]/2
            spacing = config["sample_spacing"]
            points = _sample_points(config, iso_code, buffer_size, spacing=spacing, sname=sname)
            
            points = data_utils._prepare_data(
                config=config,
                data=points,
                iso_code=iso_code,
                category=config["neg_class"],
                source='AUG',
                columns=config["columns"]
            )
            
            if "clean" in negatives.columns:
                points["clean"] = 0
            if "validated" in negatives.columns:
                points["validated"] = 0
                
            logging.info(f"{iso_code} {points.shape} {n_pos - n_neg}")
            points = points.sample((n_pos - n_neg), random_state=SEED)
            
            negatives = data_utils._concat_data(
                [points, negatives], 
                neg_file,
                verbose=False
            )
        data.append(negatives)

    # Save combined dataset
    vector_dir = os.path.join(cwd, config["vectors_dir"])
    out_file = os.path.join(vector_dir, config["neg_class"], f"{sname}.geojson")
    data = data_utils._concat_data(data, out_file)
    
    return data


def _filter_keywords(data, exclude, column="name"):
    """
    Filters out rows from data based on keyword exclusions in a specified column.

    Args:
    - data (DataFrame): DataFrame containing the specified column to filter.
    - exclude (list): List of keywords to exclude.
    - column (str, optional): Name of the column to filter. Defaults to "name".

    Returns:
    - DataFrame: Filtered DataFrame after excluding rows containing specified keywords.
    """
   
    exclude = [f"\\b{x.upper()}\\b" for x in exclude]
    data = data[
        ~data[column]
        .str.upper()
        .str.contains(r"|".join(exclude), case=False, na=False)
    ]
    return data


def _filter_pois_within_proximity(data, proximity, priority):
    """
    Filters points of interest (POIs) within a specified proximity or distance.

    Args:
    - data (DataFrame): DataFrame containing POI information.
    - proximity (float): Proximity for filtering POIs in meters.
    - priority (list): Priority for filtering duplicates.

    Returns:
    - DataFrame: Processed DataFrame with filtered POIs within the specified distance.
    """

    data = data_utils._connect_components(data, buffer_size=proximity/2)
    data = data_utils._drop_duplicates(data, priority=priority)
    return data


def _filter_uninhabited_locations(data, config, pbar):
    """
    Filters uninhabited locations based on buffer size (in meters).

    Args:
    - config (dict): Configuration settings.
    - data (DataFrame): Pandas DataFrame containing location data.
    - buffer_size (float): Buffer size (in meters) around the points.
    - source (str, optional): Name of the source. Defaults to "ms".
    - pbar (tqdm, optional): Progress bar. Defaults to None.

    Returns:
    - DataFrame: Filtered DataFrame containing inhabited locations.
    """
    # Get current working directory
    cwd = os.path.dirname(os.getcwd())
    data = data.reset_index(drop=True)
    iso_code = data.iso.values[0]
    buffer_size = config["filter_buffer_size"]

    # Generate file paths based on the iso_code and source
    raster_dir = os.path.join(cwd, config["rasters_dir"])
    src_path = os.path.join(raster_dir, "ms_buildings", f"{iso_code}_ms.tif")
    ghsl_path = os.path.join(raster_dir, "ghsl", config["ghsl_built_c_file"])
    if not os.path.exists(src_path):
        src_path = ghsl_path

    # Loop through each index in the DataFrame
    pixel_sums = []
    for index in range(len(data)):
        description = f"Processing {iso_code} {index}/{len(data)}"
        pbar.set_description(description)

        # Extract a single row from the DataFrame
        subdata = data.iloc[[index]]
        subdata = subdata.to_crs("EPSG:3857")
        subdata["geometry"] = subdata["geometry"].buffer(buffer_size, cap_style=3)

        # Mask the raster data with the buffered geometry
        image = []
        pixel_sum = 0
        with rio.open(src_path) as src:
            try:
                geometry = [subdata.iloc[0]["geometry"]]
                image, transform = rio.mask.mask(src, geometry, crop=True)
                image[image == 255] = 1
                pixel_sum = np.sum(image)
            except:
                pass

        # If no pixels found and source is not GHSL, attempt with GHSL data
        if (pixel_sum == 0) and (src_path != ghsl_path):
            with rio.open(ghsl_path) as src:
                subdata = subdata.to_crs("ESRI:54009")
                geometry = [subdata.iloc[0]["geometry"]]
                image, transform = rio.mask.mask(src, geometry, crop=True)
                image[image == 255] = 0 # no pixel value
                pixel_sum = np.sum(image)

        # Appending the pixel sum to the list
        pixel_sums.append(pixel_sum) 

    # Filter data based on pixel sums and updating DataFrame accordingly
    data["sum"] = pixel_sums
    data = data[data["sum"] > 0]
    data = data.reset_index(drop=True)
    return data


def _filter_pois_within_object_proximity(config, proximity, sname="clean"):
    """
    Filters points of interest (POIs) located within proximity of positive objects (e.g. schools).

    Args:
    - config (dict): Configuration settings.
    - proximity (float): Max proximity of objects (in meters) to be grouped.
    - sname (str, optional): Name of the dataset. Defaults to "clean".

    Returns:
    - GeoDataFrame: Processed and filtered GeoDataFrame containing non-school POIs.
    """
    
    # Get the current working directory
    cwd = os.path.dirname(os.getcwd())
    data_dir = config["vectors_dir"]

    # Set up directories and configurations
    neg_dir = os.path.join(data_dir, config["neg_class"])
    negatives = data_utils.read_data(neg_dir, exclude=[f"{sname}.geojson"])

    data = []
    # Iterate over ISO codes to process data
    for iso_code in (pbar := data_utils._create_progress_bar(config["iso_codes"])):
        pbar.set_description(f"Processing {iso_code}")
        
        filename = f"{iso_code}_{sname}.geojson"
        pos_file = os.path.join(cwd, data_dir, config["pos_class"], sname, filename)
        positives = gpd.read_file(pos_file)
        if "clean" in positives.columns:
            positives = positives[positives["clean"] == 0]
        if "validated" in positives.columns:
            positives = positives[positives["validated"] == 0]

        # Filter non-school POIs within the buffer of school locations
        pos_sub = positives[positives["iso"] == iso_code]
        neg_sub = negatives[negatives["iso"] == iso_code]

        # Convert school and non-school data CRS to EPSG:3857
        neg_temp = data_utils._convert_crs(neg_sub, target_crs="EPSG:3857")
        neg_temp["geometry"] = neg_temp["geometry"].buffer(proximity/2, cap_style=3)
        neg_temp["index"] = neg_sub.index
        pos_temp = data_utils._convert_crs(pos_sub, target_crs="EPSG:3857")
        pos_temp["geometry"] = pos_temp["geometry"].buffer(proximity/2, cap_style=3)

        # Filter out non-school POIs that intersect with buffered school locations
        intersecting = pos_temp.sjoin(neg_temp, how="inner")["index"]
        neg_sub = neg_sub[~neg_temp["index"].isin(intersecting)]

        # Save filtered country-level dataset
        subdata = neg_sub[config["columns"]]
        data.append(subdata)

    # Combine and save datasets
    data = data_utils._concat_data(data)
    return data


def _filter_pois_with_matching_names(data, proximity, threshold, priority):
    """
    Filters points of interest (POIs) with matching names within a certain buffer size.

    Args:
    - data (DataFrame): DataFrame containing POI information.
    - proximity (float): Max proximity of objects (in meters) for matching names.
    - threshold (int): Threshold score for similarity of names.
    - priority (list): Priority for filtering duplicates.

    Returns:
    - DataFrame: Processed DataFrame with filtered POIs based on matching names.
    """

    # Get connected components within a given buffer size and get groups with size > 1
    data = data_utils._connect_components(data, buffer_size=proximity/2)
    group_count = data.group.value_counts()
    groups = group_count[group_count > 1].index

    uid_network = []
    for index in range(len(groups)):
        # Get pairwise combination of names within a group
        subdata = data[data.group == groups[index]][["UID", "name"]]
        subdata = list(subdata.itertuples(index=False, name=None))
        combs = itertools.combinations(subdata, 2)

        # Compute rapidfuzz partial ratio score
        uid_edge_list = []
        for comb in combs:
            # Compute the partial ratio score between cleaned names
            score = fuzz.partial_ratio(
                data_utils._clean_text(comb[0][1]), 
                data_utils._clean_text(comb[1][1])
            )
            uid_edge_list.append(
                (comb[0][0], comb[0][1], comb[1][0], comb[1][1], score)
            )
        columns = ["source", "name_1", "target", "name_2", "score"]
        uid_edge_list = pd.DataFrame(uid_edge_list, columns=columns)
        uid_network.append(uid_edge_list)

    # Generate graph and get connected components
    if len(uid_network) > 0:
        uid_network = pd.concat(uid_network)
        uid_network = uid_network[uid_network.score >= threshold]

        columns = ["source", "target", "score"]
        graph = nx.from_pandas_edgelist(uid_network[columns])
        connected_components = nx.connected_components(graph)
        groups = {
            num: index
            for index, group in enumerate(connected_components, start=1)
            for num in group
        }

        if len(groups) > 0:
            # Assign groups to data points based on connected components
            data["group"] = np.nan
            for uid, value in groups.items():
                data.loc[data["UID"] == uid, "group"] = value
            max_group = int(data["group"].max()) + 1
            fillna = list(range(max_group, len(data) + max_group))
            data["group"] = data.apply(
                lambda x: x["group"]
                if not np.isnan(x["group"])
                else fillna[int(x.name)],
                axis=1,
            )
            data = data_utils._drop_duplicates(data, priority)

    return data


def clean_data(config, category, name="clean", source="ms", id="UID"):
    """
    Process and clean spatial data based on specified categories.

    Parameters:
    - config (dict): A dictionary containing configuration settings.
        - "vectors_dir" (str): Directory to store vector data.
        - "iso_codes" (list): List of ISO country codes.
        - "object_proximity" (float): Max proximity (in meters) to positive objects (e.g. schools)

    Optional Parameters:
    - name (str): Name identifier for the cleaned data (default is "clean").

    Returns:
    - data (geopandas.GeoDataFrame): Cleaned spatial data in GeoDataFrame format.

    Notes:
    This function cleans spatial data based on specified categories such as schools and non-schools.
    It involves data filtering, removal of specific objects based on proximity, keyword exclusion,
    and saving the cleaned data as GeoJSON files in the specified directory.
    """

    def _get_condition(data, name, id, ids, shape_name):
        return (
            (data[name] == 0) 
            & (~data[id].isin(ids))
            & (data["shapeName"] == shape_name)
        )
    
    # Define the output directory for processed data based on the category and name
    out_dir = data_utils._makedir(os.path.join(config["vectors_dir"], category, name))
    
    if category == config["pos_class"]:
        data_dir = os.path.join(config["vectors_dir"], category)
        data = data_utils.read_data(data_dir, exclude=[f"{name}.geojson"])
            
    # For negative samples, remove POIs within the positive object vicinity
    elif category == config["neg_class"]:
        data = _filter_pois_within_object_proximity(config, proximity=config["object_proximity"])

    out_data = []
    for iso_code in (pbar := data_utils._create_progress_bar(config["iso_codes"])):
        pbar.set_description(f"Processing {iso_code}")
        out_subfile = os.path.join(out_dir, f"{iso_code}_{name}.geojson")

        if not os.path.exists(out_subfile):
            # Join the dataset with ADM1 geoboundaries
            subdata = data[data["iso"] == iso_code].reset_index(drop=True)
            geoboundaries = data_utils._get_geoboundaries(config, iso_code, adm_level="ADM1")
            geoboundaries = geoboundaries[["shapeName", "geometry"]].dropna(subset=["shapeName"])
            subdata = subdata.sjoin(geoboundaries, how="left", predicate="within")
            subdata[name], subdata[source] = 0, 0
    
            # Split the data into smaller admin boundaries for scalability
            for shape_name in subdata.shapeName.unique():
                pbar.set_description(f"Processing {iso_code} {shape_name}")
                subsubdata = subdata[subdata["shapeName"] == shape_name]
                subsubdata = subsubdata[config["columns"]].reset_index(drop=True)
                if len(subsubdata) == 0:
                    continue

                # Remove objects containing certain keywords
                if category == config["pos_class"]:
                    subsubdata = _filter_keywords(
                        subsubdata, exclude=config["exclude"]
                    )[config["columns"]]
                    ids = subsubdata[id].values
                    condition = _get_condition(subdata, name, id, ids, shape_name)
                    subdata.loc[condition, name] = 1
    
                # Remove POIs within proximity of each other
                subsubdata = _filter_pois_within_proximity(
                    subsubdata,
                    proximity=config["proximity"],
                    priority=config["priority"],
                )[config["columns"]]
                ids = subsubdata[id].values
                condition = _get_condition(subdata, name, id, ids, shape_name)
                subdata.loc[condition, name] = 2
    
                # Remove POIs with matching names within proximity of each other
                subsubdata = _filter_pois_with_matching_names(
                    subsubdata,
                    priority=config["priority"],
                    threshold=config["name_match_threshold"],
                    proximity=config["name_match_proximity"],
                )
                ids = subsubdata[id].values
                condition = _get_condition(subdata, name, id, ids, shape_name)
                subdata.loc[condition, name] = 3

                # Filter uninhabited locations based on specified buffer size
                subsubdata = _filter_uninhabited_locations(
                    subsubdata,
                    config,
                    pbar=pbar
                )[config["columns"]]
                ids = subsubdata[id].values
                condition = _get_condition(subdata, name, id, ids, shape_name)
                subdata.loc[condition, name] = 4
                
            # Save cleaned file as a GeoJSON
            subdata = subdata[config["columns"]+[name]].reset_index(drop=True)
            out_subdata = data_utils._concat_data([subdata], out_file=out_subfile)
    
        # Read and store the cleaned data
        out_subdata = gpd.read_file(out_subfile).reset_index(drop=True)
        out_subdata.to_file(out_subfile, driver="GeoJSON")
        out_data.append(out_subdata)
    
    # Save combined dataset
    out_file = os.path.join(os.path.dirname(out_dir), f"{name}.geojson")
    data = data_utils._concat_data(out_data, out_file)

    if category == config["neg_class"]:
        data = _augment_negative_samples(config, sname=name)
    return data
    