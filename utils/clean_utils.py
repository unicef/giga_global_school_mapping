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
    

def _generate_additional_negative_samples(
    config, 
    iso_code, 
    buffer_size,
    spacing,
    sname="clean",
    source="ms",
    categories=["school", "non_school"]
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

     # Get geographical boundaries for the ISO code at the specified administrative level
    bounds = data_utils._get_geoboundaries(config, iso_code, adm_level="ADM0")
    bounds = bounds.to_crs("EPSG:3857") # Convert to EPSG:3857

    # Calculate bounds for generating XY coordinates
    xmin, ymin, xmax, ymax = bounds.total_bounds 
    xcoords = [c for c in np.arange(xmin, xmax, spacing)]
    ycoords = [c for c in np.arange(ymin, ymax, spacing)] 
    
    # Create all combinations of XY coordinates
    coordinate_pairs = np.array(np.meshgrid(xcoords, ycoords)).T.reshape(-1, 2) 
    # Create a list of Shapely points
    geometries = gpd.points_from_xy(coordinate_pairs[:,0], coordinate_pairs[:,1]) 

    # Create a GeoDataFrame of points and perform spatial join with bounds
    points = gpd.GeoDataFrame(geometry=geometries, crs=bounds.crs).reset_index(drop=True)
    points = gpd.sjoin(points, bounds, predicate='within')
    points = points.drop(['index_right'], axis=1)

    # Read positive data and perform buffer operation on geometries
    filename = f"{iso_code}_{sname}.geojson"
    pos_file = os.path.join(cwd, config["vectors_dir"], categories[0], sname, filename)
    pos_df = gpd.read_file(pos_file).to_crs("EPSG:3857")
    pos_df["geometry"] = pos_df["geometry"].buffer(buffer_size, cap_style=3)
    points["geometry"] = points["geometry"].buffer(buffer_size, cap_style=3)

     # Identify intersecting points and remove them
    points["index"] = points.index
    intersecting = pos_df.sjoin(points, how="inner")["index"]
    points = points[~points["index"].isin(intersecting)]
    points["geometry"] = points["geometry"].centroid
    points = points.to_crs("EPSG:3857")

    # Sample points from the microsoft raster
    coord_list = [(x, y) for x, y in zip(points["geometry"].x, points["geometry"].y)]
    source_path = os.path.join(cwd, config["rasters_dir"], source, config[f"{source}_file"])

    point[source] = None
    if os.path.exists(source_path):
        with rio.open(source_path) as src:
            points[source] = [x[0] for x in src.sample(coord_list)]
    
        # Filter points with ms greater than 0 and convert back to EPSG:4326
        points = points[points[source] > 0]
    
    points = points.to_crs("EPSG:4326")
    return points


def _augment_negative_samples(config, name="augmented", sname="clean", source="ms", categories=["school", "non_school"]):
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
    logging.info(counts)

    items = list(counts.index[::-1])
    for iso_code in (pbar := data_utils._create_progress_bar(items)):
        pbar.set_description(f"Processing {iso_code}")
        subdata = counts[counts.index == iso_code]
        
        neg_file = os.path.join(
            cwd, 
            config["vectors_dir"], 
            categories[1], 
            sname, 
            f"{iso_code}_{sname}.geojson"
        )
        negatives = gpd.read_file(neg_file)

        if subdata[categories[1]].iloc[0] < subdata[categories[0]].iloc[0]:
            buffer_size = config["pos_object_proximity"]/2
            spacing = config["neg_sample_spacing"]
            points = _generate_additional_negative_samples(
                config, iso_code, buffer_size, spacing=spacing, categories=categories
            )
            points = data_utils._prepare_data(
                config=config,
                data=points,
                iso_code=iso_code,
                category=categories[1],
                source=source,
                columns=config["columns"]+[source]
            )
            size = subdata[pos].iloc[0] - subdata[neg].iloc[0]
            points = points.sample(size, random_state=SEED)
            out_dir = _makedir(
                os.path.join(
                    cwd, 
                    config["vectors_dir"], 
                    categories[1], 
                    name
                )
            )
            out_file = os.path.join(out_dir, f"{iso_code}_{name}.geojson")
            negatives = data_utils._concat_data(
                [points, negatives], 
                out_file,
                verbose=False
            )
        data.append(negatives)

    # Save combined dataset
    out_file = os.path.join(cwd, config["vectors_dir"], categories[1], f"{name}.geojson")
    data = data_utils._concat_data(data, out_file)
    return data


def _filter_uninhabited_locations(data, config, source="ms", pbar=None):
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

    # Reset index of the DataFrame
    data = data.reset_index(drop=True)

    # Extract iso_code from the DataFrame
    iso_code = data.iso.values[0]

    # Retrieve buffer size from the configuration dictionary
    buffer_size = config["filter_buffer_size"]

    # Generate file paths based on the iso_code and source
    src_path = os.path.join(cwd, config["rasters_dir"], f"{source}_buildings", f"{iso_code}_{source}.tif")
    ghsl_path = os.path.join(cwd, config["rasters_dir"], "ghsl", config["ghsl_file"])

    # If source path doesn't exist, fallback to GHSL path
    if not os.path.exists(src_path):
        src_path = ghsl_path

    # Initialize an empty list to store pixel sums
    ms_sum = []

    # Loop through each index in the DataFrame
    for index in range(len(data)):
        if pbar:
            # Update progress bar description if available
            description = f"Processing {iso_code} {index}/{len(data)}"
            pbar.set_description(description)

        # Extract a single row from the DataFrame
        subdata = data.iloc[[index]]

        # Convert subdata to EPSG:3857 coordinate reference system
        subdata = subdata.to_crs("EPSG:3857")

        # Create a buffer around the geometry points based on buffer_size
        subdata["geometry"] = subdata["geometry"].buffer(buffer_size, cap_style=3)

        # Mask the raster data with the buffered geometry
        image = []
        pixel_sum = 0
        with rio.open(src_path) as src:
            try:
                geometry = [subdata.iloc[0]["geometry"]]
                image, transform = rio.mask.mask(src, geometry, crop=True)
                image[image < 0] = 0
                image[image == 255] = 1
                pixel_sum = image.sum()
            except:
                pass            

        # If no pixels found and source is not GHSL, attempt with GHSL data
        if pixel_sum == 0 and src != ghsl_path:
            with rio.open(ghsl_path) as src:
                try:
                    geometry = [subdata.iloc[0]["geometry"]]
                    image, transform = rio.mask.mask(src, geometry, crop=True)
                    pixel_sum = image.sum()
                except:
                    pass

        # Appending the pixel sum to the list
        ms_sum.append(pixel_sum)        

    # Filter data based on pixel sums and updating DataFrame accordingly
    if len(ms_sum) > 0:
        data[source] = ms_sum
        data = data[data[source] > 0]
        data = data.reset_index(drop=True)
    else:
        data[source] = None

    # Return the filtered DataFrame
    return data


def _filter_pois_within_object_proximity(
    config, 
    proximity, 
    sname="clean", 
    name="filtered",
    categories=["school", "non_school"]
):
    """
    Filters points of interest (POIs) located within proximity of positive objects (e.g. schools).

    Args:
    - config (dict): Configuration settings.
    - proximity (float): Max proximity of objects (in meters) to be grouped.
    - sname (str, optional): Name of the dataset. Defaults to "clean".
    - name (str, optional): Name for the filtered output. Defaults to "filtered".

    Returns:
    - GeoDataFrame: Processed and filtered GeoDataFrame containing non-school POIs.
    """
    
    # Get the current working directory
    cwd = os.path.dirname(os.getcwd())

    # Set up directories and configurations
    data_dir = config["vectors_dir"]
    out_dir = os.path.join(data_dir, categories[1], name)
    out_dir = data_utils._makedir(out_dir)
    iso_codes = config["iso_codes"]

    # Read negative samples dataset
    neg_dir = os.path.join(data_dir, categories[1])
    exclude = [f"{sname}.geojson", f"{name}.geojson"]
    negatives = data_utils._read_data(neg_dir, exclude=exclude)

    data = []
    # Iterate over ISO codes to process data
    for iso_code in (pbar := data_utils._create_progress_bar(iso_codes)):
        filename = f"{iso_code}_{sname}.geojson"
        pos_file = os.path.join(cwd, data_dir, categories[0], sname, filename)
        positives = gpd.read_file(pos_file)

        pbar.set_description(f"Processing {iso_code}")
        out_subfile = os.path.join(out_dir, f"{iso_code}_{name}.geojson")

        if not os.path.exists(out_subfile):
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
            columns = config["columns"]
            neg_sub = neg_sub[columns]
            neg_sub.to_file(out_subfile, driver="GeoJSON")

        subdata = gpd.read_file(out_subfile).reset_index(drop=True)
        data.append(subdata)

    # Combine and save datasets
    filtered_file = os.path.join(cwd, neg_dir, f"{name}.geojson")
    data = data_utils._concat_data(data, out_file=filtered_file)
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


def clean_data(config, categories=["school", "non_school"],  name="clean"):
    """
    Process and clean spatial data based on specified categories.

    Parameters:
    - config (dict): A dictionary containing configuration settings.
        - "vectors_dir" (str): Directory to store vector data.
        - "iso_codes" (list): List of ISO country codes.
        - "pos_object_proximity" (float): Max proximity (in meters) to positive objects (e.g. schools)

    Optional Parameters:
    - categories (list): List of categories to process (default is ["school", "non_school"]).
    - name (str): Name identifier for the cleaned data (default is "clean").

    Returns:
    - data (geopandas.GeoDataFrame): Cleaned spatial data in GeoDataFrame format.

    Notes:
    This function cleans spatial data based on specified categories such as schools and non-schools.
    It involves data filtering, removal of specific objects based on proximity, keyword exclusion,
    and saving the cleaned data as GeoJSON files in the specified directory.
    """
    
    for category in categories:
        # Define the output directory for processed data based on the category and name
        out_dir = os.path.join(config["vectors_dir"], category, name)
        out_dir = data_utils._makedir(out_dir) # Create directory if it doesn't exist
        iso_codes = config["iso_codes"] # Fetch ISO codes from the config
    
        if category == categories[0]:
            data_dir = os.path.join(config["vectors_dir"], category)
            # Read data for the positive category, excluding the specified filename
            data = data_utils._read_data(data_dir, exclude=[f"{name}.geojson"])
            
        # For negative samples, remove POIs within the positive object vicinity
        elif category == categories[1]:
            data = _filter_pois_within_object_proximity(
                config, proximity=config["pos_object_proximity"], categories=categories
            )
    
        out_data = []
        for iso_code in (pbar := data_utils._create_progress_bar(iso_codes)):
            pbar.set_description(f"Processing {iso_code}")
            out_subfile = os.path.join(out_dir, f"{iso_code}_{name}.geojson")
    
            if not os.path.exists(out_subfile):
                # Join the dataset with ADM1 geoboundaries
                subdata = data[data["iso"] == iso_code].reset_index(drop=True)
                geoboundaries = data_utils._get_geoboundaries(config, iso_code, adm_level="ADM1")
                geoboundaries = geoboundaries[["shapeName", "geometry"]].dropna(subset=["shapeName"])
                subdata = subdata.sjoin(geoboundaries, how="left", predicate="within")
    
                # Split the data into smaller admin boundaries for scalability
                out_subdata = []
                for shape_name in subdata.shapeName.unique():
                    pbar.set_description(f"Processing {iso_code} {shape_name}")
                    subsubdata = subdata[subdata["shapeName"] == shape_name]
                    subsubdata = subsubdata.drop(["index_right", "shapeName"], axis=1)
                    subsubdata = subsubdata.reset_index(drop=True)
                    columns = config["columns"]
    
                    if len(subsubdata) > 0:
                        # Remove objects containing certain keywords
                        if category == categories[0]:
                            subsubdata = _filter_keywords(
                                subsubdata, exclude=config["exclude"]
                            )[columns]
    
                        # Remove POIs within proximity of each other
                        subsubdata = _filter_pois_within_proximity(
                            subsubdata,
                            proximity=config["proximity"],
                            priority=config["priority"],
                        )[columns]
    
                        # Remove POIs with matching names within proximity of each other
                        subsubdata = _filter_pois_with_matching_names(
                            subsubdata,
                            priority=config["priority"],
                            threshold=config["name_match_threshold"],
                            proximity=config["name_match_proximity"],
                        )[columns]
    
                        # Filter uninhabited locations based on specified buffer size
                        subsubdata = _filter_uninhabited_locations(
                            subsubdata,
                            config,
                            pbar=pbar
                        )
                        out_subdata.append(subsubdata)
    
                # Save cleaned file as a GeoJSON
                out_subdata = data_utils._concat_data(
                    out_subdata, 
                    out_file=out_subfile, 
                    verbose=False
                )
    
            # Read and store the cleaned data
            out_subdata = gpd.read_file(out_subfile).reset_index(drop=True)
            out_subdata.to_file(out_subfile, driver="GeoJSON")
            out_data.append(out_subdata)
    
        # Save combined dataset
        out_file = os.path.join(os.path.dirname(out_dir), f"{name}.geojson")
        data = data_utils._concat_data(out_data, out_file)
    
    #data = _augment_negative_samples(config, categories=categories)
    return data
    