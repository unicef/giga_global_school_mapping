import re
import os
import geojson
import itertools
import logging

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio

import networkx as nx
from rapidfuzz import fuzz
from tqdm import tqdm
from scipy.sparse.csgraph import connected_components

import data_utils
import gee_utils
import warnings

SEED = 42
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
logging.basicConfig(level=logging.INFO)


def _clean_text(text):
    """
    Cleans the text by removing all non-word characters and converting to uppercase.

    Args:
    - text (str): Text to be cleaned.

    Returns:
    - str: Cleaned text with non-word characters removed and converted to uppercase.
    """

    if text:
        text = re.sub(r"[^\w\s]", " ", text)
        return text.upper()
    else:
        return ""


def _drop_duplicates(data, priority):
    """
    Drops duplicate rows based on specified priority.

    Args:
    - data (DataFrame): DataFrame containing rows to check for duplicates.
    - priority (list): Priority for filtering duplicates.

    Returns:
    - DataFrame: DataFrame after dropping duplicates based on the specified priority.
    """

    data["temp_source"] = pd.Categorical(
        data["source"], categories=priority, ordered=True
    )
    data = data.sort_values("temp_source", ascending=True).drop_duplicates(["group"])
    data = data.reset_index(drop=True)
    return data


def _connect_components(data, buffer_size):
    """
    Connects components within a specified buffer size.
    Dissolve overlapping geometries based on: https://gis.stackexchange.com/a/271737

    Args:
    - data (DataFrame): DataFrame containing geometries to connect.
    - buffer_size (float): Buffer size for connecting geometries in meters.

    Returns:
    - DataFrame: DataFrame with connected components marked by a 'group' column.
    """

    temp = data.copy()
    if data.crs != "EPSG:3857":
        temp = data_utils._convert_to_crs(data, target_crs="EPSG:3857")
    geometry = temp["geometry"].buffer(buffer_size, cap_style=3)
    overlap_matrix = geometry.apply(lambda x: geometry.overlaps(x)).values.astype(int)
    n, groups = connected_components(overlap_matrix, directed=False)
    data["group"] = groups
    return data


def _filter_pois_within_distance(data, buffer_size, priority):
    """
    Filters points of interest (POIs) within a specified distance.

    Args:
    - data (DataFrame): DataFrame containing POI information.
    - buffer_size (float): Buffer size for filtering POIs within distance in meters.
    - priority (list): Priority for filtering duplicates.

    Returns:
    - DataFrame: Processed DataFrame with filtered POIs within the specified distance.
    """

    data = _connect_components(data, buffer_size)
    data = _drop_duplicates(data, priority)
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


def generate_ghsl_per_country(config, layer="ghsl", iso_codes=[], exclude=[]):
    """
    Generates GHSL (Global Human Settlement Layer) data per country.

    Args:
    - config (dict): Configuration settings.
    - layer (str, optional): Name of the layer. Defaults to "ghsl".
    - exclude (list, optional): List of ISO codes to exclude. Defaults to an empty list.

    Returns:
    - None
    """

    cwd = os.path.dirname(os.getcwd())
    if len(iso_codes) == 0:
        iso_codes = list(set(config["iso_codes"]) - set(exclude))

    for iso_code in (pbar := data_utils._create_progress_bar(iso_codes)):
        ghsl_path = os.path.join(cwd, config["rasters_dir"], layer, config["ghsl_file"])
        out_tif = os.path.join(cwd, config["rasters_dir"], layer, f"{iso_code}_{layer}.tif")
        pbar.set_description(f"Processing {iso_code}")

        if not os.path.exists(out_tif):
            geoboundary = data_utils._get_geoboundaries(config, iso_code, adm_level="ADM0")
            geoboundary = geoboundary.to_crs("ESRI:54009")

            with rio.open(ghsl_path) as src:
                geometry = [geoboundary.iloc[0]["geometry"]]
                out_image, out_transform = rio.mask.mask(src, geometry, crop=True)

                out_meta = src.meta
                out_meta.update(
                    {
                        "driver": "GTiff",
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform,
                    }
                )

            # Save the cropped image as a temporary TIFF file.
            with rio.open(out_tif, "w", **out_meta) as dest:
                dest.write(out_image)


def _generate_additional_non_school(
    config, 
    iso_code, 
    buffer_size,
    spacing,
    sname="clean"
):
    """
    Generates additional non-school points based on given configurations and spatial parameters.

    Args:
    - config (dict): Configuration settings.
    - iso_code (str): ISO code for a specific location.
    - buffer_size (float): Buffer size for points.
    - spacing (float): Spacing between points.
    - sname (str, optional): Name identifier (default is "clean").

    Returns:
    - GeoDataFrame: GeoDataFrame containing generated non-school points.
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

    # Read school data and perform buffer operation on geometries
    filename = f"{iso_code}_{sname}.geojson"
    school_file = os.path.join(cwd, config["vectors_dir"], "school", sname, filename)
    school = gpd.read_file(school_file).to_crs("EPSG:3857")
    school["geometry"] = school["geometry"].buffer(buffer_size, cap_style=3)
    points["geometry"] = points["geometry"].buffer(buffer_size, cap_style=3)

     # Identify intersecting points and remove them
    points["index"] = points.index
    intersecting = school.sjoin(points, how="inner")["index"]
    points = points[~points["index"].isin(intersecting)]
    points["geometry"] = points["geometry"].centroid
    points = points.to_crs("ESRI:54009")

    # Sample points from the GHSL raster
    coord_list = [(x, y) for x, y in zip(points["geometry"].x, points["geometry"].y)]
    ghsl_path = os.path.join(cwd, config["rasters_dir"], "ghsl", config["ghsl_file"])
    with rio.open(ghsl_path) as src:
        points["ghsl"] = [x[0] for x in src.sample(coord_list)]

    # Filter points with GHSL greater than 0 and convert back to EPSG:4326
    points = points[points['ghsl'] > 0]
    points = points.to_crs("EPSG:4326")
    return points


def _augment_non_school_data(config, category="non_school", name="clean"):
    """
    Augments non-school data by generating additional points and combining datasets 
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
        
        filename = f"{iso_code}_{name}.geojson"
        non_school_file = os.path.join(cwd, config["vectors_dir"], category, name, filename)
        non_school = gpd.read_file(non_school_file)

        if subdata.non_school.iloc[0] < subdata.school.iloc[0]:
            buffer_size = config["school_buffer_size"]
            points = _generate_additional_non_school(
                config, iso_code, buffer_size, spacing=buffer_size*2
            )
            points = data_utils._prepare_data(
                config=config,
                data=points,
                iso_code=iso_code,
                category=category,
                source="GHSL",
                columns=config["columns"]+["ghsl"],
            )
            size = subdata.school.iloc[0] - subdata.non_school.iloc[0]
            points = points.sample(size, random_state=SEED)
            non_school = data_utils._concat_data(
                [points, non_school], 
                non_school_file,
                verbose=False
            )
            
        data.append(non_school)

    # Save combined dataset
    out_dir = os.path.join(cwd, config["vectors_dir"], category)
    out_file = os.path.join(out_dir, f"{name}.geojson")
    data = data_utils._concat_data(data, out_file)
    return data


def _filter_uninhabited_locations(config, data, buffer_size, layer="ghsl", pbar=None):
    """
    Filters uninhabited locations based on buffer size (in meters).

    Args:
    - config (dict): Configuration settings.
    - data (DataFrame): Pandas DataFrame containing location data.
    - buffer_size (float): Buffer size (in meters) around the points.
    - layer (str, optional): Name of the layer. Defaults to "ghsl".
    - pbar (tqdm, optional): Progress bar. Defaults to None.

    Returns:
    - DataFrame: Filtered DataFrame containing inhabited locations.
    """
    
    cwd = os.path.dirname(os.getcwd())
    rasters_dir = config["rasters_dir"]
    data = data.reset_index(drop=True)
    iso_code = data.iso.values[0]

    ghsl_sum = []
    for index in range(len(data)):
        if pbar:
            description = f"Processing {iso_code} {index}/{len(data)}"
            pbar.set_description(description)

        subdata = data.iloc[[index]]
        subdata = subdata.to_crs("EPSG:3857")
        subdata["geometry"] = subdata["geometry"].centroid.buffer(
            buffer_size, cap_style=3
        )

        ghsl_path = os.path.join(cwd, rasters_dir, layer, f"{iso_code}_{layer}.tif")
        if not os.path.exists(ghsl_path):
            ghsl_path = os.path.join(cwd, rasters_dir, layer, config["ghsl_file"])

        image = []
        if os.path.exists(ghsl_path):
            with rio.open(ghsl_path) as src:
                try:
                    subdata = subdata.to_crs(src.crs)
                    geometry = [subdata.iloc[0]["geometry"]]
                    image, transform = rio.mask.mask(src, geometry, crop=True)
                except:
                    image = []
        else:
            image, region = gee_utils.generate_gee_image(subdata[["geometry"]], layer)
            file = gee_utils.export_image(
                image, filename="temp", region=region, mode="local"
            )
            with rio.open(file[0], "r") as src:
                image = src.read(1)

        if len(image) > 0:
            image[image < 0] = 0
            image[image == 255] = 0
            pixel_sum = image.sum()

        ghsl_sum.append(pixel_sum)

    data[layer] = ghsl_sum
    data = data[data[layer] > 0]
    data = data.reset_index(drop=True)
    return data


def _filter_pois_within_school_vicinity(
    config, buffer_size, sname="clean", name="filtered"
):
    """
    Filters points of interest (POIs) located within the vicinity of schools.

    Args:
    - config (dict): Configuration settings.
    - buffer_size (float): Buffer size for school vicinity in meters.
    - sname (str, optional): Name of the school data. Defaults to "clean".
    - name (str, optional): Name for the filtered output. Defaults to "filtered".

    Returns:
    - GeoDataFrame: Processed and filtered GeoDataFrame containing non-school POIs.
    """
    
    # Get the current working directory
    cwd = os.path.dirname(os.getcwd())

    # Set up directories and configurations
    data_dir = config["vectors_dir"]
    out_dir = os.path.join(data_dir, "non_school", name)
    out_dir = data_utils._makedir(out_dir)
    iso_codes = config["iso_codes"]

    # Read non-school dataset
    nonschool_dir = os.path.join(data_dir, "non_school")
    exclude = [f"{sname}.geojson", f"{name}.geojson"]
    nonschool = data_utils._read_data(nonschool_dir, exclude=exclude)

    data = []
    # Iterate over ISO codes to process data
    for iso_code in (pbar := data_utils._create_progress_bar(iso_codes)):
        filename = f"{iso_code}_{sname}.geojson"
        school_file = os.path.join(cwd, data_dir, "school", sname, filename)
        school = gpd.read_file(school_file)

        pbar.set_description(f"Processing {iso_code}")
        out_subfile = os.path.join(out_dir, f"{iso_code}_{name}.geojson")

        if not os.path.exists(out_subfile):
            # Filter non-school POIs within the buffer of school locations
            school_sub = school[school["iso"] == iso_code]
            nonschool_sub = nonschool[nonschool["iso"] == iso_code]

            # Convert school and non-school data CRS to EPSG:3857
            nonschool_temp = data_utils._convert_to_crs(
                nonschool_sub, target_crs="EPSG:3857"
            )
            nonschool_temp["geometry"] = nonschool_temp["geometry"].buffer(buffer_size, cap_style=3)
            nonschool_temp["index"] = nonschool_sub.index
            school_temp = data_utils._convert_to_crs(school_sub, target_crs="EPSG:3857")
            school_temp["geometry"] = school_temp["geometry"].buffer(buffer_size, cap_style=3)

            # Filter out non-school POIs that intersect with buffered school locations
            intersecting = school_temp.sjoin(nonschool_temp, how="inner")["index"]
            nonschool_sub = nonschool_sub[~nonschool_temp["index"].isin(intersecting)]

            # Save filtered country-level dataset
            columns = config["columns"]
            nonschool_sub = nonschool_sub[columns]
            nonschool_sub.to_file(out_subfile, driver="GeoJSON")

        subdata = gpd.read_file(out_subfile).reset_index(drop=True)
        data.append(subdata)

    # Combine and save datasets
    filtered_file = os.path.join(cwd, nonschool_dir, f"{name}.geojson")
    data = data_utils._concat_data(data, out_file=filtered_file)
    return data


def _filter_pois_with_matching_names(data, buffer_size, threshold, priority):
    """
    Filters points of interest (POIs) with matching names within a certain buffer size.

    Args:
    - data (DataFrame): DataFrame containing POI information.
    - buffer_size (float): Buffer size for matching names in meters.
    - threshold (int): Threshold score for similarity of names.
    - priority (list): Priority for filtering duplicates.

    Returns:
    - DataFrame: Processed DataFrame with filtered POIs based on matching names.
    """

    # Get connected components within a given buffer size and get groups with size > 1
    data = _connect_components(data, buffer_size)
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
            score = fuzz.partial_ratio(_clean_text(comb[0][1]), _clean_text(comb[1][1]))
            uid_edge_list.append(
                (comb[0][0], comb[0][1], comb[1][0], comb[1][1], score)
            )
        columns = ["source", "name_1", "target", "name_2", "score"]
        uid_edge_list = pd.DataFrame(uid_edge_list, columns=columns)
        uid_network.append(uid_edge_list)

    # Generate graph and get connected components
    if len(uid_network) > 0:
        uid_network = pd.concat(uid_network)
        uid_network = uid_network[uid_network.score > threshold]

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
            data = _drop_duplicates(data, priority)

    return data


def clean_data(config, category, iso_codes=None, name="clean", gee=False):
    """
    Cleans and processes geographic data based on the specified category.

    Args:
    - config (dict): Configuration settings.
    - category (str): Type of data category ('school' or 'non_school').
    - iso_codes (list or None, optional): List of ISO codes. Defaults to None.
    - name (str, optional): Name for the output file. Defaults to 'clean'.
    - gee (bool, optional): Flag to initialize Google Earth Engine. Defaults to False.

    Returns:
    - GeoDataFrame: Processed and cleaned GeoDataFrame.
    """

    if gee:
        gee_utils._initialize_gee() # Initialize Google Earth Engine if specified

    # Define the output directory for processed data based on the category and name
    out_dir = os.path.join(config["vectors_dir"], category, name)
    out_dir = data_utils._makedir(out_dir) # Create directory if it doesn't exist
    iso_codes = config["iso_codes"] # Fetch ISO codes from the config

    if category == "school":
        data_dir = os.path.join(config["vectors_dir"], category)
        # Read data for the 'school' category, excluding the specified filename
        data = data_utils._read_data(data_dir, exclude=[f"{name}.geojson"])
    # For non-school locations, remove POIs within school vicinity
    elif category == "non_school":
        data = _filter_pois_within_school_vicinity(
            config, buffer_size=config["school_buffer_size"]
        )

    out_data = []
    for iso_code in (pbar := data_utils._create_progress_bar(iso_codes)):
        pbar.set_description(f"Processing {iso_code}")
        out_subfile = os.path.join(out_dir, f"{iso_code}_{name}.geojson")

        if not os.path.exists(out_subfile):
            # Join the dataset with ADM1 geoboundaries
            subdata = data[data["iso"] == iso_code].reset_index(drop=True)
            geoboundaries = data_utils._get_geoboundaries(
                config, iso_code, adm_level="ADM1"
            )
            geoboundaries = geoboundaries[["shapeName", "geometry"]].dropna(
                subset=["shapeName"]
            )
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
                    # Remove schools containing certain keywords
                    if category == "school":
                        subsubdata = _filter_keywords(
                            subsubdata, exclude=config["exclude"]
                        )[columns]

                    # Remove POIs within 50 meters of each other
                    subsubdata = _filter_pois_within_distance(
                        subsubdata,
                        buffer_size=config["buffer_size"],
                        priority=config["priority"],
                    )[columns]

                    # Remove POIs with matching names within 500 meters of each other
                    subsubdata = _filter_pois_with_matching_names(
                        subsubdata,
                        priority=config["priority"],
                        threshold=config["name_match_threshold"],
                        buffer_size=config["name_match_buffer_size"],
                    )[columns]

                    # Filter uninhabited locations based on specified buffer size
                    subsubdata = _filter_uninhabited_locations(
                        config,
                        subsubdata,
                        buffer_size=config["ghsl_buffer_size"],
                        pbar=pbar,
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

    if category == "non_school":
        data = _augment_non_school_data(config, category)
    
    return data
    