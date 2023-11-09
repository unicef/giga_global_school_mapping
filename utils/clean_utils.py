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
import data_utils
import gee_utils
from scipy.sparse.csgraph import connected_components

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
logging.basicConfig(level=logging.INFO)


def _clean_text(text):
    """Remove all non-word characters"""
    if text:
        text = re.sub(r"[^\w\s]", " ", text)
        text = text.upper()
    else:
        text = ""
    return text


def _drop_duplicates(data, priority):
    data["temp_source"] = pd.Categorical(data["source"], categories=priority, ordered=True)
    data = data.sort_values("temp_source", ascending=True).drop_duplicates(["group"])
    data = data.reset_index(drop=True)
    return data
    

def _connect_components(data, buffer_size):
    # Dissolve overlapping geometries based on: https://gis.stackexchange.com/a/271737

    temp = data.copy()
    if data.crs != "EPSG:3857": temp = data_utils._convert_to_crs(data, target_crs="EPSG:3857")
    geometry = temp["geometry"].buffer(buffer_size)
    overlap_matrix = geometry.apply(lambda x: geometry.overlaps(x)).values.astype(int)
    n, groups = connected_components(overlap_matrix, directed=False)
    data["group"] = groups
    return data


def _filter_pois_within_distance(data, buffer_size, priority):
    data = _connect_components(data, buffer_size)
    data = _drop_duplicates(data, priority)
    return data


def _filter_keywords(data, exclude):
    exclude =  [f"\\b{x.upper()}\\b" for x in exclude]
    data = data[~data["name"].str.upper().str.contains(r'|'.join(exclude), case=False, na=False)]
    return data


def _filter_uninhabited_locations(config, data, buffer_size, layer="ghsl", pbar=None):
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
        subdata["geometry"] = subdata["geometry"].centroid.buffer(buffer_size, cap_style=3)

        ghsl_file = "GHS_BUILT_C_FUN_E2018_GLOBE_R2023A_54009_10_V1_0.tif"
        ghsl_path = os.path.join(cwd, rasters_dir, layer, "global", ghsl_file)
        if os.path.exists(ghsl_path):
            subdata = subdata.to_crs("ESRI:54009")
        else:
            ghsl_path = os.path.join(cwd, rasters_dir, layer, f"{iso_code}_{layer}.tif")
            subdata = subdata.to_crs("EPSG:4326")
        
        if os.path.exists(ghsl_path):  
            with rio.open(ghsl_path) as src:
                geometry = [subdata.iloc[0]['geometry']]
                image, _ = rio.mask.mask(src, geometry, crop=True)
        else:
            image, region = gee_utils.generate_gee_image(subdata[["geometry"]], layer)
            file = gee_utils.export_image(image, filename="temp", region=region, mode='local')
    
            with rio.open(file[0], "r") as src:
                image = src.read(1)
                
        image[image == -32768] = 0
        sum_ = image.sum()
        ghsl_sum.append(sum_)

    data[layer] = ghsl_sum
    data = data[data[layer] > 0]
    data = data.reset_index(drop=True)
    return data


def _filter_pois_within_school_vicinity(config, buffer_size, sname="clean", name="filtered"):
    cwd = os.path.dirname(os.getcwd())
    data_dir = config["vectors_dir"]
    
    out_dir = os.path.join(data_dir, "non_school", name)
    out_dir = data_utils._makedir(out_dir)
    iso_codes = config["iso_codes"]

    nonschool_dir = os.path.join(data_dir, "non_school")
    exclude = [f"{sname}.geojson", f"{name}.geojson"]
    nonschool = data_utils._read_data(nonschool_dir, exclude=exclude)

    data = []
    for iso_code in (pbar := data_utils._create_progress_bar(iso_codes)):
        filename = f"{iso_code}_{sname}.geojson"
        school_file_ = os.path.join(cwd, data_dir, "school", sname, filename)
        if not os.path.exists(school_file_): 
            continue
        school = gpd.read_file(school_file_)

        pbar.set_description(f"Processing {iso_code}")
        out_subfile = os.path.join(out_dir, f"{iso_code}_{name}.geojson")

        if not os.path.exists(out_subfile):
            school_sub = school[school["iso"] == iso_code]
            nonschool_sub = nonschool[nonschool["iso"] == iso_code]

            # Convert school and non-school data CRS to EPSG:3857
            nonschool_temp = data_utils._convert_to_crs(nonschool_sub, target_crs="EPSG:3857")
            nonschool_temp["geometry"] = nonschool_temp["geometry"].buffer(buffer_size)
            nonschool_temp["index"] = nonschool_sub.index
            school_temp = data_utils._convert_to_crs(school_sub, target_crs="EPSG:3857")
            school_temp["geometry"] = school_temp["geometry"].buffer(buffer_size)

            # Filter out non-school POIs that intersect with buffered school locations
            intersecting = school_temp.sjoin(nonschool_temp, how="inner")["index"]
            nonschool_sub = nonschool_sub[~nonschool_temp["index"].isin(intersecting)]

            # Save country-level dataset
            columns = config["columns"]
            nonschool_sub = nonschool_sub[columns]
            nonschool_sub.to_file(out_subfile, driver="GeoJSON")

        subdata = gpd.read_file(out_subfile).reset_index(drop=True)
        data.append(subdata)

    # Combine datasets
    filtered_file = os.path.join(cwd, nonschool_dir, f"{name}.geojson")
    data = data_utils._concat_data(data)
    data.to_file(filtered_file, driver="GeoJSON")
    return data


def _filter_pois_with_matching_names(data, buffer_size, threshold, priority):
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
            score = fuzz.partial_ratio(_clean_text(comb[0][1]), _clean_text(comb[1][1]))
            uid_edge_list.append((comb[0][0], comb[0][1], comb[1][0], comb[1][1], score))
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
            num : index
            for index, group in enumerate(connected_components, start=1)
            for num in group
        }

        if len(groups) > 0:
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
    if gee: gee_utils._initialize_gee()

    # Output directory is data/vectors/school/clean
    out_dir = os.path.join(config["vectors_dir"], category, name)
    out_dir = data_utils._makedir(out_dir)
    iso_codes = config["iso_codes"]

    if category == "school":
        data_dir = os.path.join(config["vectors_dir"], category)
        data = data_utils._read_data(data_dir, exclude=[f"{name}.geojson"])

    # For non-school locations, remove POIs within school vicinity
    elif category == "non_school":
        data = _filter_pois_within_school_vicinity(config, buffer_size=config["school_buffer_size"])

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

            # Split the data into smaller admin boundaries fo scalability
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
                    
                    # Remove schools within 50 meters of each other
                    subsubdata = _filter_pois_within_distance(
                        subsubdata, 
                        buffer_size=config['buffer_size'], 
                        priority=config["priority"],
                    )[columns]

                    # Remove schools with matching names within 500 meters of each other
                    subsubdata = _filter_pois_with_matching_names(
                        subsubdata,
                        priority=config["priority"],
                        threshold=config["name_match_threshold"],
                        buffer_size=config["name_match_buffer_size"]
                    )[columns]

                    # Filter uninhabited locations
                    subsubdata = _filter_uninhabited_locations(
                        config,
                        subsubdata,
                        buffer_size=config["ghsl_buffer_size"], 
                        pbar=pbar
                    )
                    out_subdata.append(subsubdata)

            # Save cleaned file
            out_subdata = data_utils._concat_data(out_subdata, verbose=False)                
            out_subdata.to_file(out_subfile, driver="GeoJSON")
            
        out_subdata = gpd.read_file(out_subfile).reset_index(drop=True)
        out_subdata.to_file(out_subfile, driver="GeoJSON")
        out_data.append(out_subdata)

    # Save combined dataset
    out_file = os.path.join(os.path.dirname(out_dir), f"{name}.geojson")
    data = data_utils._concat_data(out_data, out_file)
    data.to_file(out_file, driver="GeoJSON")

    return data
