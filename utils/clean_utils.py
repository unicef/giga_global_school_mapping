import re
import os
import geojson
import pandas as pd
import geopandas as gpd
from tqdm.notebook import tqdm
import data_utils

import itertools
import numpy as np
import networkx as nx
from rapidfuzz import fuzz
import data_utils


def filter_nonschools_within_school_vicinity(
    school_data_file,
    nonschool_data_dir,
    out_dir,
    out_file="filtered.geojson",
    iso_codes=None,
    buffer_size=200
):
    cwd = os.path.dirname(os.getcwd())
    data_config = data_utils._load_data_config()
    
    out_dir = data_utils._makedir(out_dir)
    filtered_file = os.path.join(os.path.join(cwd, nonschool_data_dir), out_file)
    
    school = gpd.read_file(os.path.join(cwd, school_data_file))
    nonschool = data_utils._read_data(os.path.join(cwd, nonschool_data_dir))
    
    if not iso_codes:
        iso_codes = school.iso.unique()
    
    data = []
    pbar = tqdm(enumerate(iso_codes), total=len(iso_codes))
    for _, iso_code in pbar:
        pbar.set_description(f"Processing {iso_code}")
        
        out_subfile = f"{iso_code}_filtered.geojson"
        out_subfile = os.path.join(out_dir, out_subfile)
        
        if not os.path.exists(out_subfile):
            school_subdata = school[school["iso"] == iso_code]
            nonschool_subdata = nonschool[nonschool["iso"] == iso_code]

            nonschool_temp = data_utils._convert_to_crs(
                nonschool_subdata, target_crs="EPSG:3857"
            )
            nonschool_temp['geometry'] = nonschool_temp["geometry"]
            nonschool_temp['index'] = nonschool_subdata.index

            school_temp = data_utils._convert_to_crs(
                school_subdata, target_crs="EPSG:3857"
            )
            school_temp['geometry'] = school_temp["geometry"].buffer(buffer_size)

            intersecting = school_temp.sjoin(
                nonschool_temp, predicate="intersects", how='inner'
            )['index']
            nonschool_subdata = nonschool_subdata[
                ~nonschool_temp['index'].isin(intersecting)
            ]
            columns = data_config["COLUMNS"]
            nonschool_subdata = nonschool_subdata[columns]
            nonschool_subdata.to_file(out_subfile, driver="GeoJSON")
            
        subdata = gpd.read_file(out_subfile).reset_index(drop=True)
        data.append(subdata)
    
    # Combine datasets
    data = data_utils._concat_data(data, filtered_file)
    return data
    


def filter_schools_with_matching_names(data, threshold, prioritization, buffer_size):
    # Get connected components within a given buffer size
    # and get groups with size > 1
    data = data_utils._connect_components(data, buffer_size)
    group_count = data.group.value_counts()
    groups = group_count[group_count > 1].index

    uid_network = []
    for index in range(len(groups)):
        # Get pairwise combination of names within a group
        subdata = data[data.group == groups[index]][["UID", "name"]]
        subdata = list(subdata.itertuples(index=False, name=None))
        combs = itertools.combinations(subdata, 2)

        # Compute rapidfuzz score (partial ratio)
        uid_edge_list = []
        for comb in combs:
            score = fuzz.partial_ratio(
                data_utils._clean_text(comb[0][1]), data_utils._clean_text(comb[1][1])
            )
            uid_edge_list.append(
                (comb[0][0], comb[0][1], comb[1][0], comb[1][1], score)
            )

        # Create edge list
        columns = ["source", "name_1", "target", "name_2", "score"]
        uid_edge_list = pd.DataFrame(uid_edge_list, columns=columns)
        uid_network.append(uid_edge_list)

    # Generate graph and get connected components
    if len(uid_network) > 0:
        uid_network = pd.concat(uid_network)
        uid_network = uid_network[uid_network.score > threshold]
        columns = ["source", "target", "score"]
        graph = nx.from_pandas_edgelist(uid_network[columns])
        groups = {
            n: i
            for i, g in enumerate(nx.connected_components(graph), start=1)
            for n in g
        }

        if len(groups) > 0:
            # Assign groups to data
            data["group"] = np.nan
            for uid, value in groups.items():
                data.loc[data["UID"] == uid, "group"] = value
            max_group = int(data["group"].max()) + 1

            # Fill nan values
            fillna = list(range(max_group, len(data) + max_group))
            data["group"] = data.apply(
                lambda x: x["group"]
                if not np.isnan(x["group"])
                else fillna[int(x.name)],
                axis=1,
            )

            # Drop duplicates in group based on prioritization
            data = data_utils._prioritization(data, prioritization)
    return data


def deduplicate_data(
    out_dir,
    data_file=None,
    data_dir=None,
    out_file="clean.geojson",
    iso_codes=None,
    buffer_size=50,
    threshold=90,
    matching_names_buffer_size=150,
    prioritization=["OVERTURE", "OSM", "UNICEF"],
):
    # Read files in data_dir --> should contain Overture, OSM, and UNICEF files
    cwd = os.path.dirname(os.getcwd())
    out_dir = data_utils._makedir(out_dir)
    data_config = data_utils._load_data_config()
    
    if data_file:
        data = gpd.read_file(os.path.join(cwd, data_file))
    elif data_dir:
        data = data_utils._read_data(data_dir, out_file)
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
            geoboundaries = data_utils._get_geoboundaries(iso_code, adm_level="ADM1")[
                columns
            ]
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
                    columns = data_config["COLUMNS"]
                    subsubdata = data_utils._connect_components(subsubdata, buffer_size)
                    subsubdata = data_utils._prioritization(subsubdata, prioritization)

                    if "giga_id_school" in subsubdata.columns:
                        columns = columns + ["giga_id_school"]
                    subsubdata = subsubdata[columns]
                    subsubdata = filter_schools_with_matching_names(
                        subsubdata,
                        threshold=threshold,
                        prioritization=prioritization,
                        buffer_size=matching_names_buffer_size,
                    )[columns]
                    out_subdata.append(subsubdata)

            # Save cleaned file
            out_subdata = data_utils._concat_data(
                out_subdata, out_subfile, verbose=False
            )

        out_subdata = gpd.read_file(out_subfile).reset_index(drop=True)
        out_data.append(out_subdata)

    # Save combined dataset
    out_dir = os.path.dirname(out_dir)
    out_file = os.path.join(out_dir, out_file)
    data = data_utils._concat_data(out_data, out_file)
    return data
