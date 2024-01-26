import os
import pandas as pd
import geopandas as gpd
import logging
import torch

import config_utils
import embed_utils
import data_utils


def generate_pred_tiles(config, iso_code, spacing, buffer_size, adm_level="ADM2", shapename=None):
    cwd = os.path.dirname(os.getcwd())
    points = data_utils._generate_samples(
        config, 
        iso_code=iso_code, 
        buffer_size=buffer_size, 
        spacing=spacing, 
        adm_level=adm_level, 
        shapename=shapename
    )
    points["points"] = points["geometry"]
    points["geometry"] = points.buffer(buffer_size, cap_style=3)

    filtered = []
    ms_dir = os.path.join(cwd, config["vectors_dir"], "ms_buildings", iso_code)
    pbar = data_utils._create_progress_bar(os.listdir(ms_dir))
    for file in pbar:
        filename = os.path.join(ms_dir, file)
        ms = gpd.read_file(filename)
        ms = ms.to_crs(points.crs)
        filtered.append(
            gpd.sjoin(points, ms, predicate='intersects', how="inner")
        )
        
    filtered = gpd.GeoDataFrame(pd.concat(filtered), geometry="geometry")
    filtered = filtered.drop_duplicates("geometry", keep="first")

    return filtered
