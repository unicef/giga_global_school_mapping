import os
import argparse
import logging
import pandas as pd
import geopandas as gpd
import logging
import joblib
import torch

import sys
sys.path.insert(0, "../src")
import sat_download

sys.path.insert(0, "../utils/")
import data_utils
import config_utils
import pred_utils
import embed_utils
from torchcam.methods import LayerCAM

logging.basicConfig(level=logging.INFO)


def main(args):
    cwd = os.path.dirname(os.getcwd())
    iso_code = args.iso
    
    data_config_file = os.path.join(cwd, args.data_config)
    data_config = config_utils.load_config(data_config_file)
    
    sat_config_file = os.path.join(cwd, args.sat_config)
    sat_creds_file = os.path.join(cwd, args.sat_creds)
    
    sat_config = config_utils.load_config(sat_config_file)
    sat_creds = config_utils.create_config(sat_creds_file)

    tiles = pred_utils.generate_pred_tiles(
        data_config, iso_code, args.spacing, args.buffer_size, args.adm_level, args.shapename
    ).reset_index()
    logging.info(f"Total tiles: {tiles.shape}")
    
    data = tiles.copy()
    data["geometry"] = data["points"]
    sat_dir = os.path.join(cwd, "output", iso_code, "images", args.shapename)
    sat_download.download_sat_images(sat_creds, sat_config, data=data, out_dir=sat_dir)

    model_config_file = os.path.join(cwd, args.model_config)
    model_config = config_utils.load_config(model_config_file)
    geotiff_dir = data_utils._makedir(os.path.join("output", iso_code, "geotiff", args.shapename))

    if "cnn" in model_config_file:
        results = pred_utils.cnn_predict(
            tiles, iso_code, args.shapename, model_config, sat_dir, n_classes=2
        )
        subdata = results[results["pred"] == model_config["pos_class"]]
        pred_utils.georeference_images(subdata, sat_config, sat_dir, geotiff_dir)
        out_file = f"{iso_code}_{args.shapename}_{model_config['model']}_cam.gpkg"
        pred_utils.cam_predict(model_config, geotiff_dir, out_file)
    else:
        results = pred_utils.vit_pred(
            tiles, model_config, iso_code, args.shapename, sat_dir
        )
    return results


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description="Model Prediction")
    parser.add_argument("--data_config", help="Data config file")
    parser.add_argument("--model_config", help="Model config file")
    parser.add_argument("--sat_config", help="Maxar config file")
    parser.add_argument("--sat_creds", help="Credentials file")
    parser.add_argument("--shapename", help="Model shapename")
    parser.add_argument("--adm_level", help="Admin level", default="ADM2")
    parser.add_argument("--spacing", help="Tile spacing", default=150)
    parser.add_argument("--buffer_size", help="Buffer size", default=150)
    parser.add_argument("--iso", help="ISO code")
    args = parser.parse_args()
    logging.info(args)

    main(args)

    