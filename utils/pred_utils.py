import os
import pandas as pd
import geopandas as gpd
import logging
import torch

import os
import pandas as pd
import geopandas as gpd
from PIL import Image
import logging
import joblib
import torch

import cnn_utils
import data_utils
import config_utils
import pred_utils
import embed_utils

import torch
import torch.nn.functional as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO)
        

def cnn_predict_images(data, model, config, in_dir, classes):
    """
    Predicts labels and probabilities for the given dataset using the provided model.

    Args:
    - data (Pandas DataFrame): Input dataset.
    - model (torch.nn.Module): Trained neural network model.
    - c (dict): Configuration dictionary.
    - in_file (str): Path to the input file.
    - out_dir (str): Directory path to save output files.
    - classes (dict): Dictionary containing class labels.
    - scale (float, optional): Scale factor for cropping shapes (default: 1.5).

    Returns:
    - Results as a GeoDataFrame containing predicted labels and probabilities for the dataset.
    """
    
    files = data_utils.get_image_filepaths(config, data, in_dir)
    preds, probs = [], []
    pbar = data_utils._create_progress_bar(files)
    for file in pbar:
        image = Image.open(file).convert("RGB")
        transforms = cnn_utils.get_transforms(config["img_size"])
        output = model(transforms["test"](image.to(device)).unsqueeze(0))
        prob = nn.softmax(output, dim=1).detach().numpy()[0]
        probs.append(prob)
        _, pred = torch.max(output, 1)
        label = str(classes[int(pred[0])])
        preds.append(label)

    probs_col = [f"{classes[index]}_PROB" for index in range(len(classes))]
    probs = pd.DataFrame(probs, columns=probs_col)
    data["pred"] = preds
    data["prob"] = probs.max(axis=1)

    return data


def cnn_predict(tiles, iso_code, shapename, config, in_dir, n_classes=None):
    """
    Predicts classes for buildings using a trained model and input image file.

    Args:
    - bldgs (GeoDataFrame): Building dataset to predict classes for.
    - in_file (str): Input image file used for prediction.
    - exp_config (str): Path to the experiment configuration file.
    - model_file (str, optional): Path to the trained model file.
    - prefix (str, optional): Prefix for file paths.

    Returns:
    - Predictions for the building dataset using the trained model.
    """
    cwd = os.path.dirname(os.getcwd())
    classes = {0: config["neg_class"], 1: config["pos_class"]}
    
    exp_dir = os.path.join(cwd, config["exp_dir"], f"{iso_code}_{config['config_name']}")
    model_file = os.path.join(exp_dir, f"{iso_code}_{config['config_name']}.pth")
    model = load_cnn(config, classes, model_file)

    out_dir = os.path.join("output", iso_code, "results")
    out_dir = data_utils._makedir(out_dir)
    name = f"{iso_code}_{shapename}"
    
    out_file = os.path.join(out_dir, f"{name}_{config['config_name']}_results.gpkg")
    results = cnn_predict_images(tiles, model, config, in_dir, classes)
    results = results[["UID", "geometry", "shapeName", "pred", "prob"]]
    results = gpd.GeoDataFrame(results, geometry="geometry")
    results.to_file(out_file, driver="GPKG")
    
    return results


def load_cnn(c, classes, model_file=None):
    """
    Loads a pre-trained model based on the provided configuration.

    Args:
    - c (dict): Configuration dictionary containing model details.
    - classes (dict): Dictionary containing class labels.
    - model_file (str, optional): Path to the pre-trained model file.

    Returns:
    - Loaded pre-trained model based on the provided configuration.
    """
    
    n_classes = len(classes)
    model = cnn_utils.get_model(c["model"], n_classes, c["dropout"])
    model.load_state_dict(torch.load(model_file, map_location=device))
    model = model.to(device)
    model.eval()
    logging.info(f"Device: {device}")
    
    logging.info("Model file {} successfully loaded.".format(model_file))
    return model


def load_vit(config):
    model = torch.hub.load("facebookresearch/dinov2", config["embed_model"])
    model.name = config["embed_model"]
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    logging.info(f"Device: {device}")
    return model


def vit_pred(tiles, config, iso_code, shapename, sat_dir, id_col="UID"):
    cwd = os.path.dirname(os.getcwd())
    model = load_vit(config)

    # Generate Embeddings
    logging.info("Generating embeddings...")
    out_dir = os.path.join("output", iso_code, "embeddings")
    name = f"{iso_code}_{shapename}"
    embeddings = embed_utils.get_image_embeddings(
        config, tiles, model, in_dir=sat_dir, out_dir=out_dir, name=name
    )
    if id_col in embeddings.columns:
        embeddings = embeddings.set_index(id_col)

    # Load shallow model
    exp_dir = os.path.join(cwd, config["exp_dir"], f"{iso_code}-{config['config_name']}")
    model_file = os.path.join(exp_dir, f"{iso_code}-{config['config_name']}.pkl")
    model = joblib.load(model_file)
    logging.info(f"Loaded {model_file}")

    # Model prediction
    preds = model.predict(embeddings)
    tiles["pred"] = preds
    results = tiles[["UID", "geometry", "shapeName", "pred"]]
    results = gpd.GeoDataFrame(results, geometry="geometry")

    # Save results
    out_dir = os.path.join("output", iso_code, "results")
    out_dir = data_utils._makedir(out_dir)
    out_file = os.path.join(out_dir, f"{name}_{config['config_name']}_results.gpkg")
    results.to_file(out_file, driver="GPKG")
    logging.info(f"Generated {out_file}")
    
    return results


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
