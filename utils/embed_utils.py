import os
import json
import glob

from tqdm import tqdm
import numpy as np
import logging
import data_utils
import pandas as pd

import cv2
import torch
from torchvision import transforms
from PIL import Image, ImageFile

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random
SEED = 42

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

def get_image_filepaths(config, data, in_dir=None):
    filepaths = []
    cwd = os.path.dirname(os.getcwd())
    for index, row in data.iterrows():
        file = f"{row['UID']}.tiff"
        if not in_dir:
            filepath = os.path.join(
                cwd, 
                config["rasters_dir"], 
                config["maxar_dir"], 
                row["iso"], 
                row["class"],
                file 
            )
        else:
            filepath = os.path.join(in_dir, file)
        filepaths.append(filepath)
    return filepaths
    

def load_image(image_file, image_size) -> torch.Tensor:
    """
    Load an image and return a tensor that can be used as an input to DINOv2.
    """
    image = Image.open(image_file)
    image = np.array(image)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(image_size, antialias=True),
        transforms.CenterCrop(image_size), 
        transforms.Normalize([0.5], [0.5])
    ])
    return transform(image)[:3].unsqueeze(0)


def compute_embeddings(files, model, image_size):        
    embeddings = []
    with torch.no_grad():
        pbar = data_utils._create_progress_bar(files)
        for file in pbar:
            embedding = model(load_image(file, image_size).to(device))
            embedding = np.array(embedding[0].cpu().numpy()).reshape(1, -1)[0]
            embeddings.append(embedding)
    return embeddings


def get_image_embeddings(
    config, 
    data, 
    model, 
    out_dir, 
    in_dir=None,
    columns=[],
    name=None
):
    files = get_image_filepaths(config, data, in_dir)
    if not name: 
        name = config["name"] if "name" in config else data.iloc[0].iso        

    out_dir = data_utils._makedir(out_dir)
    filename = os.path.join(out_dir, f"{name}_{model.name}_embeds.csv")
    
    if os.path.exists(filename):
        embeddings = pd.read_csv(filename)
        logging.info(f"Reading file {filename}")
        return embeddings
    
    embeddings = compute_embeddings(files, model, config["image_size"])
    embeddings = pd.DataFrame(data=embeddings, index=data.UID)

    for column in columns:
        embeddings[column] = data[column].values
    embeddings.to_csv(filename)
    
    logging.info(f"Saved to {filename}")
    return embeddings

def visualize_embeddings(
    config,
    data,
    model,
    batch_size=4
):  
    # Source: https://betterprogramming.pub/dinov2-the-new-frontier-in-self-supervised-learning-b3a939f6d533
    image_size = config["image_size"]
    imgs_tensor = torch.zeros(batch_size, 3, image_size, image_size)
    filepaths = get_image_filepaths(config, data)
    indexes = [random.randint(0, len(data)) for x in range(batch_size)]
    for i, index in enumerate(indexes):
        image = load_image(filepaths[index], image_size)
        imgs_tensor[i] = image[:3]
    
    with torch.no_grad():
      features_dict = model.forward_features(imgs_tensor.to(device))
      features = features_dict['x_norm_patchtokens']
    feature_shape = features.shape
    
    # Compute PCA between the patches of the image
    features = features.reshape(batch_size*feature_shape[1], feature_shape[2])
    features = features.cpu()
    pca = PCA(n_components=3)
    pca.fit(features)
    pca_features = pca.transform(features)
    
    # Visualize the first PCA component
    figsize=(15, 6)
    fig, axes = plt.subplots(2, batch_size, figsize=figsize)
    for i, index in enumerate(indexes):
        image = np.asarray(Image.open(filepaths[index]))
        size = int(feature_shape[1])
        features = pca_features[i*size: (i+1)*size, 0].reshape(
            int(np.sqrt(size)), int(np.sqrt(size))
        )
        axes[0, i].imshow(image)
        axes[1, i].imshow(features)
        category = filepaths[index].split('/')[-2]
        axes[0, i].set_title(category, fontdict={"fontsize": 9})
    plt.show()
