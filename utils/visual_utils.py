import os
import folium
import geopandas as gpd
import rasterio as rio
import data_utils

from rasterio.plot import show
import matplotlib.pyplot as plt
from IPython.display import display


def map_coords(index, category, iso, zoom_start=18, max_zoom=20, name="clean"):
    data_config = data_utils._load_data_config()    
    cwd = os.path.dirname(os.getcwd())
    
    vector_dir = data_config['data_dir']
    filename =  f"{iso}_{name}.geojson"
    filename = os.path.join(cwd, vector_dir, category, name, filename)
    
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
    iso,
    category,
    n_rows=8,
    n_cols=4,
    index=0,
    figsize=(15, 35),
    filename=None,
    name="clean",
):
    data_config = data_utils._load_data_config()
    cwd = os.path.dirname(os.getcwd())
    
    image_dir = data_config['rasters_dir']
    vector_dir = data_config['data_dir']
    
    if not filename:
        filename =  f"{iso}_{name}.geojson"
        filename = os.path.join(cwd, vector_dir, category, name, filename)
    
    data = gpd.read_file(filename)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    samples = data.iloc[index : index + (n_rows * n_cols)]
    row_index, col_index = 0, 0

    increment = 1
    for idx, item in samples.iterrows():
        class_dir = os.path.join(cwd, image_dir, f"{iso}/{category.lower()}")
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