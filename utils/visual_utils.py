import os
import folium
import geopandas as gpd
import rasterio as rio
import data_utils
import logging

from ipywidgets import (
    Layout,
    GridspecLayout, 
    Button, 
    Image
)
from rasterio.plot import show
import matplotlib.pyplot as plt
from IPython.display import display

logging.basicConfig(level=logging.INFO)


def _get_filename(cwd, iso, vector_dir, category, name):
    filename = os.path.join(
        cwd, vector_dir, category, name, f"{iso}_{name}.geojson"
    )
    return filename 


def map_coordinates(
    config, 
    index, 
    category, 
    iso, 
    filename=None, 
    zoom_start=18, 
    max_zoom=20, 
):
    cwd = os.path.dirname(os.getcwd())
    vector_dir = config["VECTORS_DIR"]

    if not filename:
        filename = _get_filename(cwd, iso, vector_dir, category, "validate")
        if not os.path.exists(filename):
            filename = _get_filename(cwd, iso, vector_dir, category, "clean")

    data = gpd.read_file(filename)
    logging.info(data.iloc[index]['name'])
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


def data_cleaning(
    config, 
    iso, 
    category, 
    start_index=0,
    row_inc=3, 
    n_rows=4, 
    n_cols=4, 
    filename=None
):
    cwd = os.path.dirname(os.getcwd())
    image_dir = config["RASTERS_DIR"]
    vector_dir = config["VECTORS_DIR"]
    dir_ = config["DIR"]

    if not filename:
        filename = _get_filename(cwd, iso, vector_dir, category, "validate")
        if not os.path.exists(filename):
            data = gpd.read_file(_get_filename(cwd, iso, vector_dir, category, "clean"))
            data["index"] = data.index
            data["validate"] = 0
            dir = os.path.dirname(filename)
            if not os.path.exists(dir):
                os.makedirs(dir)
            data.to_file(filename, driver="GeoJSON")
        else: 
            data = gpd.read_file(filename)
    
    samples = data.iloc[start_index : start_index + (n_rows * n_cols)]
    grid = GridspecLayout(n_rows*row_inc+n_rows, n_cols)
    button_dict = {0: ("primary", category), -1: ("warning", "removed")}

    def _add_image(item):
        class_dir = os.path.join(cwd, image_dir, dir_, iso, category.lower())
        filepath = os.path.join(class_dir, f"{item.UID}.tiff")
        img = open(filepath, 'rb').read()
        return Image(
            value=img, 
            format='png', 
            layout=Layout(
                justify_content="center",
                border="solid",
                width='auto',
                height='auto'
            )
        )

    def _on_button_click(button):
        index = int(button.description.split(" ")[0])
        item = data.iloc[index]

        change_value = -1
        if item.validate == -1:
            change_value = 0
        button_style, category = button_dict[change_value]
        
        button.button_style = button_style
        button.description = f"{item.name} {category.upper()}"

        data.loc[index, 'validate'] = change_value
        data.to_file(filename, driver="GeoJSON")
        
    
    def _create_button(item):
        val = item.validate
        button_style, category = button_dict[val]
        description = f"{item.name} {category.upper()}"
        
        return Button(
            description=description, 
            button_style=button_style, 
            layout=Layout(
                justify_content="center",
                border="solid",
                width='auto',
                height='10'
            )
        )
    
    row_index, col_index = 0, 0
    for index, item in samples.iterrows():
        grid[row_index:row_index+row_inc, col_index] = _add_image(item)    
        button = _create_button(item)
        button.on_click(_on_button_click)
        grid[row_index+row_inc, col_index] = button
        
        col_index += 1
        if col_index >= n_cols:
            row_index += row_inc + 1
            col_index = 0
            
    return grid


def inspect_images(
    config,
    iso,
    category,
    n_rows=4,
    n_cols=4,
    start_index=0,
    figsize=(15, 15),
    filename=None,
    name="validate",
):
    cwd = os.path.dirname(os.getcwd())

    image_dir = config["RASTERS_DIR"]
    vector_dir = config["VECTORS_DIR"]
    dir_ = config["DIR"]

    if not filename:
        filename = _get_filename(cwd, iso, vector_dir, category, "validate")
        if not os.path.exists(filename):
            filename = _get_filename(cwd, iso, vector_dir, category, "clean")

    data = gpd.read_file(filename)
    if 'validate' in data.columns:
        data = data[data["validate"] > -1]
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    samples = data.iloc[start_index : start_index + (n_rows * n_cols)]
    row_index, col_index = 0, 0

    increment = 1
    for idx, item in samples.iterrows():
        class_dir = os.path.join(cwd, image_dir, dir_, iso, category.lower())
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
