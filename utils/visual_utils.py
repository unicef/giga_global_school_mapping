import os
import folium
import geopandas as gpd
import rasterio as rio
import data_utils
import clean_utils
import logging

from ipywidgets import Layout, GridspecLayout, Button, Image
from rasterio.plot import show
import matplotlib.pyplot as plt
from IPython.display import display
import translators as ts

logging.basicConfig(level=logging.INFO)


def _get_filename(cwd, iso, vector_dir, category, name):
    """
    Generate a file path based on provided parameters.

    Args:
    - cwd (str): Current working directory.
    - iso (str): ISO code for the country.
    - vector_dir (str): Directory for vector data.
    - category (str): Category of the data.
    - name (str): Name used for the file.

    Returns:
    - str: File path generated using the parameters.
    """
    
    filename = os.path.join(
        cwd, 
        vector_dir, 
        category, name, 
        f"{iso}_{name}.geojson"
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
    name="clean",
    id_col="UID",
    name_col="name"
):
    """
    Generate and display a Folium map centered around the coordinates of a specific data point.

    Args:
    - config (dict): Configuration settings.
    - index (int): Index of the data point to be displayed.
    - category (str): Category of the data.
    - iso (str): ISO code for the country.
    - filename (str, optional): File name to load the data. Defaults to None.
    - zoom_start (int, optional): Initial zoom level for the map. Defaults to 18.
    - max_zoom (int, optional): Maximum zoom level allowed for the map. Defaults to 20.
    - name (str, optional): Column name used for data filtering. Defaults to "validated".

    Returns:
    - folium.Map: Folium map displaying the location of the specified data point.
    """
    
    cwd = os.path.dirname(os.getcwd())
    vector_dir = config["vectors_dir"]
    
    filename = _get_filename(cwd, iso, vector_dir, category, "clean")
    data = gpd.read_file(filename)

    name = data[data.index == index].iloc[0][name_col]
    logging.info(data[data.index == index].iloc[0][id_col])
    logging.info(name)
    if name:
        logging.info(ts.translate_text(name, translator="google"))
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


def validate_data(
    config,
    iso,
    category,
    start_index=0,
    row_inc=3,
    n_rows=4,
    n_cols=4,
    filename=None,
    name="clean",
    id_col="UID",
    show_validated=True
):
    """
    Perform data cleaning and provide an interactive widget for data inspection and validation.

    Args:
    - config (dict): Configuration settings.
    - iso (str): ISO code for the country.
    - category (str): Category of the data.
    - start_index (int, optional): Starting index for data sampling. Defaults to 0.
    - row_inc (int, optional): Increment value for rows. Defaults to 3.
    - n_rows (int, optional): Number of rows in the visualization grid. Defaults to 4.
    - n_cols (int, optional): Number of columns in the visualization grid. Defaults to 4.
    - filename (str, optional): File name to load or save the cleaned data. Defaults to None.
    - name (str, optional): Column name used for data filtering. Defaults to "validated".

    Returns:
    - GridspecLayout: An interactive widget for inspecting and validating data.
    """
    
    cwd = os.path.dirname(os.getcwd())
    image_dir = config["rasters_dir"]
    vector_dir = config["vectors_dir"]
    maxar_dir = config["maxar_dir"]

    if not filename:
        filename = _get_filename(cwd, iso, vector_dir, category, name)
    data = gpd.read_file(filename)
    
    if 'validated' not in data.columns:
        data["validated"] = 0
        
    samples = data[(data['clean'] == 0)]
    if not show_validated:
        samples = samples[(samples['validated'] == 0)]
    samples = samples.iloc[start_index : start_index + (n_rows * n_cols)]
    
    grid = GridspecLayout(n_rows * row_inc + n_rows, n_cols)
    button_dict = {0: ("primary", category), -1: ("warning", "unrecognized")}

    def _add_image(item):
        """
        Loads an image associated with an item and returns it as a widget for display.
    
        Args:
        - item (pandas.Series): Data item containing information about the image.
    
        Returns:
        - Image: A widget displaying the image.
        """
        
        class_dir = os.path.join(cwd, image_dir, maxar_dir, iso, category.lower())
        filepath = os.path.join(class_dir, f"{item[id_col]}.tiff")
        img = open(filepath, "rb").read()
        
        return Image(
            value=img,
            format="png",
            layout=Layout(
                justify_content="center", border="solid", width="auto", height="auto"
            ),
        )

    def _on_button_click(button):
        """
        Updates the button's style and description based on a click event.
    
        Args:
        - button (Button): The button clicked by the user.
    
        Returns:
        - None
        """
        
        index = int(button.description.split(" ")[0])
        item = data.iloc[index]

        change_value = -1
        if item["validated"] == -1:
            change_value = 0
        button_style, category = button_dict[change_value]

        button.button_style = button_style
        button.description = f"{item.name} {category.upper()}"

        data.loc[index, "validated"] = change_value
        data.to_file(filename, driver="GeoJSON")

    def _create_button(item):
        """
        Creates a Button widget based on the item's properties.
    
        Args:
        - item (DataFrame): The item from the dataset.
    
        Returns:
        - Button: A Button widget with specified properties.
        """
        
        val = item["validated"]
        button_style, category = button_dict[val]
        description = f"{item.name} {category.upper()}"

        return Button(
            description=description,
            button_style=button_style,
            layout=Layout(
                justify_content="center", border="solid", width="auto", height="10"
            ),
        )

    row_index, col_index = 0, 0
    for index, item in samples.iterrows():
        grid[row_index : row_index + row_inc, col_index] = _add_image(item)
        button = _create_button(item)
        button.on_click(_on_button_click)
        grid[row_index + row_inc, col_index] = button

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
    filename=None,
    random=False,
    id_col="UID",
    name_col="name",
    figsize=(15, 15)
):
    """
    Visualizes image samples associated with geographic data for inspection.

    Args:
    - config (dict): Configuration settings.
    - iso (str): ISO code for the country.
    - category (str): Category of the data.
    - n_rows (int, optional): Number of rows in the visualization grid. Defaults to 4.
    - n_cols (int, optional): Number of columns in the visualization grid. Defaults to 4.
    - start_index (int, optional): Starting index for data sampling. Defaults to 0.
    - figsize (tuple, optional): Size of the figure. Defaults to (15, 15).
    - filename (str, optional): File name to load the data. Defaults to None.
    - name (str, optional): Column name to consider for data filtering. Defaults to "validated".

    Returns:
    - None
    """
    
    cwd = os.path.dirname(os.getcwd())
    image_dir = config["rasters_dir"]
    vector_dir = config["vectors_dir"]
    maxar_dir = config["maxar_dir"]

    if not filename:
        filename = _get_filename(cwd, iso, vector_dir, category, "clean")

    # Load geographic data from the file
    data = gpd.read_file(filename)
    if random:
        data = data.sample(frac=1.0)
    if "clean" in data.columns:
        data = data[data["clean"] == 0]
    if "validated" in data.columns:
        data = data[data["validated"] == 0]
    logging.info(f"Data dimensions: {data.shape}")

    # Create a grid of subplots for visualizing images
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    samples = data.iloc[start_index : start_index + (n_rows * n_cols)]
    row_index, col_index = 0, 0

    # Iterate over the samples to display associated images
    for idx, item in samples.iterrows():
        class_dir = os.path.join(cwd, image_dir, maxar_dir, iso, category.lower())
        filepath = os.path.join(class_dir, f"{item[id_col]}.tiff")

        # Open and display the image on the subplot
        image = rio.open(filepath)
        show(image, ax=axes[row_index, col_index])
        axes[row_index, col_index].tick_params(
            left=False, bottom=False, labelleft=False, labelbottom=False
        )
        axes[row_index, col_index].set_axis_off()
        axes[row_index, col_index].set_title(
            f"Index: {idx}\n{item[id_col]}\n{item[name_col]}", fontdict={"fontsize": 9}
        )

        col_index += 1
        if col_index >= n_cols:
            row_index += 1
            col_index = 0
        if row_index >= n_rows:
            break
