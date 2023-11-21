import ee
import wxee
import folium
import xarray as xr
import eeconvert as eec
import data_utils


def _initialize_gee():
    """
    Authenticates and initializes Google Earth Engine (GEE) for usage.
    This function prompts authentication if necessary and initializes the Earth Engine API.
    """
    
    ee.Authenticate()
    ee.Initialize()


def generate_gee_images(config, folder, iso_codes=None, layer="ghsl"):
    """
    Generates Google Earth Engine (GEE) images for specified ISO codes.

    Args:
    - config (dict): Configuration settings.
    - folder (str): Name of the folder to save the generated images.
    - iso_codes (list, optional): List of ISO codes to generate images for. Defaults to None.
    - layer (str, optional): Layer name for GEE images. Defaults to "ghsl".

    Returns:
    - None
    """
    
    if not iso_codes: 
        iso_codes = config["iso_codes"]
        
    for iso_code in (pbar := data_utils._create_progress_bar(iso_codes)):
        pbar.set_description(f"Processing {iso_code}")
        geoboundary = data_utils._get_geoboundaries(config, iso_code)
        image, region = generate_gee_image(geoboundary, layer=layer)

        filename = f"{iso_code}_{layer}"
        export_image(image, filename, mode='gdrive', region=region, folder=folder)


def generate_gee_image(data, layer="ghsl"):
    """
    Generates an image from Google Earth Engine based on provided data.

    Args:
    - data (GeoDataFrame): Geospatial data used to generate the image.
    - layer (str, optional): Layer name for GEE images. Defaults to "ghsl".

    Returns:
    - tuple: Tuple containing the generated image and its corresponding geometry.
    """
    
    geometry = eec.gdfToFc(data.dissolve())
    if layer == "ghsl":
        image = ee.Image("JRC/GHSL/P2023A/GHS_BUILT_C/2018")
        image = image.select("built_characteristics").clip(geometry)
        
    return image, geometry


def export_image(image, filename, mode='local', region=None, folder=None):
    """
    Exports an image to Google Drive or local storage.

    Args:
    - image: The image to be exported (Google Earth Engine Image).
    - filename (str): Name of the exported file.
    - mode (str, optional): Export mode, either 'gdrive' for Google Drive or 'local' for local storage. Defaults to 'local'.
    - region: The region to be exported (optional).
    - folder (str, optional): Folder name for Google Drive export. Defaults to None.

    Returns:
    - object: Task if exporting to Google Drive, file path if exporting locally.
    """

    if mode == "gdrive":
        task = ee.batch.Export.image.toDrive(
          image=image,
          driveFolder=folder,
          scale=10,
          region=region.geometry(),
          description=filename,
          fileFormat='GeoTIFF',
          crs='EPSG:4326',
          maxPixels=900000000000
        )
        task.start()
        return task

    elif mode == "local":
        file = image.wx.to_tif(
            out_dir="./",
            description=filename,
            region=region.geometry(),
            scale=10,
            crs="EPSG:4326",
            progress=False,
        )
        return file



    
