import ee
import wxee
import folium
import xarray as xr
import eeconvert as eec
import data_utils


def _initialize_gee():
    ee.Authenticate()
    ee.Initialize()


def generate_gee_images(config, folder, iso_codes=None, layer="ghsl"):
    if not iso_codes: iso_codes = config["iso_codes"]
    for iso_code in (pbar := data_utils._create_progress_bar(iso_codes)):
        pbar.set_description(f"Processing {iso_code}")
        geoboundary = data_utils._get_geoboundaries(config, iso_code)
        image, region = generate_gee_image(geoboundary, layer=layer)

        filename = f"{iso_code}_{layer}"
        export_image(image, filename, mode='gdrive', region=region, folder=folder)


def generate_gee_image(data, layer="ghsl"):
    """Generates Image from Google Earth Engine"""
    
    geometry = eec.gdfToFc(data.dissolve())
    if layer == "ghsl":
        image = ee.Image("JRC/GHSL/P2023A/GHS_BUILT_C/2018")
        image = image.select("built_characteristics").clip(geometry)
        
    return image, geometry


def export_image(image, filename, mode='local', region=None, folder=None):
    """Export Image to Google Drive."""

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



    
