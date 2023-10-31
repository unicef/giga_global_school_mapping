import os
import pandas as pd
import geopandas as gpd

import duckdb
import geojson
import overpass
import swifter
import data_utils

from tqdm.notebook import tqdm
from country_bounding_boxes import country_subunits_by_iso_code


def _query_osm(iso_code, out_file, query):
    """Queries OSM for a given ISO code."""

    api = overpass.API(timeout=1500)
    osm_query = f"""
        area["ISO3166-1"="{iso_code}"][admin_level=2];
        ({query});
        out center;
    """
    data = api.get(osm_query, verbosity="geom")
    with open(out_file, "w") as file:
        geojson.dump(data, file)

    data = gpd.read_file(out_file)
    return data


def download_osm(iso_codes, out_dir, out_file="osm.geojson", category="SCHOOL"):
    """Downloads OSM POIs based on a list of ISO codes."""

    out_dir = data_utils._makedir(out_dir)
    data_config = data_utils._load_data_config()
    osm_file = os.path.join(os.path.dirname(out_dir), out_file)

    url = data_config["ISO_REGIONAL_CODES"]
    codes = pd.read_csv(url)

    keywords = data_config[category]
    query = "".join(
        [
        f"""node["{key}"~"^({"|".join(values)})"](area);
        way["{key}"~"^({"|".join(values)})"](area);
        rel["{key}"~"^({"|".join(values)})"](area);
        """
            for key, values in keywords.items()
        ]
    )

    data = []
    pbar = tqdm(enumerate(iso_codes), total=len(iso_codes))
    for _, iso_code in pbar:
        pbar.set_description(f"Processing {iso_code}")
        filename = f"{iso_code}_osm.geojson"
        out_subfile = os.path.join(out_dir, filename)

        if not os.path.exists(out_subfile):
            alpha_2 = codes.query(f"`alpha-3` == '{iso_code}'")["alpha-2"].values[0]
            _query_osm(alpha_2, out_subfile, query)

        subdata = gpd.read_file(out_subfile).reset_index(drop=True)
        if len(subdata) > 0:
            subdata = data_utils._prepare_data(
                subdata,
                iso_code,
                category,
                source="OSM",
                columns=data_config["COLUMNS"],
                out_file=out_subfile,
            )
            data.append(subdata)

    # Combine data and save to file
    data = data_utils._concat_data(data, osm_file)
    return data


def _query_overture(iso_code, out_file, query):
    """Queries Overture Map for a given ISO code."""

    # Connect to DuckDB
    db = duckdb.connect()
    db.execute("INSTALL spatial")
    db.execute("INSTALL httpfs")
    db.execute(
        """
        LOAD spatial;
        LOAD httpfs;
        SET s3_region='us-west-2';
    """
    )

    # Fetch country bounding box and execute Overture query
    data_config = data_utils._load_data_config()
    url = data_config["OVERTURE_URL"]
    bbox = [c.bbox for c in country_subunits_by_iso_code(iso_code)][0]
    overture_query = f"""
        COPY(
            select
            JSON(names) AS names,
            ST_GeomFromWkb(geometry) AS geometry
        from
            read_parquet('{url}')
        where
            bbox.minx > {bbox[0]}
            and bbox.miny > {bbox[1]}
            and bbox.maxx < {bbox[2]}
            and bbox.maxy < {bbox[3]}
            and (
            {query}
            )
        ) TO '{out_file}'
        WITH (FORMAT GDAL, DRIVER 'GeoJSON')
    """
    db.execute(overture_query).fetchall()

    # Intersect data with the country's geoboundaries
    data = gpd.read_file(out_file)
    return data


def download_overture(
    iso_codes, out_dir, out_file="overture.geojson", category="SCHOOL", exclude=None
):
    """Downloads Overture Map POIs based on a list of ISO codes."""

    # Fetch school keywords and generate keywords query
    out_dir = data_utils._makedir(out_dir)
    overture_file = os.path.join(os.path.dirname(out_dir), out_file)
    data_config = data_utils._load_data_config()
    keywords = data_config[category]

    query = " or ".join(
        [
            f"""UPPER(names) LIKE '%{keyword.replace('_', ' ').upper()}%'
        """
            for key, values in keywords.items()
            for keyword in values
        ]
    )

    if exclude is not None:
        exclude_keywords = data_config[exclude]
        exclude_query = " and ".join(
            [
                f"""UPPER(names) NOT LIKE '%{keyword.replace('_', ' ').upper()}%'
            """
                for key, values in exclude_keywords.items()
                for keyword in values
            ]
        )
        query = f""" ({query}) and ({exclude_query})"""

    data = []
    pbar = tqdm(enumerate(iso_codes), total=len(iso_codes))
    for _, iso_code in pbar:
        pbar.set_description(f"Processing {iso_code}")
        filename = f"{iso_code}_overture.geojson"
        out_subfile = os.path.join(out_dir, filename)

        # Fetch Overture data
        if not os.path.exists(out_subfile):
            _query_overture(iso_code, out_subfile, query)

        # Post-process data
        subdata = gpd.read_file(out_subfile).reset_index(drop=True)
        if len(subdata) > 0:
            if "names" in subdata.columns:
                geoboundary = _get_geoboundaries(iso_code)
                subdata["name"] = subdata["names"].apply(
                    lambda x: x["common"][0]["value"]
                )
                subdata = gpd.sjoin(subdata, geoboundary, predicate="within")

            subdata = data_utils._prepare_data(
                subdata,
                iso_code,
                category,
                source="OVERTURE",
                columns=data_config["COLUMNS"],
                out_file=out_subfile,
            )
            data.append(subdata)

    # Combine datasets
    data = data_utils._concat_data(data, overture_file)
    return data


def load_unicef(data_dir, out_file, category="SCHOOL"):
    """Combines the UNICEF/Giga datasets into a single CSV file."""

    # Load school data files
    data_dir = data_utils._makedir(data_dir)
    data_config = data_utils._load_data_config()
    files = next(os.walk(data_dir), (None, None, []))[2]
    print(f"Number of CSV files: {len(files)}")

    data = []
    pbar = tqdm(enumerate(files), total=len(files))
    for _, file in pbar:
        # Read lat-lon file
        filename = os.path.join(data_dir, file)
        subdata = pd.read_csv(filename, low_memory=False).reset_index(drop=True)

        # Get iso, country name, region, and sub-region
        iso_code = file.split("_")[0]  # Get ISO code
        pbar.set_description(f"Processing {iso_code}")

        subdata["geometry"] = gpd.GeoSeries.from_xy(subdata["lon"], subdata["lat"])
        subdata = data_utils._prepare_data(
            subdata,
            iso_code,
            category,
            source="UNICEF",
            columns=data_config["COLUMNS"]+["giga_id_school"]
        )

        # Append to data
        data.append(subdata)

    # Combine dataset
    out_dir = os.path.dirname(data_dir)
    filename = os.path.join(out_dir, out_file)
    data = data_utils._concat_data(data, filename)
    return data
