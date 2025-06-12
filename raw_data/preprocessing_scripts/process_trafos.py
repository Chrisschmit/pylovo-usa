import time
import os
from subprocess import CalledProcessError

import geopandas as gpd
import numpy as np
import pandas as pd
import json
import subprocess
import argparse
import requests

from src import SyngridDatabaseConstructor
from src.utils import query_overpass_for_geojson

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
RELATION_ID_BASE = 3600000000  # do not change

# change relation id to desired location according to docs
RELATION_ID = 2145268
# Bavaria --> 2145268

AREA_THRESHOLD = 60
MIN_DISTANCE_BETWEEN_TRAFOS = 8
VOLTAGE_THRESHOLD = 110000
EPSG = 32633

SUBSTATIONS_QUERY_PATH = os.path.join(".", "raw_data", "transformer_data", "overpass_queries", "substations_query.txt")
SHOPPING_MALL_QUERY_PATH = os.path.join(".", "raw_data", "transformer_data", "overpass_queries", "shopping_mall_query.txt")


def main(relation_id: int) -> None:
    """Fetch transformers from Overpass API, process the fetched data, and finally load them into the database.

    Args:
        relation_id (int): relation ID of the area of interest
        ignore_existing (bool): ignore existing transformers.

    """
    # timing of the script
    start_time = time.time()

    print("Fetching transformers...")
    fetch_trafos(relation_id)
    print("Processing transformers...")
    process_trafos(relation_id)

    in_file = get_trafos_processed_geojson_path(relation_id)
    out_file = get_trafos_processed_3035_geojson_path(relation_id)

    # Convert the GeoJSON file to EPSG:3035 and write to a new file
    subprocess.run(
        [
            "ogr2ogr",
            "-f", "GeoJSON",
            "-s_srs", f"EPSG:{EPSG}",
            "-t_srs", "EPSG:3035",
            out_file,  # output
            in_file  # input
        ],
        shell=False
    )

    # write new data to transformers table
    print("Loading transformers into database...")
    trafo_dict = [
        {
            "path": out_file,
            "table_name": "transformers"
        }
    ]
    sgc = SyngridDatabaseConstructor()
    try:
        sgc.ogr_to_db(trafo_dict, skip_failures=True)
    except CalledProcessError as e:
        print("An error occurred when importing data into database:", e)
        print("\nMost likely cause is data already existing in database. Try the --ignore-existing flag.")
        exit(1)


    print("--- %s seconds ---" % (time.time() - start_time))


def get_substations_geojson_path(relation_id: int) -> str:
    return os.path.join(".", "raw_data", "transformer_data", "fetched_trafos", f"{relation_id}_substations.geojson")


def get_shopping_mall_geojson_path(relation_id: int) -> str:
    return os.path.join(".", "raw_data", "transformer_data", "fetched_trafos", f"{relation_id}_shopping_mall.geojson")


def get_trafos_processed_geojson_path(relation_id: int) -> str:
    return os.path.join(".", "raw_data", "transformer_data", "processed_trafos", f"{relation_id}_trafos_processed.geojson")


def get_trafos_processed_3035_geojson_path(relation_id: int) -> str:
    return os.path.join(".", "raw_data", "transformer_data", "processed_trafos", f"{relation_id}_trafos_processed_3035.geojson")


def fetch_trafos(relation_id: int) -> None:
    """Fetch trafos from OSM bound by area defined by given relation_id

    Args:
        relation_id (int): relation_id of area to fetch trafos from

    """
    with open(SUBSTATIONS_QUERY_PATH, "r") as f:
        overpass_query_substations = f.read()
    with open(SHOPPING_MALL_QUERY_PATH, "r") as f:
        overpass_query_mall = f.read()

    # set chosen relation_id in query
    overpass_query_substations = overpass_query_substations.replace("$relation_id$", str(RELATION_ID_BASE + relation_id))
    overpass_query_mall = overpass_query_mall.replace("$relation_id$", str(RELATION_ID_BASE + relation_id))

    geojson_bayern = query_overpass_for_geojson(OVERPASS_URL, overpass_query_substations)
    geojson_mall = query_overpass_for_geojson(OVERPASS_URL, overpass_query_mall)

    # save resulting GeoJSON-s
    with open(get_substations_geojson_path(relation_id), "w") as f:
        json.dump(geojson_bayern, f, indent=2)
    with open(get_shopping_mall_geojson_path(relation_id), "w") as f:
        json.dump(geojson_mall, f, indent=2)


def process_trafos(relation_id: int) -> None:
    """Process trafo data and output it as GeoJSON into output_geojson.

    Args:
        relation_id (int): relation ID of the area of interest that specifies which file is used as processing input

    """
    # Import geojson of substations/trafos. Trafos of "Deutsche Bahn" and historic trafos have already been deleted.
    gdf_substations = gpd.read_file(get_substations_geojson_path(relation_id))
    print('start:')
    print(len(gdf_substations))

    # the geodata imported from the geojson is imported in the CRS (Coordinate Reference System) WGS84, "EPSG","4326".
    # It has lan and lat values. For area calculations it needs to be converted into a planar projection.
    gdf_substations = gdf_substations.to_crs(EPSG)

    # 1. eliminate all trafos that lay within other trafo geometries
    gdf_substations['geom_type'] = gdf_substations.geom_type
    gdf_points = gdf_substations.groupby('geom_type').get_group('Point')
    gdf_polygon = gdf_substations.groupby('geom_type').get_group('Polygon')
    union_of_polygons = gdf_polygon.geometry.unary_union
    gdf_points['within_poly'] = gdf_points.within(union_of_polygons)
    gdf_substations.drop(gdf_points[gdf_points['within_poly'] == True].index, inplace=True)
    print('After step 1:')
    print(len(gdf_substations))

    # 2. eliminate all Umspannungswerke area larger than threshold and all transformers that are tagged within that area
    gdf_substations['area'] = gdf_substations.area
    gdf_substations.drop(gdf_substations[gdf_substations['area'] >= AREA_THRESHOLD].index, inplace=True)
    print('After step 2:')
    print(len(gdf_substations))

    # 3. Delete all high voltage transformers
    # replace any values that cannot be converted to float by nan
    gdf_substations['voltage'] = (
        gdf_substations['voltage'].fillna(1).apply(lambda x: pd.to_numeric(x, errors='coerce')))
    gdf_substations['voltage'] = gdf_substations['voltage'].astype(float)
    gdf_substations.drop(gdf_substations[gdf_substations['voltage'] >= VOLTAGE_THRESHOLD].index, inplace=True)
    print('After step 3:')
    print(len(gdf_substations))

    # 4. how many transformers are there in a radius of 5, 10, 15 m of each other
    gdf_substations['centroid'] = gdf_substations.centroid
    distance_matrix = gdf_substations['centroid'].apply(lambda c: gdf_substations['centroid'].distance(c))
    # set lower triangle of matrix to nan
    distance_matrix = distance_matrix.where(np.triu(np.ones(distance_matrix.shape)).astype(bool))
    # set diagonal to nan
    np.fill_diagonal(distance_matrix.values, float('nan'))
    distance_matrix = distance_matrix[(distance_matrix < MIN_DISTANCE_BETWEEN_TRAFOS).any(axis=1)]
    index_list = list(distance_matrix.index)
    gdf_substations.drop(index=index_list, inplace=True)
    print('After step 4:')
    print(len(gdf_substations))

    # 5. how many trafos are there in / next to mall?
    gdf_shopping = gpd.read_file(get_shopping_mall_geojson_path(relation_id))
    gdf_shopping = gdf_shopping.to_crs(EPSG)
    union_of_shopping = gdf_shopping.geometry.unary_union
    gdf_substations['within_shopping'] = gdf_substations.within(union_of_shopping)
    gdf_substations.drop(gdf_substations[gdf_substations['within_shopping']].index, inplace=True)
    print('After step 5:')
    print(len(gdf_substations))

    # drop geometry that can be of type polygon and point, use centroid as new geometry instead
    # drop tag columns
    gdf_substations.drop('geometry', axis=1, inplace=True)
    gdf_substations.rename(columns={"centroid": "geometry"}, inplace=True)
    gdf_substations.dropna(axis='columns', inplace=True)

    # transform column id into osm_id as is used for buildings
    gdf_substations['id'] = gdf_substations.apply(lambda row: f"{row['type']}/{row['id']}", axis=1)
    gdf_substations.rename(columns={"id": "osm_id"}, inplace=True)
    if "@id" in gdf_substations:
        gdf_substations.drop('@id', axis=1, inplace=True)

    # export geojson
    gdf_substations.to_file(get_trafos_processed_geojson_path(relation_id), driver='GeoJSON')


def handle_user_input() -> int:
    parser = argparse.ArgumentParser(
        prog="process_trafos",
        description="Fetch transformers from Overpass API, process the fetched data,"
                    "and finally load them into the database."
    )
    parser.add_argument(
        "--relation-id",
        type=int,
        default=RELATION_ID,
        help="specify the relation ID of the area the script should work with",
        required=False
    )
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="do not prompt for confirmation",
        required=False
    )

    args = parser.parse_args()
    relation_id = args.relation_id
    skip_confirmation = args.yes

    print(f"Selected relation ID: {relation_id}")

    # get the hint of area and print it to the console
    response = requests.get(OVERPASS_URL, params={'data': f"[out:json]; relation({relation_id}); out tags;"})
    response.raise_for_status()
    area_hint = None
    tags = response.json()["elements"][0]["tags"]
    if "name" in tags:
        area_hint = tags["name"]
    elif "note" in tags:  # useful for postal codes
        area_hint = tags["note"]

    if area_hint is not None:
        print(f"Corresponding area: {area_hint}")
    else:
        print(f"Corresponding area tags: {tags}")

    # prompt user to make sure they selected the right relation ID
    if not skip_confirmation:
        answer = input("Do you want to continue? [Y/n] ").strip().lower()
        if answer not in ("y", "yes", ""):
            print("Aborted.")
            exit(1)
        print("Continuing...")
    return relation_id


if __name__ == "__main__":
    rel_id = handle_user_input()
    main(rel_id)