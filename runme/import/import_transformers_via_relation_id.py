import argparse
import subprocess
import time
from subprocess import CalledProcessError

import requests

import src.database.database_constructor
from src.data_import.import_transformers import (
    EPSG, OVERPASS_URL, RELATION_ID, fetch_trafos,
    get_trafos_processed_3035_geojson_path, get_trafos_processed_geojson_path,
    process_trafos)


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

    # Convert the GeoJSON file to EPSG and write to a new file
    subprocess.run(
        [
            "ogr2ogr",
            "-f", "GeoJSON",
            "-s_srs", f"EPSG:{EPSG}",
            "-t_srs", f"EPSG:{EPSG}",
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
    sgc = src.database.database_constructor.DatabaseConstructor()
    try:
        sgc.ogr_to_db(trafo_dict, skip_failures=True)
    except CalledProcessError as e:
        print("An error occurred when importing data into database:", e)
        print("\nMost likely cause is data already existing in database. Try the --ignore-existing flag.")
        exit(1)

    print("--- %s seconds ---" % (time.time() - start_time))


def handle_user_input() -> int:
    parser = argparse.ArgumentParser(
        prog="import_transformers_via_relation_id",
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
    response = requests.get(
        OVERPASS_URL, params={
            'data': f"[out:json]; relation({relation_id}); out tags;"})
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
