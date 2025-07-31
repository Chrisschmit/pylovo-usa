import logging
import sys
from pathlib import Path
from typing import Optional

import osm2geojson
import requests


def create_logger(
    name: str, log_level: int = logging.INFO, log_file: Optional[str] = None
) -> logging.Logger:
    """Create a configured logger instance.

    Args:
        name (str): The name for the logger, typically __name__ of the
            calling module.
        log_level (int): The minimum logging level to be processed (e.g.,
            logging.INFO).
        log_file (Optional[str]): Path to the log file. If provided, logs
            will also be written to this file.

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Always add a console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Add a file handler if a log file is specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def simultaneousPeakLoad(buildings_df, consumer_cat_df, vertice_ids):
    # Calculates the simultaneous peak load of buildings with given vertice ids
    subset_df = buildings_df[buildings_df['connection_point'].isin(
        vertice_ids)]
    print(f"{len(subset_df)} buildings are given.")
    # print(subset_df)
    occurring_categories = (['SFH', 'MFH', 'AB', 'TH'], [
                            'Commercial'], ['Public'], ['Industrial'])

    # Sim loads from each category to dictionary
    category_load_dict = {}
    for cat in occurring_categories:
        # Aggregate total installed power from the category cat
        installed_power = subset_df[subset_df['type'].isin(
            cat)]["peak_load_in_kw"].values.sum()  # n*P_0
        # units per buildings amount from cat
        load_count = subset_df[subset_df['type'].isin(
            cat)]['houses_per_building'].values.sum()
        if load_count == 0:
            continue

        sim_factor = consumer_cat_df.loc[cat[0]]['sim_factor']  # g_inf

        # Calculate simultaneous load (Kerber.2011) Gl. 3.2 - S. 23
        sim_load = oneSimultaneousLoad(installed_power, load_count, sim_factor)
        category_load_dict[cat[0]] = sim_load

    # print(category_load_dict)
    # Calculate total sim load (Kiefer S. 142)
    total_sim_load = sum(category_load_dict.values())
    print(f"Total sim load: {total_sim_load}")

    return total_sim_load


def oneSimultaneousLoad(installed_power, load_count, sim_factor):
    # calculation of the simultaneaous load of multiple consumers of the same
    # kind (public, commercial or residential)
    if isinstance(load_count, int):
        if load_count == 0:
            return 0

    sim_load = installed_power * \
        (sim_factor + (1 - sim_factor) * (load_count ** (-3 / 4)))

    return sim_load


def osmjson_to_geojson(osmjson: dict[str, str]) -> dict[str, str]:
    """Convert JSON dict received from overpass api to GeoJSON dictionary.

    Args:
        osmjson: JSON dictionary received from overpass api

    Returns:
        dict: GeoJSON representation of osmjson

    """
    geojson = osm2geojson.json2geojson(osmjson)

    # put attributes in "tags" directly into "properties"
    for feature in geojson['features']:
        if "tags" in feature["properties"]:
            feature["properties"].update(feature["properties"].pop("tags"))

    return geojson


def query_overpass_for_geojson(
        overpass_url: str, query: str) -> dict[str, str]:
    """Execute an overpass turbo query and convert results to GeoJSON.

    Args:
        overpass_url: Overpass API URL
        query: Query string

    Returns:
        dict: GeoJSON representation of overpass results

    """
    # call api for data
    response = requests.get(overpass_url, params={'data': query})
    response.raise_for_status()

    # convert JSON data to GeoJSON format
    osmjson = response.json()
    geojson = osmjson_to_geojson(osmjson)

    return geojson
