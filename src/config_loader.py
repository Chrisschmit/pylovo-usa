import os

import pandas as pd
import yaml
from dotenv import find_dotenv, load_dotenv


def load_yaml_config(filepath: str):
    """Loads a YAML configuration file."""
    abs_path = os.path.join(
        os.path.dirname(
            os.path.abspath(__file__)),
        filepath)

    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Config file not found: {abs_path}")

    with open(abs_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


# Load Project Root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Load all configurations with correct paths
CONFIG_DATA = load_yaml_config("../config/config_data.yaml")
CONFIG_VERSION = load_yaml_config("../config/config_version.yaml")


# Load database connection configuration from CONFIG_DATA
load_dotenv(find_dotenv(), override=True)
DBNAME = os.getenv("DBNAME", CONFIG_DATA["DBNAME"])
DB_SUFFIX = os.getenv("DB_SUFFIX", CONFIG_DATA["DB_SUFFIX"])

# Append suffix to database name if specified
if DB_SUFFIX:
    DBNAME = f"{DBNAME}_{DB_SUFFIX}"
USER = os.getenv("USER", CONFIG_DATA["USER"])
HOST = os.getenv("HOST", CONFIG_DATA["HOST"])
PORT = os.getenv("PORT", CONFIG_DATA["PORT"])
PASSWORD = os.getenv("PASSWORD", CONFIG_DATA["PASSWORD"])
TARGET_SCHEMA = os.getenv("TARGET_SCHEMA", CONFIG_DATA["TARGET_SCHEMA"])

# Assign other variables from CONFIG_DATA
RESULT_DIR = os.path.join(os.getcwd(), "results")
ANALYZE_GRIDS = CONFIG_DATA["ANALYZE_GRIDS"]
SAVE_GRID_FOLDER = CONFIG_DATA["SAVE_GRID_FOLDER"]
LOG_LEVEL = CONFIG_DATA["LOG_LEVEL"]
LOG_FILE = CONFIG_DATA["LOG_FILE"]
EPSG = CONFIG_DATA["EPSG"]
CSV_FILE_LIST = [
    {"path": os.path.join("raw_data", "equipment_data.csv"),
     "table_name": "equipment_data"},
    {"path": os.path.join("raw_data", "postcode.csv"),
     "table_name": "postcode"},
]

# Assign all variables from CONFIG_VERSION
VERSION_ID = CONFIG_VERSION["VERSION_ID"]
VERSION_COMMENT = CONFIG_VERSION["VERSION_COMMENT"]
PLOT_COLOR_DICT = CONFIG_VERSION["PLOT_COLOR_DICT"]
CONNECTION_AVAILABLE_CABLES = CONFIG_VERSION["CONNECTION_AVAILABLE_CABLES"]
CABLE_COST_DICT = CONFIG_VERSION["CABLE_COST_DICT"]
SIM_FACTOR = CONFIG_VERSION["SIM_FACTOR"]
PEAK_LOAD_HOUSEHOLD = CONFIG_VERSION["PEAK_LOAD_HOUSEHOLD"]
CONSUMER_CATEGORIES = pd.DataFrame(CONFIG_VERSION["CONSUMER_CATEGORIES"])
LARGE_COMPONENT_LOWER_BOUND = CONFIG_VERSION["LARGE_COMPONENT_LOWER_BOUND"]
LARGE_COMPONENT_DIVIDER = CONFIG_VERSION["LARGE_COMPONENT_DIVIDER"]
POWER_FACTOR = CONFIG_VERSION["POWER_FACTOR"]
RURAL_LD = CONFIG_VERSION["RURAL_LD"]
URBAN_LD = CONFIG_VERSION["URBAN_LD"]
AVG_APARTMENT_AREA = CONFIG_VERSION["AVG_APARTMENT_AREA"]
VN = CONFIG_VERSION["VN"]
V_BAND_LOW = CONFIG_VERSION["V_BAND_LOW"]
V_BAND_HIGH = CONFIG_VERSION["V_BAND_HIGH"]
LV_THRESHOLD_KW = CONFIG_VERSION["LV_THRESHOLD_KW"]
MV_THRESHOLD_KW = CONFIG_VERSION["MV_THRESHOLD_KW"]

# Assign all Data Import variables
REGION = CONFIG_DATA["REGION"]
OUTPUT_DIR = CONFIG_DATA["OUTPUT_DIR"]
INPUT_DATA = CONFIG_DATA["INPUT_DATA"]

# Building type classification thresholds
BUILDING_TYPE_THRESHOLDS = CONFIG_DATA["BUILDING_TYPE_THRESHOLDS"]
