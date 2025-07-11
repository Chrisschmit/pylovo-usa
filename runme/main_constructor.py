"""
This script creates a src database and fills with raw data from referenced files.
Do not use DatabaseConstructor class unless you want to create a new database.
"""

from src import utils
from src.config_loader import LOG_LEVEL
from src.database.database_constructor import DatabaseConstructor

logger = utils.create_logger(
    name="main_constructor",
    log_file="log.txt",
    log_level=LOG_LEVEL)


def main():
    # Create constructor class
    sgc = DatabaseConstructor()

    # Drop all tables
    logger.info("### DEV: DROP ALL TABLES ###")
    sgc.drop_all_tables()

    # Create database with predefined table structure
    logger.info("### CREATE ALL TABLES ###")
    sgc.create_table(table_name="all")

    # Add defined csv raw data from CSV_FILE_LIST to the database
    logger.info("### POPULATE DB WITH CSV RAW DATA ###")
    sgc.csv_to_db()

    # Add transformer data from geojson to the database
    logger.info(
        "### QUERY TRANSFORMERS AND INSERT THEM INTO DB (~50 min if processing new trafo data) ###")
    sgc.transformers_to_db()

    # Create table with data from osm
    logger.info("### POPULATE public_2po_4pgr TABLE (~30 min) ###")
    sgc.create_public_2po_table()

    # Transform these data into our ways table
    logger.info("### PROCESS WAYS AND INSERTING THEM INTO ways TABLE ###")
    sgc.ways_to_db()

    # Add additional required sql functions to the database
    logger.info("### DUMP NECESSARY FUNCTIONS INTO DB ###")
    sgc.dump_functions()

    logger.info("### DONE ###")


if __name__ == "__main__":
    main()
