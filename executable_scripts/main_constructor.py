"""
This script creates a pylovo database and fills with raw data from referenced files.
Do not use SyngridDatabaseConstructor unless you want to create a new database.
Make sure to have PostGIS 3.4 installed, newer versions lead to errors.
"""

from raw_data.municipal_register.join_regiostar_gemeindeverz import create_municipal_register
from pylovo.SyngridDatabaseConstructor import SyngridDatabaseConstructor


def main():
    ### Create constructor class
    sgc = SyngridDatabaseConstructor()

    ### Create database with predefined table structure
    print("### CREATE ALL TABLES ###")
    sgc.create_table(table_name="all")

    ### Add defined csv raw data from CSV_FILE_LIST to the database
    print("### POPULATE DB WITH CSV RAW DATA ###")
    sgc.csv_to_db()

    ### Add transformer data from geojson to the database
    print("### QUERY TRANSFORMERS AND INSERT THEM INTO DB (~50 min if processing new trafo data) ###")
    sgc.transformers_to_db()

    ### Create table with data from osm
    print("### POPULATE public_2po_4pgr TABLE (~30 min) ###")
    sgc.create_public_2po_table()

    ### Transform these data into our ways table
    print("### PROCESS WAYS AND INSERTING THEM INTO ways TABLE ###")
    sgc.ways_to_db()

    ### Add additional required sql functions to the database
    print("### DUMP NECESSARY FUNCTIONS INTO DB ###")
    sgc.dump_functions()

    ### Create table with entries of all German municipalities and cities
    print("### FILL municipal_register TABLE ###")
    create_municipal_register()

    print("### DONE ###")


if __name__ == "__main__":
    main()