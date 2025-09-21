import os

from src.config_loader import PROJECT_ROOT, REGION
from src.database.database_constructor import DatabaseConstructor
from src.grid_generator import GridGenerator


def import_transformers_for_single_regional_identifier(gg: GridGenerator):
    """
    Imports transformer data to the database for a given FIPS code specified in the GridGenerator object.
    Checks for existing transformers by osm_id to avoid duplicates.

    :param gg: Grid generator object for querying relevant FIPS code data
    """
    # Retrieve regional_identifier from GridGenerator object
    dbc_client = gg.dbc
    regional_identifier = gg.regional_identifier

    # Retrieve postcode entry for logging
    postcode_entry = dbc_client.get_postcode_table_for_regional_identifier(
        regional_identifier)
    gg.logger.info(
        f"Loading transformers for {
            postcode_entry.iloc[0]['regional_identifier']} "
        f"{postcode_entry.iloc[0]['county_name']}")

    # Define the path for transformer GeoJSON file
    data_path = os.path.abspath(
        os.path.join(
            PROJECT_ROOT,
            "raw_data",
            "imports",
            REGION['STATE'].replace(' ', '_'),
            REGION['COUNTY'].replace(' ', '_'),
            REGION['COUNTY_SUBDIVISION'].replace(' ', '_'),
            "OSM"
        ))

    # Look for power.geojson file
    power_geojson_path = os.path.join(data_path, "power.geojson")

    # Check if file exists
    if not os.path.isfile(power_geojson_path):
        gg.logger.warning(
            f"Transformer file not found at {power_geojson_path}. "
            f"Skipping transformer import.")
        return

    gg.logger.info(f"Found transformer file at {power_geojson_path}")

    # Check for existing transformers in database
    try:
        with dbc_client.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM transformers")
            existing_count = cur.fetchone()[0]
            if existing_count > 0:
                gg.logger.info(
                    f"Found {existing_count} existing transformers in database. "
                    f"New transformers will be added if they don't already exist.")
    except Exception as e:
        gg.logger.debug(f"Could not check existing transformers: {e}")

    # Add transformer data to the database using the simplified method
    sgc = DatabaseConstructor(dbc_obj=dbc_client)
    sgc.transformers_to_db(power_geojson_path)

    gg.logger.info(
        f"Transformers for FIPS code {regional_identifier} have been successfully processed.")


def import_transformers_for_multiple_regional_identifiers(
        regional_identifier_list: list[int]):
    """
    Imports transformer data to db for multiple regional_identifiers.

    :param regional_identifier_list: List of FIPS codes to process
    """
    for regional_identifier in regional_identifier_list:
        gg = GridGenerator(regional_identifier=regional_identifier)
        import_transformers_for_single_regional_identifier(gg)
