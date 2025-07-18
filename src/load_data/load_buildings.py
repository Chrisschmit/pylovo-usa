import glob

from src.config_loader import *
from src.database.database_constructor import DatabaseConstructor
from src.grid_generator import GridGenerator


def import_buildings_for_single_plz(gg: GridGenerator):
    """
    Imports building data to the database for a given FIPS code specified in the GridGenerator object.
    FIPS code is added to fips_log table to avoid importing the same building data again.

    :param gg: Grid generator object for querying relevant FIPS code data
    """
    # Retrieve plz from GridGenerator object
    dbc_client = gg.dbc
    plz = gg.plz

    # Check wether building data for this FIPS code is already in the database
    postcode_entry = dbc_client.get_postcode_table_for_plz(plz)
    gg.logger.info(
        f"LV grids will be generated for {postcode_entry.iloc[0]['plz']} "
        f"{postcode_entry.iloc[0]['county_name']}")

    logs = dbc_client.get_fips_log()
    if plz in logs['fips_code'].values:
        gg.logger.info(
            f"Buildings for FIPS code {plz} have already been added to the database.")
        return
    else:
        gg.logger.info(
            f"Buildings for FIPS code {plz} will be added to the database.")

    # Define the path for building shapefiles
    data_path = os.path.abspath(
        os.path.join(
            PROJECT_ROOT,
            "raw_data",
            "imports",
            REGION['STATE'].replace(' ', '_'),
            REGION['COUNTY'].replace(' ', '_'),
            REGION['COUNTY_SUBDIVISION'].replace(' ', '_'),
            "BUILDINGS_OUTPUT",
            "shp"
        ))
    print(data_path)
    shapefiles_pattern = os.path.join(
        data_path, "*.shp")  # Pattern for shapefiles
    print(shapefiles_pattern)

    # Retrieve all matching shapefiles
    files_to_add = glob.glob(shapefiles_pattern, recursive=True)
    print(files_to_add)

    # Handle cases where no matching files are found
    if not files_to_add:
        raise FileNotFoundError(
            f"No shapefiles found for PLZ {plz} in {data_path}")

    # Create a list of dictionaries for ogr_to_db()
    ogr_ls_dict = create_list_of_shp_files(files_to_add)

    # Add building data to the database
    sgc = DatabaseConstructor(dbc_obj=dbc_client)
    sgc.ogr_to_db(ogr_ls_dict, skip_failures=True)

    # adding the added fips_code to the log file
    dbc_client.write_fips_log(int(plz))

    gg.logger.info(
        f"Buildings for FIPS code {plz} have been successfully added to the database.")


def import_buildings_for_multiple_plz(plz_list: list[int]):
    """
    imports building data to db for multiple plz
    """
    # Define the path for building shapefiles
    data_path = os.path.abspath(
        os.path.join(
            PROJECT_ROOT,
            "raw_data",
            "buildings"))
    shapefiles_pattern = os.path.join(
        data_path, "*.shp")  # Pattern for shapefiles

    # retrieving all shape files
    files_list = glob.glob(shapefiles_pattern, recursive=True)

    # get all PLZs that need to be imported for the classification
    plz_to_add = list(set(plz_list))  # dropping duplicates

    # check in fips_log if any plz are already on the database
    gg = GridGenerator(plz=999999)
    dbc_client = gg.dbc
    df_log = dbc_client.get_fips_log()
    log_plz_list = df_log['fips_code'].values.tolist()
    plz_to_add = list(set(plz_to_add).difference(
        log_plz_list))  # dropping already imported plz
    plz_to_add = list(map(str, plz_to_add))

    # creating a list that only contains the files to add
    files_to_add = []
    for file in files_list:
        for plz in plz_to_add:
            if plz in file:
                files_to_add.append(file)
    files_to_add = list(set(files_to_add))  # dropping duplicates

    if files_to_add:
        # define a list of required shapefiles to add to the database for the
        # function scg.ogr_to_db()
        ogr_ls_dict = create_list_of_shp_files(files_to_add)

        # adding the buildings to the database
        sgc = DatabaseConstructor()
        sgc.ogr_to_db(ogr_ls_dict)

        # adding the added ags to the log file
        for plz in plz_to_add:
            dbc_client.write_fips_log(int(plz))


def create_list_of_shp_files(files_to_add):
    """
    Creates a list of dictionaries required for the scg.ogr_to_db() function.

    :param files_to_add: List of shapefile paths to add.
    :return: A list of dictionaries with keys "path" and "table_name".
    """
    ogr_ls_dict = []

    # Process each file path
    for file_path in files_to_add:
        # Determine table_name based on file naming convention
        if "Oth" in file_path:
            table_name = "oth"
        elif "Res" in file_path:
            table_name = "res"
        else:
            raise ValueError(
                f"Shapefile '{file_path}' cannot be assigned to 'res' or 'oth'.")

        ogr_ls_dict.append({"path": file_path, "table_name": table_name})

    # Ensure the list is not empty
    if ogr_ls_dict:
        return ogr_ls_dict
    else:
        raise Exception("No valid shapefiles found for the requested PLZ.")
