from src.grid_generator import GridGenerator

# select regional_identifier and version you want to delete the networks for
# regional_identifier = 2501711000
regional_identifier = 1
version_id = "1.0"

# delete networks
gg = GridGenerator(regional_identifier=regional_identifier)
gg.dbc.delete_regional_identifier_from_all_tables(
    regional_identifier, version_id)
gg.dbc.drop_temp_tables()
