from src.grid_generator import GridGenerator

# select version you want to delete entirely
version_id = "1.0"

# delete networks
# just a dummy plz for the initialization of the class
gg = GridGenerator(plz="91301")
gg.dbc.delete_version_from_all_tables(version_id=version_id)
