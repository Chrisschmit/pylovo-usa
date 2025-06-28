from src.grid_generator import GridGenerator

# select plz and version you want to delete the networks for
plz = 2501711000
version_id = "1.0"

# delete networks
gg = GridGenerator(plz=plz)
gg.dbc.delete_plz_from_all_tables(plz, version_id)
gg.dbc.drop_temp_tables()
