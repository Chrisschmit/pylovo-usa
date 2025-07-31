from src.grid_generator import GridGenerator

regional_identifier = 1

gg = GridGenerator(regional_identifier=regional_identifier)
dbc_client = gg.dbc
dbc_client.delete_transformers()
