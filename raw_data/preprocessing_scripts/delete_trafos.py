from src.GridGenerator import GridGenerator

plz = "91301"  # forchheim

gg = GridGenerator(plz=plz)
dbc_client = gg.dbc
dbc_client.delete_transformers()
