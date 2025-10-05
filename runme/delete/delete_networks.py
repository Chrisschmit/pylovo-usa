from src.config_loader import REGION
from src.database.database_client import DatabaseClient
from src.grid_generator import GridGenerator

# determine regional_identifier from active REGION configuration
dbc = DatabaseClient()
regional_identifier = dbc.get_regional_identifier_from_region(REGION)
version_id = "1.0"

# delete networks
gg = GridGenerator(regional_identifier=regional_identifier)
gg.dbc.delete_regional_identifier_from_all_tables(regional_identifier, version_id)
gg.dbc.drop_temp_tables()
