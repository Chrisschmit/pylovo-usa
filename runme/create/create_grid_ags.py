# generate the grid for the multiple PLZ set below
# building data import is included

import time

from src.classification.sampling.sample import get_municipal_register_as_dataframe
from src.data_import.import_buildings import import_buildings_for_multiple_plz
from src.grid_generator import GridGenerator
from src.config_loader import ANALYZE_GRIDS

ags = 9161000

# timing of the script
start_time = time.time()

# get ags info for plz areas
municipal_register = get_municipal_register_as_dataframe()
df_plz_ags = municipal_register[municipal_register['ags'] == ags]
# import buildings and generate grids
import_buildings_for_multiple_plz(sample_plz=df_plz_ags)

df_plz = df_plz_ags[['plz']]
# initialize GridGenerator
gg = GridGenerator()
gg.generate_grid_for_multiple_plz(df_plz=df_plz, analyze_grids=ANALYZE_GRIDS)

# End timing and print results
elapsed_time = time.time() - start_time
minutes, seconds = divmod(elapsed_time, 60)
print(f"--- Elapsed Time: {int(minutes)} minutes and {seconds:.2f} seconds ---")