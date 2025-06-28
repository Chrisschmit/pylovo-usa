# generate the grid for the multiple PLZ set below
# building data import is included

import time

import pandas as pd

from src.config_loader import ANALYZE_GRIDS
from src.data_import.import_buildings import import_buildings_for_multiple_plz
from src.grid_generator import GridGenerator

# start timing the script
start_time = time.time()

# Enter the fips codes for which the geodata is exported
plz_list = [2501711000, 2501713135]

# import buildings and generate grids
import_buildings_for_multiple_plz(plz_list=plz_list)

# initialize GridGenerator
gg = GridGenerator()
df_plz = pd.DataFrame(
    list(map(str, plz_list)), columns=['fips_code'])
gg.generate_grid_for_multiple_plz(
    df_plz=df_plz,
    analyze_grids=ANALYZE_GRIDS)

# end timing and print results
elapsed_time = time.time() - start_time
minutes, seconds = divmod(elapsed_time, 60)
print(
    f"--- Elapsed Time: {int(minutes)} minutes and {seconds:.2f} seconds ---")
