# generate the grid for the PLZ set below
# building data import is included

import os
import sys
import time

from plotting.plot_for_plz import plot_boxplot_plz, plot_pie_of_trafo_cables
from src.config_loader import ANALYZE_GRIDS
from src.data_import.import_buildings import import_buildings_for_single_plz
from src.grid_generator import GridGenerator

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


# enter a plz to generate grid for:
plz = 2501711000
plot_results = True

# timing of the script
start_time = time.time()

# initialize GridGenerator with the provided postal code (PLZ)
gg = GridGenerator(plz=plz)

# import building data to the database and get information about the plz
import_buildings_for_single_plz(gg)

# generate a grid for the specified region
gg.generate_grid_for_single_plz(plz=plz, analyze_grids=ANALYZE_GRIDS)

if plot_results:
    # plot data from the generated grids
    dbc_client = gg.dbc
    cluster_list = gg.dbc.get_list_from_plz(plz)
    print('The PLZ has', len(cluster_list), 'grids.')
    plot_boxplot_plz(plz)
    plot_pie_of_trafo_cables(plz)

# End timing and print results
elapsed_time = time.time() - start_time
minutes, seconds = divmod(elapsed_time, 60)
print(
    f"--- Elapsed Time: {int(minutes)} minutes and {seconds:.2f} seconds ---")
