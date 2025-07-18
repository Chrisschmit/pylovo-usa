# generate the grid for the PLZ set below
# building data import is included

import os
import sys
import time

from plotting.plot_for_plz import plot_boxplot_plz, plot_pie_of_trafo_cables
from src.config_loader import ANALYZE_GRIDS, REGION
from src.database.database_client import DatabaseClient
from src.grid_generator import GridGenerator
from src.load_data.load_buildings import import_buildings_for_single_plz

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


def main():

    plot_results = True

    # timing of the script
    start_time = time.time()
    # Resolve Cnfig data to fiupscode
    dbc = DatabaseClient()
    plz = dbc.get_plz_from_region(REGION)

    # initialize GridGenerator with the provided postal code (PLZ)
    gg = GridGenerator(plz=plz)

    # import building data to the database and get information about the plz
    import_buildings_for_single_plz(gg)

    # generate a grid for the specified region
    gg.generate_grid_for_single_plz(plz=plz, analyze_grids=ANALYZE_GRIDS)

    if plot_results:
        # plot data from the generated grids
        cluster_list = gg.dbc.get_list_from_plz(plz)
        print('The PLZ has', len(cluster_list), 'grids.')
        print(cluster_list)
        plot_boxplot_plz(plz)
        plot_pie_of_trafo_cables(plz)

    # End timing and print results
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(
        f"--- Elapsed Time: {int(minutes)} minutes and {seconds:.2f} seconds ---")


if __name__ == "__main__":
    main()
