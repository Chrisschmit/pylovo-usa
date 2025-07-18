# This script performs all processes of representative grid generation
# Enter the settings for the clustering process in config_classification and clustering.config_clustering
# also refer to the documentation of the classification

import os
import sys
import time

from src.classification.clustering.filter_grids import apply_filter_to_grids
from src.classification.sampling.sample import (create_sample_set,
                                                get_sample_set)
from src.load_data.load_buildings import import_buildings_for_multiple_plz
from src.grid_generator import GridGenerator
from src.parameter_calculator import ParameterCalculator

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


def prepare_data_for_clustering(additional_filtering: bool = False) -> None:
    # %% 1. create a sample set of PLZ for your classification
    # This takes a few seconds

    create_sample_set()
    samples = get_sample_set()

    # %% 2. import the buildings for the grid generation from the building database
    # importing a single shape file takes a few minutes. Importing the
    # buildings for a whole set will take a few hours
    start_time = time.time()

    import_buildings_for_multiple_plz(sample_plz=samples)
    print(
        "--- %s seconds for step 2: building import---" %
        (time.time() - start_time))

    # %% 3. generate the grids for your set
    # create new version if config_version, grid parameter
    # this takes around a quarter of an hour for a grid and might take a whole day for an entire set.
    # check whether grid was already generated

    start_time = time.time()
    # initialize GridGenerator with the provided postal code (PLZ)
    gg = GridGenerator()
    gg.generate_grid_for_multiple_plz(df_plz=samples, analyze_grids=True)
    print(
        "--- %s seconds for step 3: grid generation---" %
        (time.time() - start_time))

    # %% 4. calculate grid parameters

    # timing of the script
    start_time = time.time()

    pc = ParameterCalculator()
    # calculate network parameter for all plz
    for plz_index in samples['plz']:
        # compute parameters for plz
        pc.calc_parameters_per_grid(plz=plz_index)

    # end timing
    print(
        "--- %s seconds parameter calculation---" %
        (time.time() - start_time))

    # %% 5. filter values
    apply_filter_to_grids(additional_filtering=additional_filtering)


def main():
    prepare_data_for_clustering()


if __name__ == "__main__":
    main()
