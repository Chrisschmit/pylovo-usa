"""Generic Data Import Pipeline

This script serves as the main entry point for the data processing
workflow. It orchestrates a series of modules to download, process, and
integrate various geospatial datasets to model infrastructure.

The pipeline executes the following stages in order:
1.  **Census Data Processing:** Fetches census block and geometry data to
    define the target region's boundary.
2.  **NREL Data Processing:** Processes NREL RESstock/Comstock data to
    determine building vintage distributions.
3.  **OpenStreetMap (OSM) Data Extraction:** Downloads power
    infrastructure, buildings, and road networks from OSM for the target
    region.
4.  **Microsoft Buildings Integration:** Downloads and enriches the
    buildings with height data.
5.  **Building Classification:** Combines all building data sources and
    applies heuristics to classify buildings and estimate loads.
6.  **Routable Road Network Generation:** Builds a clean, routable road
    network for use with pgRouting.

Prerequisites:
  - Define your region of interest in the `config_import_data.yaml` file.

Usage:
  # Run the entire pipeline for the configured region
  $ python -m runme.import.import_data_for_target_region
"""
import time
from typing import Any, Dict, Optional

from src.config_loader import LOG_FILE, LOG_LEVEL
from src.import_data.census import CensusDataHandler
from src.import_data.county_segmentation import CountySegmentationHandler
from src.import_data.microsoft_buildings import MicrosoftBuildingsDataHandler
from src.import_data.nrel import NRELDataHandler
from src.import_data.osm.osm_data_handler import OSMDataHandler
from src.import_data.osm.road_network_builder import RoadNetworkBuilder
from src.import_data.processing.building_processor import BuildingProcessor
from src.import_data.workflow import WorkflowOrchestrator
from src.utils import create_logger

logger = create_logger(
    name="Main",
    log_level=LOG_LEVEL,
    log_file=LOG_FILE,
)


def run_full_pipeline(
    census_data: Optional[Dict[str, Any]] = None,
) -> None:
    """Run the main data import pipeline for the target region.

    Uses the WorkflowOrchestrator to manage the pipeline steps.

    Args:
        census_data: A dictionary containing pre-loaded census data,
            including 'target_region_boundary'. If provided, the census
            data processing step is skipped. Defaults to None.
    """
    start_time = time.time()
    logger.info("Starting Data Import Pipeline for Target Region")

    try:
        # # Initialize the orchestrator, loading config, setting up FIPS,
        # and creating all output directories
        orchestrator = WorkflowOrchestrator()

        # --- STEP 1: REGIONAL DATA EXTRACTION & PREPARATION ---
        logger.info("STEP 1: Regional Data Extraction & Preparation")

        census_handler = CensusDataHandler(orchestrator)
        census_data = census_handler.process(plot=False)

        # --- STEP 2: Process NREL Data ---

        logger.info("STEP 2: Processing NREL data")
        nrel_handler = NRELDataHandler(orchestrator)
        nrel_data = nrel_handler.process()

        # --- STEP 3: Extract OSM Data ---
        logger.info("STEP 3: Extracting OSM data")
        osm_handler = OSMDataHandler(orchestrator)
        osm_data = osm_handler.process(plot=False)

        # --- STEP 3.5: Process Microsoft Buildings Data ---
        logger.info("STEP 3.5: Processing Microsoft Buildings data")
        microsoft_buildings_handler = MicrosoftBuildingsDataHandler(
            orchestrator)
        microsoft_buildings_data = microsoft_buildings_handler.process()

        # # --- STEP 4: Building Classification ---
        logger.info("STEP 4: Building Classification")
        building_processor = BuildingProcessor(
            orchestrator.get_dataset_specific_output_directory(
                "BUILDINGS_OUTPUT"))

        building_processor.process(
            census_data,
            osm_data,
            microsoft_buildings_data,
            nrel_data["vintage_distribution"]
        )

        # --- STEP 5: ROUTABLE ROAD NETWORK GENERATION ---
        logger.info("STEP 5: Routable Road Network Generation")
        road_network_builder = RoadNetworkBuilder(orchestrator=orchestrator)
        _ = road_network_builder.process()

        # --- STEP 6: US County Segmentation ---
        logger.info("STEP 6: US County Segmentation")
        subcounty_segmentation_handler = CountySegmentationHandler(
            orchestrator=orchestrator)
        subcounty_segmentation_handler.process(
            state_filter=orchestrator.fips_dict['state'])

        logger.info(
            "Data Import Pipeline for Target Region completed successfully.")

    except ValueError as ve:
        logger.error(
            f"Configuration or validation error during pipeline: {ve}",
            exc_info=True)
    except RuntimeError as re:
        logger.error(
            f"Runtime error during pipeline execution: {re}",
            exc_info=True)
    except Exception as e:
        logger.error(
            f"An unexpected error occurred in the pipeline: {e}",
            exc_info=True)
    finally:
        # Calculate and log total execution time
        end_time = time.time()
        total_time = end_time - start_time

        logger.info(
            "Data Processing Pipeline completed in "
            f"{total_time} seconds"
        )


if __name__ == "__main__":
    run_full_pipeline()
