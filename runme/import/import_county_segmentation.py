#!/usr/bin/env python3
"""
Standalone script for importing US County Segmentation data.

This script processes US Census county subdivision data to augment it with
population, area, and geometry information for a single state or all states.
"""

import argparse
import sys

from src.config_loader import LOG_FILE, LOG_LEVEL
from src.database.database_constructor import DatabaseConstructor
from src.import_data.county_segmentation import CountySegmentationHandler
from src.utils import create_logger


def main():
    """Main function to run the county segmentation import process."""
    parser = argparse.ArgumentParser(
        description="Import and process US County Segmentation data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python import_county_segmentation.py --state CA
  python import_county_segmentation.py --state TX --processes 8
  python import_county_segmentation.py  # Process all states
        """,
    )

    parser.add_argument(
        "--state",
        type=str,
        help="Two-letter US state abbreviation (e.g., CA, TX, MA). If not specified, processes all states.",
    )

    parser.add_argument(
        "--processes",
        type=int,
        default=4,
        help="Number of parallel processes to use (default: 4)",
    )

    args = parser.parse_args()

    # Set up logging
    logger = create_logger(
        name="CountySegmentationImport",
        log_level=LOG_LEVEL,
        log_file=LOG_FILE,
    )

    # Create handler instance in standalone mode
    handler = CountySegmentationHandler(orchestrator=None)

    try:
        if args.state:
            logger.info(
                f"Processing county segmentation data for state: {
                    args.state}")
        else:
            logger.info(
                "Processing county segmentation data for all US states")

        # Process the data
        results = handler.process(
            state_filter=args.state,
            num_processes=args.processes
        )

        # Import the data to the database
        logger.info("Importing data to the database")
        sgc = DatabaseConstructor()
        file_dict = {"path": results["output_file"], "table_name": "postcode"}
        sgc.csv_to_db(file_dict, overwrite=False)

        if "error" in results:
            logger.error(f"Processing failed: {results['error']}")
            return 1

        if "output_file" in results:
            logger.info(
                f"Successfully processed data. Output saved to: {
                    results['output_file']}")

        if "not_found" in results and results["not_found"]:
            logger.warning(
                f"Found {len(results['not_found'])} subdivisions that could not be processed")

        logger.info("County segmentation import completed successfully!")
        return 0

    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error during processing: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
