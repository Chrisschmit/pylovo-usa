# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is **pylovo** (PYthon tool for LOw-VOltage distribution grid generation) - a comprehensive module that generates synthetic low-voltage distribution grids for freely-selected research areas in the USA, based on open data sources. The tool extends the original pylovo framework from Bavaria/Germany to work with US data sources.

## Key Data Sources & Requirements

- **PostgreSQL Database**: Required for storing and processing the large amounts of spatial data
- **OpenStreetMap**: Buildings, roads, and transformer data
- **US Census Bureau**: FIPS codes for region selection
- **NREL**: Residential building typology data
- **Microsoft Buildings**: Building footprint data

## Development Commands

### Code Quality & Formatting
```bash
make fix-lint                    # Fix linting and formatting issues (autoflake, autopep8, isort)
```

### Database Setup
```bash
python runme/main_constructor.py # Create and populate PostgreSQL database with raw data
```

### Main Operations
```bash
python runme/main_classification.py  # Run classification pipeline for grid clustering
python runme/create/create_grid_single_region.py    # Generate grids for single postcode
python runme/create/create_grid_multi_region.py     # Generate grids for multiple postcodes
```

### Data Import Pipeline
```bash
python runme/import/import_pipeline.py           # Full data import workflow
```

## Architecture Overview

### Core Components

1. **GridGenerator** (`src/grid_generator.py`): Main class for generating distribution grids
   - Handles single/multi postcode processing
   - Manages K-means clustering for street networks
   - Positions transformers (brownfield/greenfield scenarios)
   - Installs electrical cables with pandapower networks

2. **DatabaseClient** (`src/database/database_client.py`): Main database interface using mixins
   - **PreprocessingMixin**: Data preparation and building processing
   - **ClusteringMixin**: K-means clustering and component analysis
   - **GridMixin**: Grid generation and electrical network creation
   - **AnalysisMixin**: Parameter calculation and analysis
   - **UtilsMixin**: Utility functions for database operations

3. **Import Data Pipeline** (`src/import_data/`):
   - **workflow.py**: Main import orchestration
   - **census.py**: US Census data processing
   - **microsoft_buildings.py**: Building footprint processing
   - **nrel.py**: NREL residential typology integration
   - **osm/**: OpenStreetMap data handling and road network building

### Key Configuration Files

- **config/config_data.yaml**: Database connection, region selection (FIPS codes), file paths
- **config/config_classification.yaml**: Classification and clustering parameters
- **config/config_clustering.yaml**: Specific clustering algorithm settings

### Database Architecture

The system uses PostgreSQL with PostGIS for spatial operations:
- **Raw Data Tables**: Buildings (res/oth), transformers, ways, postcode
- **Temporary Tables**: Processing tables (buildings_tem, ways_tem)
- **Result Tables**: Final grid data (buildings_result, grid_result, lines_result)

### Grid Generation Process

1. **Prepare Data**: Load buildings, transformers, ways for postcode area
2. **Apply Clustering**: Use K-means to segment street networks into manageable components
3. **Position Transformers**: Handle existing (brownfield) and new (greenfield) transformer placement
4. **Install Cables**: Create pandapower networks and connect buildings with appropriate cable sizing

## Region Configuration

The system uses US Census FIPS codes for region selection in `config/config_data.yaml`:
```yaml
REGION:
  STATE: "NC"  # North Carolina
  COUNTY: "Guilford County"
  COUNTY_SUBDIVISION: "Morehead township"
```

## Testing & Validation

- No formal test suite is present
- Validation occurs through grid parameter analysis and visualization in QGIS
- Notebook tutorials in `notebook_tutorials/` provide step-by-step validation examples

## Dependencies

The project requires PostgreSQL with PostGIS extension and Python packages including:
- geopandas, pandas, numpy for data processing
- pandapower for electrical network modeling
- scikit-learn for clustering algorithms
- SQLAlchemy, psycopg2 for database operations
- matplotlib, plotly, seaborn for visualization

## Development Notes

- The codebase follows German naming conventions in some areas (historical from original pylovo)
- Extensive logging is configured via LOG_LEVEL and LOG_FILE in config
- Parallel processing is implemented for cable installation using multiprocessing
- EPSG:5070 (NAD83 / Conus Albers) is used for US spatial projections