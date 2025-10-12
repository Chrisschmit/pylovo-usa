.. raw:: html

   <p align="left">
     <img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt="DAI-Lab Logo" />
     <i>An open source project from Data to AI Lab at MIT.</i>
   </p>

**pylovo-usa**
==============

**PYthon tool for LOw-VOltage distribution grid generation - US Edition**

A tool to generate synthetic low-voltage distribution grids for the United States based on open data.

.. list-table::
   :widths: auto

   * - License
     - |badge_license|
   * - Documentation
     - |badge_documentation|
   * - Original Framework
     - `pylovo <https://github.com/tum-ens/pylovo/>`_

Overview
========

This tool provides a comprehensive public-data-based module to generate synthetic low-voltage distribution grids for
freely-selected research areas in the United States. Built on the `pylovo framework <https://github.com/tum-ens/pylovo/>`_
by Beneharo Reveron Baecker et al., this US implementation extends the original tool to work with US data sources and
electrical standards.

**pylovo-usa** was developed in tandem with `GridTracer <https://github.com/DAI-Lab/gridtracer>`_, a companion data fusion
pipeline that serves as the preprocessing engine for pylovo-usa. GridTracer collects and processes geospatial data from
multiple US sources (Census, NREL, OpenStreetMap, Microsoft Buildings) to create the comprehensive building-level datasets
that feed into pylovo-usa's synthetic grid generation algorithms.

.. image:: docs/source/images/gridtracer_pylovo_workflow.pdf
    :width: 800
    :alt: GridTracer and Pylovo-USA Workflow

The workflow diagram above illustrates how GridTracer and pylovo-usa work together to transform raw geospatial data into
complete synthetic distribution grids with hierarchical MV-LV topology.

**Data Sources:**

- **OpenStreetMap**: Buildings, roads, and existing transformer geographic data
- **Microsoft Buildings**: High-quality building footprint data for the USA
- **NREL**: Residential building typology and energy consumption data
- **US Census Bureau**: Region boundaries using FIPS codes (state, county, county subdivision)

**Key Features:**

- Hierarchical MV-LV distribution grid generation with realistic topology
- Region selection using US Census FIPS codes
- Backend-agnostic electrical network modeling (supports pandapower and OpenDSS)
- Three-phase split-phase transformer modeling (120V/240V residential, 277V/480V commercial)
- Automated grid statistics and parameter analysis
- QGIS visualization support

Due to the large amount of spatial data, users need to set up a local PostgreSQL database with PostGIS extension
for the grid generation process. Step-by-step tutorials to understand the product of this tool can be found in
the notebook_tutorials directory.

Quick Start
===========

Installation
------------

**Prerequisites:**

- Python 3.10 or higher
- PostgreSQL with PostGIS extension
- `uv <https://astral.sh/>`_ (fast Python package manager)

**Install uv (if not already installed):**

::

    # Check if uv is installed
    uv --version

    # If not installed, run:
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Restart your shell or run:
    source $HOME/.cargo/env

**Setup pylovo-usa:**

::

    # Clone the repository
    git clone https://github.com/DAI-Lab/pylovo-usa.git
    cd pylovo-usa

    # One-command setup (creates venv + installs all dependencies + sets up pre-commit)
    make setup-dev

    # Activate virtual environment
    source .venv/bin/activate

Configuration
-------------

Configure your region of interest in ``config/config_data.yaml``:

::

    REGION:
      STATE: "NC"                      # State abbreviation (e.g., "NC", "MA")
      COUNTY: "Guilford County"        # Full county name
      COUNTY_SUBDIVISION: "Morehead township"  # County subdivision (optional)

Usage
-----

**1. Import data using GridTracer** (companion tool):

See `GridTracer documentation <https://github.com/DAI-Lab/gridtracer>`_ for data preprocessing.

**2. Generate grids:**

::

    # Generate grid for single region
    python runme/create/create_grid_single_region.py

    # Generate grids for multiple regions
    python runme/create/create_grid_multi_region.py

**3. Visualize results:**

Open the QGIS project file in the ``QGIS/`` directory to visualize generated grids.

Development
-----------

**Linting and formatting:**

::

    # Run all quality checks (linting, formatting, type checking)
    make lint

**Testing:**

::

    pytest tests/

For detailed documentation, see the `GitBook documentation <https://DAI-Lab.github.io/pylovo-usa>`_.

License
====================
| The code of this repository is licensed under the **MIT License** (MIT).
| See `LICENSE.txt <LICENSE.txt>`_ for rights and obligations.
| Copyright: `pylovo <https://github.com/tum-ens/pylovo/>`_ © `TUM ENS`_ | `MIT <LICENSE.txt>`_

Citation
====================
| If you use this code in a scientific publication, please cite the original pylovo framework:
* Reveron Baecker et al. (2025): `Generation of low-voltage synthetic grid data for energy system modeling with the pylovo tool <https://doi.org/10.1016/j.segan.2024.101617>`_

Acknowledgment
====================
This US implementation (pylovo-usa) is built upon the original `pylovo framework <https://github.com/tum-ens/pylovo/>`_
developed by Beneharo Reveron Baecker and the TUM ENS team. The original framework was designed for Bavarian and German
data sources. This extension adapts the methodology for United States data sources and electrical standards.


.. |badge_license| image:: https://img.shields.io/github/license/DAI-Lab/pylovo-usa
    :target: LICENSE.txt
    :alt: License

.. |badge_documentation| image:: https://img.shields.io/badge/docs-GitBook-blue
    :target: https://DAI-Lab.github.io/pylovo-usa
    :alt: Documentation
