Generate Synthetic Grids
**********

Configuration
=============
| To allow distinction for different parameters you can define grids with different version entries in ``config_version.py``.
| Please enter your VERSION_ID and your VERSION_COMMENT in the ``config_version.py`` file.
| If you don't want to change any parameters you can start with the current base version configurations.

Create your first grid
=========================================
After defining your region (FIPS codes) in the ``config/config_data.yaml`` file, you can run:

::

    python runme/create/create_grid_single_region.py

If the grids for the given region and version are already generated, the code will terminate.

Apart from this you can:

- create grids for multiple regions using FIPS county subdivision codes.
- activate the flags to analyze the grid and visualize some basic results.
- export the grid data as csv or GeoJSON.
- delete specified grids/versions.

.. note::
    - Before running the scripts make sure you followed all steps described in the installation section.
    - Ensure your PostgreSQL database is properly configured in the ``config/config_data.yaml`` file.

Result inspection with QGIS
==================
- Download `QGIS <https://www.qgis.org/download/>`_. Go to the `QGIS` directory in pylovo and open the QGIS file.
- The database connection settings have to be set to the database that is used by pylovo.
- Initial data (ways, buildings and transformers) as well as the networks (transformers, cables, buildings) can be visualised.
- See :doc:`../../visualisation/qgis/qgis` visualisation docu for more details

Tutorials / Examples
=====================
In the `notebook_tutorials/` directory you will learn more about the following topics:

* visualizing individual networks
* the objects and elements the LV grids are made up of
* the electrical network models and grid hierarchies
* graph representation of the networks
* parameter visualization and analysis options
