Building Data Import
=====================

The building data is the basis for the grid generation as it contains geographical information as
well as the load that each consumer requires. The data is imported from multiple US-specific sources:

- **OpenStreetMap**: Building footprints, roads, and existing transformers
- **Microsoft Buildings**: High-quality building footprint data for the USA
- **NREL**: Residential building typology and energy consumption data
- **US Census Bureau**: Region boundaries and demographic data using FIPS codes

The data import process is automated through the import pipeline:

::

    python runme/import/import_pipeline.py

The imported buildings can be inspected using the QGIS visualisation :doc:`../../visualisation/qgis/qgis` in the
:code:`raw_data` tab
