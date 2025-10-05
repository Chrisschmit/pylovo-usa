.. pylovo gui documentation master file, created by sphinx-quickstart on Wed Jul 12 12:49:48 2023

Introduction
******************************************************************
Overview pylovo (PYthon tool for LOw-VOltage distribution grid generation)
===========================================================
This tool provides a comprehensive public-data-based module to generate synthetic low-voltage distribution grids for a
freely-selected research area in the United States. The main data inputs include:

- **OpenStreetMap**: Buildings, roads, and existing transformer geographic data
- **Microsoft Buildings**: High-quality building footprint data for the USA
- **NREL**: Residential building typology and energy consumption data
- **US Census Bureau**: Region boundaries using FIPS codes for state, county, and county subdivision selection

The tool outputs feasible hierarchical MV-LV distribution grid networks within the research scope and automatically
analyzes important grid statistics to enable users to evaluate the general grid properties for the generated synthetic grids.

This US implementation extends the original pylovo framework from Bavaria/Germany to work with US data sources and electrical standards.
Due to the large amount of spatial data, users need to set up a local PostgreSQL database with PostGIS extension for the grid generation process.
Step-by-step tutorials to understand the product of this tool can be found in the notebook_tutorials directory.


.. note::

    | **Citation**: In case you use pylovo in a scientific publication, we kindly request you to cite our publication listed in the :doc:`further_reading` section.
    | **Collaboration**: pylovo is open-source available on GitHub and open for collaboration.

Contents
===========================================================
In this documentation you can find instructions and information on:

* How to install pylovo in :doc:`installation/installation`.
* How to generate grids in :doc:`grid_generation/index`.
* How the grids are generated in :doc:`grid_generation/explanation/grid_generation_process`.
* How to visualize your results in :doc:`visualisation/index`.

Legal Notice
==========================
`MIT License <https://opensource.org/license/MIT>`_ , Copyright (C) 2023-2025 Beneharo Reveron Baecker

Acknowledgement
==========================
The development of this software has been supported by contributions of the following persons: Soner Candas, Deniz Tepe,
Tong Ye, Daniel Baur, Julian Zimmer and Berkay Olgun.

Structure
===========================================================

.. toctree::
    :maxdepth: 2

    self
    installation/installation
    grid_generation/index
    visualisation/index
    further_reading
    docs_sphinx/index
