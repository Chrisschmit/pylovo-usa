"""
OSM data processing module.

This module contains utilities for processing OpenStreetMap (OSM) data
for the data import pipeline, including road network generation.
"""

from src.import_data.osm.road_network_builder import RoadNetworkBuilder

__all__ = ['RoadNetworkBuilder']
