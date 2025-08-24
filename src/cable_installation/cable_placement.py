"""
Cable placement algorithms extracted from grid_generator.py

This module contains the core cable placement algorithms that determine optimal
cable routing and sizing for distribution networks. The algorithms use a
branch-by-branch approach starting from the furthest nodes and working toward
transformers.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .. import utils
from ..config_loader import *
from ..database.database_client import DatabaseClient
from ..electrical_backend.component_specs import (BusSpec, ComponentSpec,
                                                  LineSpec, LoadSpec)
from .cable_selection import VoltageAwareCableSelector


class CablePlacementAlgorithm:
    """
    Implements cable placement algorithms for electrical distribution networks.

    Extracted from GridGenerator._install_cables_for_cluster with refactoring
    to work with component specifications instead of direct pandapower creation.
    """

    def __init__(self, database: Optional[DatabaseClient]
                 = None, logger: Optional[logging.Logger] = None):
        """Initialize cable placement algorithm with voltage-aware cable selection."""
        self.logger = logger or logging.getLogger(__name__)

        # Store database reference for coordinate generation
        self.database = database

        # Initialize voltage-aware cable selector if database is available
        if database:
            self.cable_selector = VoltageAwareCableSelector(
                database, logger=self.logger)
            self.use_voltage_aware_selection = True
        else:
            self.cable_selector = None
            self.use_voltage_aware_selection = False
            self.logger.warning(
                "No database provided - falling back to static cable selection")

    def _generate_line_coordinates(
            self, from_vertex: int, to_vertex: int) -> Optional[List[Tuple[float, float]]]:
        """
        Generate coordinate list for a line between two vertices using database routing.

        Args:
            from_vertex: Source vertex ID
            to_vertex: Destination vertex ID

        Returns:
            List of (x, y) coordinate tuples or None if database unavailable
        """
        if not self.database:
            return None

        try:
            # Get shortest path between vertices
            path_nodes = self.database.get_path_to_bus(from_vertex, to_vertex)

            # Convert path nodes to coordinates
            coordinates = []
            if path_nodes:
                for node_id in path_nodes:
                    coord = self.database.get_node_geom(node_id)
                    if coord:
                        coordinates.append((float(coord[0]), float(coord[1])))

            # If we got valid coordinates from routing, return them
            if len(coordinates) >= 2:
                return coordinates

            # Fallback: direct point-to-point connection using node geometry
            try:
                from_coord = self.database.get_node_geom(from_vertex)
                to_coord = self.database.get_node_geom(to_vertex)

                if from_coord and to_coord:
                    return [
                        (float(from_coord[0]), float(from_coord[1])),
                        (float(to_coord[0]), float(to_coord[1]))
                    ]
            except Exception as e:
                self.logger.debug(
                    f"Fallback coordinate generation failed: {e}")

            return None

        except Exception as e:
            self.logger.warning(
                f"Failed to generate coordinates for line {from_vertex}->{to_vertex}: {e}")
            return None

    def create_lv_network_components(
        self,
        cluster_id: str,
        lv_bus: str,
        vertices_dict: Dict[int, float],
        transformer_vertex: int,
        buildings_df: pd.DataFrame,
        consumer_df: pd.DataFrame,
        connection_nodes: List[int],
        equipment_lookup: callable
    ) -> List[ComponentSpec]:
        """
        Create component specifications for LV network using branch-by-branch algorithm.

        This is the main algorithm extracted from _install_cables_for_cluster.

        Args:
            cluster_id: Unique identifier for this cluster (e.g. "K123_S456_B789")
            lv_bus: Name of the LV bus (transformer secondary side)
            vertices_dict: Mapping of vertex_id -> distance_to_transformer
            transformer_vertex: Vertex ID of transformer location
            buildings_df: DataFrame with building information
            consumer_df: DataFrame with consumer categories
            connection_nodes: List of connection point vertices (excluding buildings)
            equipment_lookup: Function to get cable equipment by name

        Returns:
            List of ComponentSpec objects for buses, lines, and loads
        """
        component_specs = []

        # Calculate load data for all consumers
        consumer_list = buildings_df.vertice_id.to_list()
        consumer_list = list(dict.fromkeys(consumer_list))  # Remove duplicates

        Pd, load_units, load_type = self._get_consumer_simultaneous_load_dict(
            consumer_list, buildings_df, consumer_df
        )

        # Create buses for all connection nodes
        for node_id in connection_nodes:
            bus_spec = BusSpec(
                name=f"Bus_Node_{node_id}_{cluster_id}",
                coordinates=self._get_node_coordinates(
                    node_id)  # TODO: Implement coordinate lookup
            )
            component_specs.append(bus_spec)

        # Create buses and loads for all consumers
        for consumer_id in consumer_list:
            # Bus for each consumer
            bus_spec = BusSpec(
                name=f"Bus_Consumer_{consumer_id}_{cluster_id}",
                coordinates=self._get_node_coordinates(consumer_id)
            )
            component_specs.append(bus_spec)

            # Load for each consumer
            building_data = buildings_df[buildings_df.vertice_id ==
                                         consumer_id].iloc[0]
            load_spec = LoadSpec(
                name=f"Load_{consumer_id}_{cluster_id}",
                bus=f"Bus_Consumer_{consumer_id}_{cluster_id}",
                kw=Pd[consumer_id] * 1000,  # Convert MW to kW
                kvar=Pd[consumer_id] * 1000 * 0.3,  # Assume 0.3 power factor
                kv=VN,
                n_phases=3,
                conn="wye"
            )
            component_specs.append(load_spec)

        # First, install consumer cables from connection points to buildings
        consumer_specs = self._install_consumer_cables(
            cluster_id=cluster_id,
            connection_nodes=connection_nodes,
            consumer_list=consumer_list,
            transformer_vertex=transformer_vertex,
            vertices_dict=vertices_dict,
            Pd=Pd,
            buildings_df=buildings_df,
            equipment_lookup=equipment_lookup
        )
        component_specs.extend(consumer_specs)

        # Apply branch-by-branch cable algorithm for main distribution
        local_cable_usage = {}  # Will track cables as they are used
        remaining_nodes = connection_nodes.copy()
        branch_counter = 0

        while remaining_nodes:
            if len(remaining_nodes) == 1:
                # Handle final node
                line_specs, cable_usage = self._install_final_branch(
                    remaining_nodes[0], lv_bus, cluster_id, transformer_vertex,
                    vertices_dict, buildings_df, consumer_df, equipment_lookup
                )
                component_specs.extend(line_specs)
                self._update_cable_usage(local_cable_usage, cable_usage)
                break

            # Find furthest node path and determine maximum load branch
            furthest_path = self._find_furthest_node_path_list(
                remaining_nodes, vertices_dict, transformer_vertex
            )

            branch_nodes, max_current = self._determine_maximum_load_branch(
                furthest_path, buildings_df, consumer_df
            )

            # Install cables for this branch
            branch_specs, cable_usage = self._install_branch_cables(
                branch_nodes, lv_bus, cluster_id, transformer_vertex,
                vertices_dict, Pd, max_current, equipment_lookup, branch_counter
            )

            component_specs.extend(branch_specs)
            self._update_cable_usage(local_cable_usage, cable_usage)

            # Remove processed nodes
            for node in branch_nodes:
                if node in remaining_nodes:
                    remaining_nodes.remove(node)

            branch_counter += 1

        # Log cable usage summary
        total_length = sum(local_cable_usage.values())
        self.logger.debug(
            f"Cable installation completed for {cluster_id}: {
                total_length:.1f}km total")

        return component_specs

    def create_mv_network_components(
        self,
        cluster_id: str,
        mv_main_bus: str,
        lv_transformers: List[dict],
        mv_buildings: List[dict],
        equipment_lookup: callable,
        substation_vertex_id: Optional[int] = None
    ) -> List[ComponentSpec]:
        """
        Create component specifications for MV network using branch-by-branch algorithm.

        This method adapts the LV algorithm for MV networks, treating both MV buildings
        and distribution transformers as load points to be connected to the MV main bus.

        Args:
            cluster_id: Unique identifier (e.g., "MV_K1_S0")
            mv_main_bus: Name of the MV main bus (20kV)
            lv_transformers: List of distribution transformers with capacity and location
            mv_buildings: List of direct MV connections with load and location
            equipment_lookup: Function to get cable equipment by name

        Returns:
            List of ComponentSpec objects for MV buses, lines, and loads
        """
        component_specs = []

        if not lv_transformers and not mv_buildings:
            self.logger.warning(f"No MV load points found for {cluster_id}")
            return component_specs

        # Consolidate all MV load points
        mv_load_points = []

        # Add distribution transformers as load points --> TODO: is this
        # correct?
        for lv_tx in lv_transformers:
            mv_load_points.append({
                'vertex_id': lv_tx['transformer_vertice_id'],
                # Same as vertex_id for transformers
                'connection_point': lv_tx['transformer_vertice_id'],
                # Transformer capacity
                'load_kw': lv_tx['transformer_rated_power'],
                'type': 'transformer',
                'name': f"DistTx_B{lv_tx['bcid']}",
                # For transformer connection reference
                'transformer_id': lv_tx['bcid'],
                'bcid': lv_tx['bcid']
            })

        # Add direct MV buildings as load points
        # Note: We use connection_point for MV feeder routing, not the building centroid
        # The service line from connection_point to building is handled
        # separately
        for mv_bldg in mv_buildings:
            mv_load_points.append({
                # Use connection point for feeder routing
                'vertex_id': mv_bldg['connection_point'],
                # Explicit connection point
                'connection_point': mv_bldg['connection_point'],
                'load_kw': mv_bldg['peak_load_kw'],
                'type': 'mv_building',  # Changed from 'building' to be more explicit
                'name': f"MVBldg_{mv_bldg['osm_id']}",
                'osm_id': mv_bldg['osm_id'],
                # Store building centroid for reference
                'building_vertex_id': mv_bldg.get('vertice_id')
            })

        if not mv_load_points:
            return component_specs

        self.logger.info(
            f"Planning MV network for {
                len(mv_load_points)} load points")

        # First, create MV building service connections (connection point -> building)
        # This is exactly like LV consumer cables, but for MV commercial
        # buildings
        if mv_buildings:
            mv_building_specs = self._install_mv_consumer_cables(
                cluster_id=cluster_id,
                mv_buildings=mv_buildings,
                equipment_lookup=equipment_lookup
            )
            component_specs.extend(mv_building_specs)

        # Create buses for all MV load points
        for load_point in mv_load_points:
            # Get coordinates for the load point vertex
            coordinates = None
            if self.database:
                try:
                    coord = self.database.get_node_geom(
                        load_point['vertex_id'])
                    if coord:
                        coordinates = (float(coord[0]), float(coord[1]))
                except Exception as e:
                    self.logger.debug(
                        f"Could not get coordinates for vertex {
                            load_point['vertex_id']}: {e}")

            bus_spec = BusSpec(
                name=f"MV_Node_{load_point['vertex_id']}_{cluster_id}",
                coordinates=coordinates
            )
            component_specs.append(bus_spec)

            # Only create load specification for actual MV buildings, not transformers
            # Transformers are created separately as TransformerSpec components
            if load_point['type'] != 'transformer':
                # Convert Decimal to float
                load_kw = float(load_point['load_kw'])
                load_spec = LoadSpec(
                    name=f"MV_Load_{load_point['name']}_{cluster_id}",
                    bus=f"MV_Node_{load_point['vertex_id']}_{cluster_id}",
                    kw=load_kw,
                    kvar=load_kw * 0.3,  # Assume 0.3 reactive power factor
                    kv=20.0,  # 20kV MV
                    n_phases=3,
                    conn="delta"  # Typical for MV
                )
                component_specs.append(load_spec)

        # Apply MV cable placement using branch-by-branch approach
        mv_line_specs = self._install_mv_cables_branch_by_branch(
            mv_load_points, mv_main_bus, cluster_id, equipment_lookup, substation_vertex_id
        )

        component_specs.extend(mv_line_specs)

        total_mv_load = sum(lp['load_kw'] for lp in mv_load_points)
        self.logger.info(
            f"✓ MV network completed: {
                len(mv_line_specs)} lines, {
                total_mv_load:.0f}kW total load")

        return component_specs

    def _install_mv_cables_branch_by_branch(
        self,
        mv_load_points: List[dict],
        mv_main_bus: str,
        cluster_id: str,
        equipment_lookup: callable,
        substation_vertex_id: Optional[int] = None
    ) -> List[ComponentSpec]:
        """
        Install MV cables using trunk+branch pattern (same as LV algorithm but for MV scale).

        This creates realistic MV distribution topology with shared trunks and branches,
        instead of inefficient radial connections from substation to each load point.

        Args:
            mv_load_points: List of MV load points (transformers + buildings)
            mv_main_bus: MV main bus name
            cluster_id: Network identifier
            equipment_lookup: Equipment lookup function
            substation_vertex_id: Substation location for routing

        Returns:
            List of LineSpec components for MV trunk+branch cables
        """
        line_specs = []

        if not mv_load_points or not substation_vertex_id:
            self.logger.warning(
                f"Cannot create MV network: no load points or substation location")
            return line_specs

        # Step 1: Create distance mapping for MV load points (like
        # vertices_dict for LV)
        mv_vertices_dict = self._create_mv_vertices_dict(
            mv_load_points, substation_vertex_id)

        # Step 2: Apply branch-by-branch algorithm (same pattern as LV)
        remaining_load_points = mv_load_points.copy()
        branch_counter = 0
        local_cable_usage = {}

        self.logger.info(
            f"Building MV trunk+branch network for {len(mv_load_points)} load points")

        while remaining_load_points:
            if len(remaining_load_points) == 1:
                # Handle final MV load point (same pattern as LV)
                final_specs, cable_usage = self._install_final_mv_branch(
                    remaining_load_points[0], mv_main_bus, cluster_id,
                    substation_vertex_id, mv_vertices_dict, equipment_lookup
                )
                line_specs.extend(final_specs)
                self._update_cable_usage(local_cable_usage, cable_usage)
                break

            # Step 3: Find furthest MV load point path (same logic as LV)
            furthest_path = self._find_furthest_mv_path(
                remaining_load_points, mv_vertices_dict, substation_vertex_id
            )

            # Step 4: Determine maximum MV branch (limited by MV cable
            # capacity)
            branch_load_points, max_current = self._determine_maximum_mv_branch(
                furthest_path, equipment_lookup
            )

            # Step 5: Install MV branch trunk + connections
            branch_specs, cable_usage = self._install_mv_branch_trunk(
                branch_load_points, mv_main_bus, cluster_id,
                substation_vertex_id, mv_vertices_dict, max_current,
                equipment_lookup, branch_counter
            )
            line_specs.extend(branch_specs)
            self._update_cable_usage(local_cable_usage, cable_usage)

            # Step 6: Remove processed load points and increment branch
            for lp in branch_load_points:
                if lp in remaining_load_points:
                    remaining_load_points.remove(lp)

            branch_counter += 1

        # Log MV cable usage summary
        total_mv_length = sum(local_cable_usage.values())
        self.logger.info(
            f"✓ MV trunk+branch network completed: {len(line_specs)} lines, "
            f"{total_mv_length:.1f}km total cable, {branch_counter} branches"
        )

        return line_specs

    def _create_mv_vertices_dict(
            self, mv_load_points: List[dict], substation_vertex_id: int) -> Dict[int, float]:
        """
        Create distance mapping for MV load points (equivalent to vertices_dict for LV).

        Maps each MV load point vertex to its routing distance from the substation.
        This enables the branch-by-branch algorithm to work at MV scale.

        Args:
            mv_load_points: List of MV transformers and buildings
            substation_vertex_id: Substation location vertex

        Returns:
            Dict mapping vertex_id -> distance_from_substation (in meters)
        """
        mv_vertices_dict = {}

        if not self.database:
            self.logger.warning(
                "No database connection for MV routing distance calculation")
            # Fallback: assign dummy distances
            for i, load_point in enumerate(mv_load_points):
                # 1km, 2km, 3km...
                mv_vertices_dict[load_point['vertex_id']] = (i + 1) * 1000.0
            return mv_vertices_dict

        for load_point in mv_load_points:
            load_vertex = load_point['vertex_id']

            try:
                # Get routing path from substation to this load point
                path_nodes = self.database.get_path_to_bus(
                    load_vertex, substation_vertex_id)

                if path_nodes and len(path_nodes) >= 2:
                    # Calculate total path distance by summing segments
                    total_distance = 0.0

                    for i in range(len(path_nodes) - 1):
                        # Get coordinates of consecutive nodes
                        coord1 = self.database.get_node_geom(path_nodes[i])
                        coord2 = self.database.get_node_geom(path_nodes[i + 1])

                        if coord1 and coord2:
                            # Calculate Euclidean distance between nodes
                            dx = float(coord2[0]) - float(coord1[0])
                            dy = float(coord2[1]) - float(coord1[1])
                            segment_distance = np.sqrt(dx * dx + dy * dy)
                            total_distance += segment_distance

                    # Store distance in meters
                    mv_vertices_dict[load_vertex] = max(
                        total_distance, 100.0)  # Min 100m

                else:
                    # Fallback: direct distance to substation
                    substation_coord = self.database.get_node_geom(
                        substation_vertex_id)
                    load_coord = self.database.get_node_geom(load_vertex)

                    if substation_coord and load_coord:
                        dx = float(load_coord[0]) - float(substation_coord[0])
                        dy = float(load_coord[1]) - float(substation_coord[1])
                        distance = np.sqrt(dx * dx + dy * dy)
                        mv_vertices_dict[load_vertex] = max(
                            distance, 100.0)  # Min 100m
                    else:
                        # Final fallback: assign based on position in list
                        mv_vertices_dict[load_vertex] = (
                            len(mv_vertices_dict) + 1) * 1000.0

            except Exception as e:
                self.logger.debug(
                    f"Error calculating MV distance for vertex {load_vertex}: {e}")
                # Fallback distance
                mv_vertices_dict[load_vertex] = (
                    len(mv_vertices_dict) + 1) * 1000.0

        # Log the created distance mapping
        distances_km = [dist / 1000.0 for dist in mv_vertices_dict.values()]
        self.logger.debug(
            f"Created MV vertices dict: {len(mv_vertices_dict)} load points, "
            f"distances: {min(distances_km):.1f}-{max(distances_km):.1f}km"
        )

        return mv_vertices_dict

    def _find_furthest_mv_path(self, remaining_load_points: List[dict],
                               mv_vertices_dict: Dict[int, float],
                               substation_vertex_id: int) -> List[dict]:
        """
        Find path to furthest MV load point (adapted from LV algorithm).

        This identifies the longest routing path from substation to any remaining
        load point, which becomes the "trunk" for this branch.

        Args:
            remaining_load_points: Unprocessed MV load points
            mv_vertices_dict: Distance mapping for all MV load points
            substation_vertex_id: Substation location

        Returns:
            List of load points in path from substation to furthest point
        """
        if not remaining_load_points:
            return []

        # Find the load point with maximum distance from substation
        furthest_load_point = max(
            remaining_load_points,
            key=lambda lp: mv_vertices_dict.get(lp['vertex_id'], 0)
        )

        # Get the routing path from substation to furthest point
        if self.database:
            try:
                path_vertices = self.database.get_path_to_bus(
                    furthest_load_point['vertex_id'], substation_vertex_id
                )

                # Filter path to only include remaining load points (potential
                # branch nodes)
                path_load_points = []
                {lp['vertex_id'] for lp in remaining_load_points}

                for vertex_id in path_vertices:
                    # Find load point matching this vertex
                    matching_lp = next(
                        (
                            lp for lp in remaining_load_points if lp['vertex_id'] == vertex_id),
                        None
                    )
                    if matching_lp:
                        path_load_points.append(matching_lp)

                # Return path in correct order (furthest first for
                # branch-by-branch)
                return path_load_points

            except Exception as e:
                self.logger.debug(
                    f"Error finding MV path to furthest point: {e}")

        # Fallback: return just the furthest point
        return [furthest_load_point]

    def _determine_maximum_mv_branch(self, furthest_path: List[dict],
                                     equipment_lookup: callable) -> Tuple[List[dict], float]:
        """
        Determine maximum MV branch that can be served by heaviest MV cable.

        This adapts the LV logic for MV scale - starts from furthest point and works
        back toward substation, accumulating load until MV cable capacity is reached.

        Args:
            furthest_path: Path from substation to furthest MV load point
            equipment_lookup: Function to get cable equipment specs

        Returns:
            Tuple of (branch_load_points, max_current)
        """
        if not furthest_path:
            return [], 0.0

        branch_load_points = []

        # MV cable capacity limit (adapt from LV's 0.313 kA limit)
        # For MV 20kV networks, typical heavy-duty cables can handle 200-300A
        MV_MAX_CURRENT_A = 200.0  # Conservative limit for MV trunk cables

        # Accumulate load points starting from furthest
        for load_point in furthest_path:
            branch_load_points.append(load_point)

            # Calculate total load for current branch
            total_load_kw = sum(float(lp['load_kw'])
                                for lp in branch_load_points)

            # Calculate required current for MV (20kV, 3-phase, 0.9 pf)
            required_current_a = (total_load_kw * 1000) / \
                (np.sqrt(3) * 20000 * 0.9)

            # Check if we've exceeded MV cable capacity
            if required_current_a >= MV_MAX_CURRENT_A and len(
                    branch_load_points) > 1:
                # Remove the last load point that pushed us over the limit
                branch_load_points.remove(load_point)
                break
            elif required_current_a >= MV_MAX_CURRENT_A and len(branch_load_points) == 1:
                # Even single load point exceeds capacity - will need special
                # handling
                self.logger.warning(
                    f"Single MV load point {
                        load_point['name']} requires {
                        required_current_a:.1f}A "
                    f"(exceeds standard MV cable limit of {MV_MAX_CURRENT_A}A)"
                )
                break

        # Calculate final current for the selected branch
        if branch_load_points:
            total_branch_load_kw = sum(
                float(lp['load_kw']) for lp in branch_load_points)
            max_current = (total_branch_load_kw * 1000) / \
                (np.sqrt(3) * 20000 * 0.9)
        else:
            max_current = 0.0

        self.logger.debug(
            f"MV branch determined: {len(branch_load_points)} load points, "
            f"{total_branch_load_kw:.0f}kW, {max_current:.1f}A"
        )

        return branch_load_points, max_current

    def _install_mv_branch_trunk(self, branch_load_points: List[dict], mv_main_bus: str,
                                 cluster_id: str, substation_vertex_id: int,
                                 mv_vertices_dict: Dict[int, float], max_current: float,
                                 equipment_lookup: callable, branch_id: int) -> Tuple[List[ComponentSpec], Dict[str, float]]:
        """
        Install MV branch trunk and connections to load points.

        This creates the main MV trunk cable along the optimal path and then
        connects each load point (transformer or MV building) to this trunk.

        Args:
            branch_load_points: List of MV load points for this branch
            mv_main_bus: Main MV bus name at substation
            cluster_id: Cluster identifier for naming
            substation_vertex_id: Vertex ID of substation connection
            mv_vertices_dict: Distance mapping from substation
            max_current: Maximum current for this branch
            equipment_lookup: Function to get equipment specifications
            branch_id: Branch counter for unique naming

        Returns:
            Tuple of (line_specs, cable_usage)
        """
        line_specs = []
        cable_usage = {}

        if not branch_load_points or not mv_vertices_dict:
            self.logger.warning(
                "Missing branch load points or vertices dictionary for MV trunk installation")
            return line_specs, cable_usage

        # Select MV cable for trunk line based on total branch load
        trunk_cable_name, trunk_parallel = self.find_optimal_mv_cable(
            required_current_a=max_current,
            distance_km=0  # Distance will be calculated per segment
        )

        if not trunk_cable_name:
            # Fallback to conservative MV cable selection
            self.logger.warning("No optimal MV cable found, using fallback")
            trunk_cable_name = "AL150"  # Conservative choice
            trunk_parallel = 1

        trunk_cable_equipment = equipment_lookup(trunk_cable_name)

        # Sort load points by distance from substation (create trunk path)
        # Use vertex_id for distance lookup since mv_vertices_dict maps
        # vertex_id -> distance
        sorted_load_points = sorted(
            branch_load_points,
            key=lambda lp: mv_vertices_dict.get(lp['vertex_id'], float('inf'))
        )

        # Note: MV buses are already created in create_mv_network_components
        # No need to create duplicate buses here

        # Install trunk segments between consecutive load points
        previous_point = None
        previous_distance = 0.0

        for i, load_point in enumerate(sorted_load_points):
            current_point = load_point['connection_point']
            # Use vertex_id for distance lookup (consistent with sorting logic)
            current_vertex_id = load_point['vertex_id']
            current_distance = mv_vertices_dict.get(current_vertex_id, 0.0)

            if previous_point is not None:
                # Create trunk segment between previous and current load point
                segment_distance_km = abs(
                    current_distance - previous_distance) / 1000.0
                segment_distance_km = max(
                    segment_distance_km, 0.01)  # Minimum 10m

                # Generate coordinates for trunk segment
                coordinates = self._generate_line_coordinates(
                    previous_point, current_point)

                trunk_spec = LineSpec(
                    name=f"MV_Trunk_B{branch_id}_S{i}_{cluster_id}",
                    cable_equipment=trunk_cable_equipment,
                    bus1=f"MV_Node_{previous_point}_{cluster_id}",
                    bus2=f"MV_Node_{current_point}_{cluster_id}",
                    length_km=segment_distance_km,
                    parallel=trunk_parallel,
                    coordinates=coordinates
                )
                line_specs.append(trunk_spec)

                # Track cable usage
                cable_usage[trunk_cable_name] = cable_usage.get(
                    trunk_cable_name, 0) + segment_distance_km

            previous_point = current_point
            previous_distance = current_distance

        # Connect first load point to substation (trunk connection)
        if sorted_load_points:
            first_point = sorted_load_points[0]['connection_point']
            first_vertex_id = sorted_load_points[0]['vertex_id']
            first_distance = mv_vertices_dict.get(first_vertex_id, 0.0)
            trunk_distance_km = first_distance / 1000.0
            trunk_distance_km = max(
                trunk_distance_km,
                0.001)  # Minimum connection distance

            # Generate coordinates for substation connection
            coordinates = self._generate_line_coordinates(
                substation_vertex_id, first_point)

            trunk_connection_spec = LineSpec(
                name=f"MV_Trunk_B{branch_id}_Main_{cluster_id}",
                cable_equipment=trunk_cable_equipment,
                bus1=mv_main_bus,
                bus2=f"MV_Node_{first_point}_{cluster_id}",
                length_km=trunk_distance_km,
                parallel=trunk_parallel,
                coordinates=coordinates
            )
            line_specs.append(trunk_connection_spec)

            # Track cable usage for main trunk connection
            cable_usage[trunk_cable_name] = cable_usage.get(
                trunk_cable_name, 0) + trunk_distance_km

        # Now connect each load point to its actual load (transformer or MV
        # building)
        for load_point in sorted_load_points:
            load_type = load_point.get('type', 'unknown')
            connection_point = load_point['connection_point']

            if load_type == 'transformer':
                # Create transformer connection - transformer is already created elsewhere
                # This connects the MV node to the transformer bus
                transformer_id = load_point.get(
                    'transformer_id', load_point['name'])

                # Short service connection from MV node to transformer
                service_distance_km = 0.005  # 5m typical service connection

                # Calculate current requirement for transformer service
                # (conservative estimate)
                transformer_load_kw = float(load_point.get(
                    'load_kw', 400))  # Default 400kW transformer
                service_current_a = (
                    transformer_load_kw * 1000) / (np.sqrt(3) * 20000 * 0.9)

                # Select appropriate service cable
                service_cable_name, service_cable_equipment = self._select_smallest_mv_cable(
                    required_current_a=service_current_a,
                    equipment_lookup=equipment_lookup
                )

                # Generate coordinates for transformer service
                transformer_vertex_id = load_point.get(
                    'vertex_id', connection_point)
                coordinates = self._generate_line_coordinates(
                    connection_point, transformer_vertex_id)

                transformer_service_spec = LineSpec(
                    name=f"MV_Service_T{transformer_id}_{cluster_id}",
                    cable_equipment=service_cable_equipment,
                    bus1=f"MV_Node_{connection_point}_{cluster_id}",
                    # Connect to MV side of transformer
                    bus2=f"Trafo_{transformer_id}_{cluster_id}_MV",
                    length_km=service_distance_km,
                    parallel=1,
                    coordinates=coordinates
                )
                line_specs.append(transformer_service_spec)

                # Track service cable usage
                cable_usage[service_cable_name] = cable_usage.get(
                    service_cable_name, 0) + service_distance_km

            elif load_type == 'mv_building':
                # MV building connections will be handled by _install_mv_consumer_cables
                # The trunk infrastructure is now in place for them to connect
                # to
                pass

        self.logger.debug(
            f"MV branch {branch_id} installed: {
                len(sorted_load_points)} load points, "
            f"trunk cable: {trunk_cable_name}, total usage: {
                sum(
                    cable_usage.values()):.2f}km"
        )

        return line_specs, cable_usage

    def _install_final_mv_branch(self, final_load_point: dict, mv_main_bus: str,
                                 cluster_id: str, substation_vertex_id: int,
                                 mv_vertices_dict: Dict[int, float],
                                 equipment_lookup: callable) -> Tuple[List[ComponentSpec], Dict[str, float]]:
        """
        Install cables for the final remaining MV load point.

        This handles the case when only one MV load point remains, creating a direct
        connection from substation to the load point.

        Args:
            final_load_point: The last remaining MV load point
            mv_main_bus: Main MV bus name at substation
            cluster_id: Cluster identifier for naming
            substation_vertex_id: Vertex ID of substation connection
            mv_vertices_dict: Distance mapping from substation
            equipment_lookup: Function to get equipment specifications

        Returns:
            Tuple of (line_specs, cable_usage)
        """
        line_specs = []
        cable_usage = {}

        connection_point = final_load_point['connection_point']
        load_kw = float(final_load_point['load_kw'])
        load_type = final_load_point.get('type', 'unknown')

        # Calculate required current for this single load point
        required_current_a = (load_kw * 1000) / (np.sqrt(3) * 20000 * 0.9)

        # Get distance from substation using vertex_id
        vertex_id = final_load_point.get('vertex_id', connection_point)
        distance_m = mv_vertices_dict.get(vertex_id, 100.0)  # Default 100m
        distance_km = max(distance_m / 1000.0, 0.001)  # Minimum 1m

        # Select appropriate MV cable for this load
        cable_name, parallel_count = self.find_optimal_mv_cable(
            required_current_a=required_current_a,
            distance_km=distance_km
        )

        if not cable_name:
            # Fallback cable selection
            cable_name = "AL50"  # Conservative choice for final branch
            parallel_count = 1

        cable_equipment = equipment_lookup(cable_name)

        # Note: MV bus is already created in create_mv_network_components
        # No need to create duplicate bus here

        # Create direct connection from substation to final load point
        coordinates = self._generate_line_coordinates(
            substation_vertex_id, connection_point)

        final_line_spec = LineSpec(
            name=f"MV_Final_{connection_point}_{cluster_id}",
            cable_equipment=cable_equipment,
            bus1=mv_main_bus,
            bus2=f"MV_Node_{connection_point}_{cluster_id}",
            length_km=distance_km,
            parallel=parallel_count,
            coordinates=coordinates
        )
        line_specs.append(final_line_spec)

        # Track cable usage
        cable_usage[cable_name] = cable_usage.get(cable_name, 0) + distance_km

        # Handle the actual load connection (transformer or MV building)
        if load_type == 'transformer':
            transformer_id = final_load_point.get(
                'transformer_id', final_load_point['name'])

            # Short service connection from MV node to transformer
            service_distance_km = 0.005  # 5m typical service connection

            # Calculate current requirement for transformer service
            transformer_load_kw = load_kw
            service_current_a = (transformer_load_kw * 1000) / \
                (np.sqrt(3) * 20000 * 0.9)

            # Select appropriate service cable
            service_cable_name, service_cable_equipment = self._select_smallest_mv_cable(
                required_current_a=service_current_a,
                equipment_lookup=equipment_lookup
            )

            # Generate coordinates for transformer service
            transformer_vertex_id = final_load_point.get(
                'vertex_id', connection_point)
            coordinates = self._generate_line_coordinates(
                connection_point, transformer_vertex_id)

            transformer_service_spec = LineSpec(
                name=f"MV_Service_T{transformer_id}_{cluster_id}",
                cable_equipment=service_cable_equipment,
                bus1=f"MV_Node_{connection_point}_{cluster_id}",
                # Connect to MV side of transformer
                bus2=f"Trafo_{transformer_id}_{cluster_id}_MV",
                length_km=service_distance_km,
                parallel=1,
                coordinates=coordinates
            )
            line_specs.append(transformer_service_spec)

            # Track service cable usage
            cable_usage[service_cable_name] = cable_usage.get(
                service_cable_name, 0) + service_distance_km

        elif load_type == 'mv_building':
            # MV building connections will be handled by _install_mv_consumer_cables
            # The infrastructure is now in place for them to connect to
            pass

        self.logger.debug(
            f"Final MV load point installed: {
                load_kw:.0f}kW, {
                required_current_a:.1f}A, "
            f"cable: {cable_name}, distance: {distance_km:.3f}km"
        )

        return line_specs, cable_usage

    def _select_smallest_mv_cable(
            self, required_current_a: float, equipment_lookup: callable) -> Tuple[str, object]:
        """
        Select the smallest available MV cable that can handle the required current.

        This is used for MV transformer service connections where we want minimal cable sizing.
        Uses only cables available in the equipment_data table.

        Args:
            required_current_a: Required current capacity in Amperes
            equipment_lookup: Function to get equipment specifications

        Returns:
            Tuple of (cable_name, cable_equipment)
        """
        # Use the standard MV cable selection which already handles fallbacks
        cable_name, parallel_count = self.find_optimal_mv_cable(
            required_current_a=required_current_a,
            distance_km=0.005  # Short service connection
        )

        if cable_name:
            return cable_name, equipment_lookup(cable_name)

        # If the MV cable selection fails, this indicates a configuration issue
        # since find_optimal_mv_cable should have robust fallbacks
        self.logger.error(
            f"MV cable selection failed for {required_current_a}A service connection")
        raise ValueError(
            f"No suitable MV cable available from equipment_data table for {required_current_a}A current requirement")

    def _get_consumer_simultaneous_load_dict(
        self,
        consumer_list: List[int],
        buildings_df: pd.DataFrame,
        consumer_df: pd.DataFrame
    ) -> Tuple[Dict[int, float], Dict[int, int], Dict[int, str]]:
        """Calculate simultaneous load for each consumer."""
        Pd = {consumer: 0 for consumer in consumer_list}
        load_units = {consumer: 0 for consumer in consumer_list}
        load_type = {consumer: "SFH" for consumer in consumer_list}

        for row in buildings_df.itertuples():
            load_units[row.vertice_id] = row.houses_per_building
            load_type[row.vertice_id] = row.type

            # Look up simultaneity factor
            gzf = CONSUMER_CATEGORIES.loc[
                CONSUMER_CATEGORIES.definition == row.type, "sim_factor"
            ].item()

            # Calculate simultaneous load in MW
            Pd[row.vertice_id] = utils.oneSimultaneousLoad(
                row.peak_load_in_kw * 1e-3, row.houses_per_building, gzf
            )

        return Pd, load_units, load_type

    def _find_furthest_node_path_list(
        self,
        connection_nodes: List[int],
        vertices_dict: Dict[int, float],
        transformer_vertex: int
    ) -> List[int]:
        """Find path to furthest node from transformer."""
        if not connection_nodes:
            return []

        # Find node with maximum distance to transformer
        furthest_node = max(
            connection_nodes,
            key=lambda x: vertices_dict.get(
                x,
                0))

        # For simplicity, return direct path (in real implementation,
        # this would use graph algorithms to find actual path)
        return [furthest_node]

    def _determine_maximum_load_branch(
        self,
        node_path: List[int],
        buildings_df: pd.DataFrame,
        consumer_df: pd.DataFrame
    ) -> Tuple[List[int], float]:
        """Determine maximum load branch that can be served by heaviest cable."""
        if not node_path:
            return [], 0.0

        # Calculate simultaneous load for nodes in path
        sim_load = utils.simultaneousPeakLoad(
            buildings_df, consumer_df, node_path)

        # Calculate maximum current (3-phase)
        max_current = sim_load / (VN * V_BAND_LOW * np.sqrt(3))

        return node_path, max_current

    def _install_final_branch(
        self,
        final_node: int,
        lv_bus: str,
        cluster_id: str,
        transformer_vertex: int,
        vertices_dict: Dict[int, float],
        buildings_df: pd.DataFrame,
        consumer_df: pd.DataFrame,
        equipment_lookup: callable
    ) -> Tuple[List[ComponentSpec], Dict[str, float]]:
        """Install cables for the final remaining node."""
        # Calculate load for final node
        sim_load = utils.simultaneousPeakLoad(
            buildings_df, consumer_df, [final_node])
        max_current = sim_load / (VN * V_BAND_LOW * np.sqrt(3))

        # Select appropriate cable
        cable_name, parallel_count = self._find_minimal_cable(
            max_current, equipment_lookup)
        cable_equipment = equipment_lookup(cable_name)

        # Calculate connection distance
        if final_node == transformer_vertex:
            distance = 0.001  # Minimal distance for direct connection
        else:
            # Calculate distance between final node and transformer
            distance = self._calculate_node_distance(
                final_node, transformer_vertex, vertices_dict)

        # Generate coordinates for the line
        coordinates = self._generate_line_coordinates(
            final_node, transformer_vertex)

        # Create line specification
        line_spec = LineSpec(
            name=f"Line_Final_{final_node}_{cluster_id}",
            cable_equipment=cable_equipment,
            bus1=f"Bus_Node_{final_node}_{cluster_id}",
            bus2=lv_bus,
            length_km=distance,
            parallel=parallel_count,
            coordinates=coordinates
        )

        cable_usage = {cable_name: distance}
        return [line_spec], cable_usage

    def _install_branch_cables(
        self,
        branch_nodes: List[int],
        lv_bus: str,
        cluster_id: str,
        transformer_vertex: int,
        vertices_dict: Dict[int, float],
        load_dict: Dict[int, float],
        max_current: float,
        equipment_lookup: callable,
        branch_id: int
    ) -> Tuple[List[ComponentSpec], Dict[str, float]]:
        """Install cables for a branch of nodes."""
        line_specs = []
        cable_usage = {}

        # Select cable for main branch
        cable_name, parallel_count = self._find_minimal_cable(
            max_current, equipment_lookup)
        cable_equipment = equipment_lookup(cable_name)

        # Connect branch nodes in sequence
        for i, node in enumerate(branch_nodes[:-1]):
            next_node = branch_nodes[i + 1]
            # Calculate distance between consecutive nodes
            distance = self._calculate_node_distance(
                node, next_node, vertices_dict)

            # Generate coordinates for the line segment
            coordinates = self._generate_line_coordinates(node, next_node)

            line_spec = LineSpec(
                name=f"Line_Branch{branch_id}_Seg{i}_{cluster_id}",
                cable_equipment=cable_equipment,
                bus1=f"Bus_Node_{node}_{cluster_id}",
                bus2=f"Bus_Node_{next_node}_{cluster_id}",
                length_km=distance,
                parallel=parallel_count,
                coordinates=coordinates
            )
            line_specs.append(line_spec)

            cable_usage[cable_name] = cable_usage.get(cable_name, 0) + distance

        # Connect branch start to LV bus
        if branch_nodes:
            start_node = branch_nodes[-1]
            if start_node == transformer_vertex:
                distance = 0.001  # Direct connection
            else:
                # Calculate distance between start node and transformer
                distance = self._calculate_node_distance(
                    start_node, transformer_vertex, vertices_dict)

            # Generate coordinates for the line
            coordinates = self._generate_line_coordinates(
                start_node, transformer_vertex)

            line_spec = LineSpec(
                name=f"Line_Branch{branch_id}_Main_{cluster_id}",
                cable_equipment=cable_equipment,
                bus1=f"Bus_Node_{start_node}_{cluster_id}",
                bus2=lv_bus,
                length_km=distance,
                parallel=1,
                coordinates=coordinates
            )
            line_specs.append(line_spec)
            cable_usage[cable_name] = cable_usage.get(cable_name, 0) + distance

        return line_specs, cable_usage

    def _find_minimal_cable(self, max_current: float, equipment_lookup: callable,
                            distance_km: float = 0) -> Tuple[str, int]:
        """Find the minimum cable that can handle the given current using voltage-aware selection."""

        if self.use_voltage_aware_selection and self.cable_selector:
            # Use enhanced voltage-aware cable selection
            cable, parallel_count = self.cable_selector.find_optimal_cable(
                required_current_a=max_current,
                voltage_level='LV',  # LV networks in this context
                distance_km=distance_km,
                application_area=None,  # Could be enhanced with settlement type detection
                voltage_drop_limit_pct=4.5,
                base_voltage_v=400
            )

            if cable:
                return cable.name, parallel_count

        # Fallback to equipment database query for LV cables
        available_lv_cables = self._get_available_cables_by_voltage_level('LV')
        for cable_name in available_lv_cables:
            try:
                cable_eq = equipment_lookup(cable_name)
                # Check current capacity (handle both max_i_a and max_i_ka
                # attributes like MV method)
                max_current_a = None
                if hasattr(cable_eq, 'max_i_a'):
                    max_current_a = cable_eq.max_i_a
                elif hasattr(cable_eq, 'max_i_ka'):
                    max_current_a = cable_eq.max_i_ka * 1000  # Convert kA to A

                if max_current_a and max_current_a >= max_current:
                    return cable_name, 1
            except (AttributeError, TypeError, KeyError, ValueError) as e:
                self.logger.debug(f"Error checking LV cable {cable_name}: {e}")
                continue

        # Final fallback - use first available LV cable
        if available_lv_cables:
            self.logger.warning(
                f"Using fallback LV cable: {
                    available_lv_cables[0]} for {max_current}A requirement")
            return available_lv_cables[0], 1
        else:
            self.logger.error("No LV cables found in equipment_data table")
            raise ValueError(
                "No LV cables available in equipment_data table. Cannot build LV network without LV cables.")

    def find_optimal_mv_cable(self, required_current_a: float, distance_km: float = 0,
                              application_area: Optional[int] = None) -> Tuple[Optional[str], int]:
        """
        Find optimal MV cable using voltage-aware selection.

        This method specifically handles MV cable selection for 20kV networks.

        Args:
            required_current_a: Required current capacity in Amperes
            distance_km: Cable length in kilometers
            application_area: Optional settlement type (1=rural, 2=suburban, 3=urban)

        Returns:
            Tuple of (cable_name, parallel_count) or (None, 0) if no suitable cable found
        """
        if self.use_voltage_aware_selection and self.cable_selector:
            cable, parallel_count = self.cable_selector.find_optimal_cable(
                required_current_a=required_current_a,
                voltage_level='MV',  # MV networks
                distance_km=distance_km,
                application_area=application_area,
                voltage_drop_limit_pct=2.5,  # Stricter for MV
                base_voltage_v=20000  # 20kV MV
            )

            if cable:
                return cable.name, parallel_count

        # Fallback - query actual MV cables from equipment_data table
        self.logger.warning(
            "No voltage-aware selector available, querying equipment database for MV cables")

        # Get available MV cables from equipment_data table
        available_mv_cables = self._get_available_cables_by_voltage_level('MV')

        # Find smallest suitable MV cable
        for cable_name in available_mv_cables:
            try:
                cable_eq = equipment_lookup(cable_name)
                # Check current capacity (handle both max_i_a and max_i_ka
                # attributes)
                max_current_a = None
                if hasattr(cable_eq, 'max_i_a'):
                    max_current_a = cable_eq.max_i_a
                elif hasattr(cable_eq, 'max_i_ka'):
                    max_current_a = cable_eq.max_i_ka * 1000  # Convert kA to A

                if max_current_a and max_current_a >= required_current_a:
                    return cable_name, 1
            except (AttributeError, TypeError, KeyError, ValueError) as e:
                self.logger.debug(f"Error checking MV cable {cable_name}: {e}")
                continue

        # Final fallback - use first available MV cable
        if available_mv_cables:
            self.logger.warning(
                f"Using fallback MV cable: {
                    available_mv_cables[0]} for {required_current_a}A requirement")
            return available_mv_cables[0], 1
        else:
            # No MV cables available - this is a configuration error, stop
            # execution
            self.logger.error("No MV cables found in equipment_data table")
            raise ValueError(
                "No MV cables available in equipment_data table. Cannot build MV network without MV cables.")

    def _get_available_cables_by_voltage_level(
            self, voltage_level: str) -> List[str]:
        """
        Get available cable names from equipment_data table by voltage level.

        Args:
            voltage_level: 'MV' or 'LV'

        Returns:
            List of cable names available for the specified voltage level
        """
        if not self.database:
            return []

        try:
            # Query equipment_data table for cables of specified voltage level
            query = """
            SELECT name
            FROM equipment_data
            WHERE type = 'Cable'
            AND voltage_level = %s
            ORDER BY name
            """

            self.database.cur.execute(query, (voltage_level,))
            rows = self.database.cur.fetchall()

            cable_names = [row[0] for row in rows]

            if not cable_names:
                self.logger.error(
                    f"No {voltage_level} cables found in equipment_data table")
            else:
                self.logger.debug(
                    f"Found {
                        len(cable_names)} {voltage_level} cables in equipment_data")

            return cable_names

        except Exception as e:
            self.logger.warning(
                f"Error querying {voltage_level} cables from equipment_data: {e}")
            return []

    def _get_node_coordinates(
            self, node_id: int) -> Optional[Tuple[float, float]]:
        """Get coordinates for a node from database."""
        if not self.database:
            return None

        try:
            coord = self.database.get_node_geom(node_id)
            if coord:
                return (float(coord[0]), float(coord[1]))
        except Exception as e:
            self.logger.debug(
                f"Could not get coordinates for node {node_id}: {e}")

        return None

    def _install_consumer_cables(
            self,
            cluster_id: str,
            connection_nodes: List[int],
            consumer_list: List[int],
            transformer_vertex: int,
            vertices_dict: Dict,
            Pd: Dict,
            buildings_df: pd.DataFrame,
            equipment_lookup: callable
    ) -> List[ComponentSpec]:
        """
        Install cables from connection points to consumer buildings (house connections).

        This creates the final segment from the street connection point to each building.

        Args:
            cluster_id: Cluster identifier
            connection_nodes: List of connection point vertices
            consumer_list: List of consumer/building vertices
            transformer_vertex: Transformer location vertex
            vertices_dict: Mapping of vertex to distance from transformer
            Pd: Power demand dictionary for each consumer
            buildings_df: Building data
            equipment_lookup: Function to get cable equipment

        Returns:
            List of LineSpec for consumer connections
        """
        line_specs = []

        # Get consumers that need to be connected through each connection point
        # This uses the database method to find which consumers connect through
        # which connection points
        if not self.database:
            self.logger.warning(
                "No database connection for consumer cable routing")
            return line_specs

        for connection_point in connection_nodes:
            # Get consumers that connect through this connection point
            consumer_vertices = self.database.get_vertices_from_connection_points([
                                                                                  connection_point])

            # Filter to only consumers in our cluster
            branch_consumers = [
                v for v in consumer_vertices if v in consumer_list]

            for consumer_vertex in branch_consumers:
                # Get the path from consumer to transformer to find the
                # connection point
                path_nodes = self.database.get_path_to_bus(
                    consumer_vertex, transformer_vertex)

                if len(path_nodes) < 2:
                    continue

                # path_nodes[0] is the consumer, path_nodes[1] is the
                # connection point
                if path_nodes[1] != connection_point:
                    continue  # Skip if not the right connection point

                # Calculate distance and required current
                distance_km = self._calculate_node_distance(
                    consumer_vertex, connection_point, vertices_dict
                )

                # Get power demand and calculate current
                sim_load_mw = Pd.get(consumer_vertex, 0)
                current_a = (sim_load_mw * 1000) / \
                    (VN * np.sqrt(3))  # Convert MW to A

                # Select appropriate cable for house connection
                cable_name, parallel_count = self._find_minimal_cable(
                    max_current=current_a,
                    equipment_lookup=equipment_lookup,
                    distance_km=distance_km
                )

                # Get cable equipment
                cable_equipment = equipment_lookup(cable_name)

                # Generate coordinates for the consumer connection line
                coordinates = self._generate_line_coordinates(
                    consumer_vertex, connection_point)

                # Create line specification from connection point to consumer
                line_spec = LineSpec(
                    name=f"Line_Consumer_{consumer_vertex}_{cluster_id}",
                    cable_equipment=cable_equipment,
                    bus1=f"Bus_Node_{connection_point}_{cluster_id}",
                    bus2=f"Bus_Consumer_{consumer_vertex}_{cluster_id}",
                    length_km=distance_km,
                    parallel=parallel_count,
                    coordinates=coordinates
                )
                line_specs.append(line_spec)

                self.logger.debug(
                    f"Connected consumer {consumer_vertex} to connection point {connection_point} "
                    f"with {cable_name} cable ({distance_km:.3f}km)"
                )

        return line_specs

    def _install_mv_consumer_cables(
        self,
        cluster_id: str,
        mv_buildings: List[dict],
        equipment_lookup: callable
    ) -> List[ComponentSpec]:
        """
        Install MV consumer cables from connection points to commercial building centroids.

        This follows exactly the same pattern as LV consumer cables but for MV buildings.
        """
        line_specs = []

        if not self.database:
            return line_specs

        # Create buses and loads for MV buildings first
        for mv_building in mv_buildings:
            building_osm_id = mv_building['osm_id']
            building_vertex_id = mv_building.get(
                'vertice_id', building_osm_id)  # Use vertice_id for centroid
            # Convert Decimal to float
            peak_load_kw = float(mv_building['peak_load_kw'])

            # Create bus for MV building at its centroid
            building_bus_spec = BusSpec(
                name=f"MV_Building_{building_osm_id}_{cluster_id}",
                coordinates=self._get_node_coordinates(building_vertex_id),
                voltage_kv=20.0
            )
            line_specs.append(building_bus_spec)

            # Create load for MV building
            building_load_spec = LoadSpec(
                name=f"MV_Load_{building_osm_id}_{cluster_id}",
                bus=f"MV_Building_{building_osm_id}_{cluster_id}",
                kw=peak_load_kw,
                kvar=peak_load_kw * 0.3,
                kv=20.0,
                n_phases=3,
                conn="wye",
                building_id=str(building_osm_id)
            )
            line_specs.append(building_load_spec)

        # Now create the consumer cables (exactly like LV pattern)
        for mv_building in mv_buildings:
            connection_point = mv_building['connection_point']
            building_osm_id = mv_building['osm_id']
            building_vertex_id = mv_building.get(
                'vertice_id', building_osm_id)  # Use vertice_id for centroid
            # Convert Decimal to float
            peak_load_kw = float(mv_building['peak_load_kw'])

            # Calculate distance between connection point and building centroid
            distance_km = 0.05  # Default MV service distance
            try:
                if self.database:
                    # Use database routing to get actual distance
                    path_nodes = self.database.get_path_to_bus(
                        building_vertex_id, connection_point)
                    if len(path_nodes) >= 2:
                        # Get coordinates for distance calculation
                        building_coord = self.database.get_node_geom(
                            building_vertex_id)
                        connection_coord = self.database.get_node_geom(
                            connection_point)

                        if building_coord and connection_coord:
                            dx = float(building_coord[0]) - \
                                float(connection_coord[0])
                            dy = float(building_coord[1]) - \
                                float(connection_coord[1])
                            distance_km = max(
                                np.sqrt(dx * dx + dy * dy) / 1000.0, 0.01)
            except BaseException:
                pass

            # Calculate required current (MV: 20kV)
            current_a = peak_load_kw * 1000 / (np.sqrt(3) * 20000 * 0.9)

            # Select MV cable - MUST use MV method, not LV method
            cable_name, parallel_count = self.find_optimal_mv_cable(
                required_current_a=current_a,
                distance_km=distance_km
            )

            if cable_name:
                cable_equipment = equipment_lookup(cable_name)

                # Generate coordinates for MV service line from connection
                # point to building centroid
                coordinates = self._generate_line_coordinates(
                    connection_point, building_vertex_id)

                # Create MV service line
                line_spec = LineSpec(
                    name=f"MV_Consumer_{building_osm_id}_{cluster_id}",
                    cable_equipment=cable_equipment,
                    bus1=f"MV_Node_{connection_point}_{cluster_id}",
                    bus2=f"MV_Building_{building_osm_id}_{cluster_id}",
                    length_km=distance_km,
                    parallel=parallel_count,
                    coordinates=coordinates
                )
                line_specs.append(line_spec)

        return line_specs

    def _calculate_node_distance(
            self, node1: int, node2: int, vertices_dict: Dict) -> float:
        """Calculate distance between two nodes using routing costs from vertices_dict.

        vertices_dict contains: vertex_id -> distance_from_transformer (in meters)
        """
        try:
            # Get routing distances from transformer for both nodes
            dist1 = vertices_dict.get(node1)
            dist2 = vertices_dict.get(node2)

            if dist1 is None or dist2 is None:
                self.logger.warning(
                    f"Missing distance data for nodes {node1} or {node2}, using default")
                return 0.1  # Default 100m

            # Calculate difference in routing distance and convert to km
            # This gives approximate distance between the two nodes
            distance_m = abs(float(dist1) - float(dist2))
            distance_km = distance_m / 1000.0

            # Ensure minimum realistic distance
            return max(distance_km, 0.01)  # At least 10m

        except (TypeError, ValueError) as e:
            self.logger.warning(
                f"Error calculating distance between nodes {node1} and {node2}: {e}")
            return 0.1  # Default 100m

    def _update_cable_usage(
            self, local_usage: Dict[str, float], new_usage: Dict[str, float]) -> None:
        """Update local cable usage dictionary."""
        for cable, length in new_usage.items():
            local_usage[cable] = local_usage.get(cable, 0) + length
