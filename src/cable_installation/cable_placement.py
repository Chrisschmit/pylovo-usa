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
from ..electrical_backend.component_specs import (
    BusSpec,
    ComponentSpec,
    LineSpec,
    LoadSpec,
)
from .cable_selection import CableSelector
from ..equipment_schema import CableEquipment


class CablePlacementAlgorithm:
    """
    Implements cable placement algorithms for electrical distribution networks.

    Extracted from GridGenerator._install_cables_for_cluster with refactoring
    to work with component specifications instead of direct pandapower creation.
    """

    def __init__(
        self,
        database: Optional[DatabaseClient] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize cable placement algorithm with voltage-aware cable selection."""
        self.logger = logger or logging.getLogger(__name__)

        # Store database reference for coordinate generation
        self.database = database

        # Initialize voltage-aware cable selector if database is available
        if database:
            self.cable_selector = CableSelector(database, logger=self.logger)
            self.use_voltage_aware_selection = True
        else:
            self.cable_selector = None
            self.use_voltage_aware_selection = False
            self.logger.warning(
                "No database provided - falling back to static cable selection"
            )

    def route_or_fallback(
        self, a: int, b: int, stub_m: float = 5.0
    ) -> List[Tuple[float, float]]:
        """
        Return a polyline between vertices a and b with robust fallbacks.

        Order of attempts:
        1) Routed path via DB, converted to coordinates (if len>=2)
        2) Straight segment between endpoints if both coords exist and a!=b
        3) Short stub from whichever endpoint exists (or origin as last resort)
        Always returns at least 2 points.
        """
        # 1) Try routed polyline
        nodes = None
        if self.database and a != b:
            try:
                nodes = self.database.get_path_to_bus(a, b)
            except Exception:
                nodes = None

        if nodes and len(nodes) >= 2:
            coords: List[Tuple[float, float]] = []
            for nid in nodes:
                c = self.database.get_node_geom(nid)
                if c:
                    coords.append((float(c[0]), float(c[1])))
            if len(coords) >= 2:
                return coords

        # 2) Straight segment if both endpoints exist and are distinct
        a_c = self.database.get_node_geom(a) if self.database else None
        b_c = self.database.get_node_geom(b) if self.database else None
        if a_c and b_c and a != b:
            return [(float(a_c[0]), float(a_c[1])), (float(b_c[0]), float(b_c[1]))]

        # 3) Stub from whichever endpoint exists (or both if equal)
        m_per_deg_lat = 111_320.0
        dlat = stub_m / m_per_deg_lat
        if a_c:
            lon, lat = float(a_c[0]), float(a_c[1])
            return [(lon, lat), (lon, lat + dlat)]
        if b_c:
            lon, lat = float(b_c[0]), float(b_c[1])
            return [(lon, lat), (lon, lat + dlat)]

        # Last resort: origin stub
        return [(0.0, 0.0), (0.0, dlat)]

    def create_lv_network_components(
        self,
        cluster_id: str,
        lv_bus: str,
        vertices_dict: Dict[int, float],
        transformer_vertex: int,
        buildings_df: pd.DataFrame,
        consumer_df: pd.DataFrame,
        connection_nodes: List[int],
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
                    node_id
                ),  # TODO: Implement coordinate lookup
            )
            component_specs.append(bus_spec)

        # Create buses and loads for all consumers
        for consumer_id in consumer_list:
            # Bus for each consumer
            bus_spec = BusSpec(
                name=f"Bus_Consumer_{consumer_id}_{cluster_id}",
                coordinates=self._get_node_coordinates(consumer_id),
            )
            component_specs.append(bus_spec)

            # Compute proper LV load parameters for 3φ wye: use L-N base
            # voltage
            kw_val = Pd[consumer_id] * 1000  # MW → kW
            kvar_val = float(kw_val) * float(np.tan(np.arccos(POWER_FACTOR)))

            load_spec = LoadSpec(
                name=f"Load_{consumer_id}_{cluster_id}",
                bus=f"Bus_Consumer_{consumer_id}_{cluster_id}",
                kw=kw_val,
                kvar=kvar_val,
                kv=VN / 1000.0,
                n_phases=3,
                conn="wye",
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
                    remaining_nodes[0],
                    lv_bus,
                    cluster_id,
                    transformer_vertex,
                    vertices_dict,
                    buildings_df,
                    consumer_df,
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
                branch_nodes,
                lv_bus,
                cluster_id,
                transformer_vertex,
                vertices_dict,
                Pd,
                max_current,
                branch_counter,
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
                total_length:.1f}km total"
        )

        return component_specs

    def create_mv_network_components(
        self,
        cluster_id: str,
        mv_main_bus: str,
        lv_transformers: List[dict],
        mv_buildings: List[dict],
        substation_vertex_id: Optional[int] = None,
    ) -> List[ComponentSpec]:
        """
        Create component specifications for MV network using branch-by-branch algorithm.

        This method adapts the LV algorithm for MV networks, treating both MV buildings
        and distribution transformers as load points to be connected to the MV main bus.

        Args:
            cluster_id: Unique identifier (e.g., "MV_K1_S0")
            mv_main_bus: Name of the MV main bus (12.47kV)
            lv_transformers: List of distribution transformers with capacity and location
            mv_buildings: List of direct MV connections with load and location

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
            mv_load_points.append(
                {
                    "vertex_id": lv_tx["transformer_vertice_id"],
                    # Same as vertex_id for transformers
                    "connection_point": lv_tx["transformer_vertice_id"],
                    # Transformer capacity
                    "load_kw": lv_tx["transformer_rated_power"],
                    "type": "transformer",
                    "name": f"DistTx_B{lv_tx['bcid']}",
                    # For transformer connection reference
                    "transformer_id": lv_tx["bcid"],
                    "bcid": lv_tx["bcid"],
                }
            )

        # Add direct MV buildings as load points
        # Note: We use connection_point for MV feeder routing, not the building centroid
        # The service line from connection_point to building is handled
        # separately
        for mv_bldg in mv_buildings:
            mv_load_points.append(
                {
                    # Use connection point for feeder routing
                    "vertex_id": mv_bldg["connection_point"],
                    # Explicit connection point
                    "connection_point": mv_bldg["connection_point"],
                    "load_kw": mv_bldg["peak_load_kw"],
                    "type": "mv_building",  # Changed from 'building' to be more explicit
                    "name": f"MVBldg_{mv_bldg['osm_id']}",
                    "osm_id": mv_bldg["osm_id"],
                    # Store building centroid for reference
                    "building_vertex_id": mv_bldg.get("vertice_id"),
                }
            )

        self.logger.info(
            f"Planning MV network for {
                len(mv_load_points)} load points"
        )

        # Create buses for all MV load points this includes transformers and MV
        # buildings
        for load_point in mv_load_points:
            # Get coordinates for the load point vertex
            coordinates = None
            if self.database:
                try:
                    coord = self.database.get_node_geom(load_point["vertex_id"])
                    if coord:
                        coordinates = (float(coord[0]), float(coord[1]))
                except Exception as e:
                    self.logger.debug(
                        f"Could not get coordinates for vertex {
                            load_point['vertex_id']}: {e}"
                    )

            bus_spec = BusSpec(
                name=f"mv_node_{load_point['vertex_id']}_{cluster_id}",
                coordinates=coordinates,
                voltage_kv=12.47,  # MV voltage level
            )
            component_specs.append(bus_spec)

        # Apply MV cable placement using branch-by-branch approach
        mv_line_specs = self._install_mv_cables_branch_by_branch(
            mv_load_points,
            mv_main_bus,
            cluster_id,
            substation_vertex_id,
        )

        component_specs.extend(mv_line_specs)

        # After feeders are in place, add MV building buses + loads and their
        # MV_Consumer_* service lines (connection point -> building centroid)
        if mv_buildings:
            mv_building_specs = self._install_mv_consumer_cables(
                cluster_id=cluster_id,
                mv_buildings=mv_buildings,
            )
            component_specs.extend(mv_building_specs)

        total_mv_load = sum(lp["load_kw"] for lp in mv_load_points)
        self.logger.info(
            f"✓ MV network completed: {
                len(mv_line_specs)} lines, {
                total_mv_load:.0f}kW total load"
        )

        return component_specs

    def _install_mv_cables_branch_by_branch(
        self,
        mv_load_points: List[dict],
        mv_main_bus: str,
        cluster_id: str,
        substation_vertex_id: Optional[int] = None,
    ) -> List[ComponentSpec]:
        """
        Install MV cables using trunk+branch pattern (same as LV algorithm but for MV scale).

        This creates realistic MV distribution topology with shared trunks and branches,
        instead of inefficient radial connections from substation to each load point.

        Args:
            mv_load_points: List of MV load points (transformers + buildings)
            mv_main_bus: MV main bus name
            cluster_id: Network identifier
            substation_vertex_id: Substation location for routing

        Returns:
            List of LineSpec components for MV trunk+branch cables
        """
        line_specs = []

        if not mv_load_points or not substation_vertex_id:
            self.logger.warning(
                f"Cannot create MV network: no load points or substation location"
            )
            return line_specs

        # Step 1: Create distance mapping for MV load points (like
        # vertices_dict for LV)
        mv_vertices_dict = self._create_mv_vertices_dict(
            mv_load_points, substation_vertex_id
        )

        # Step 2: Apply branch-by-branch algorithm (same pattern as LV)
        remaining_load_points = mv_load_points.copy()
        branch_counter = 0
        local_cable_usage = {}

        self.logger.info(
            f"Building MV trunk+branch network for {len(mv_load_points)} load points"
        )

        while remaining_load_points:
            if len(remaining_load_points) == 1:
                # Handle final MV load point (same pattern as LV)
                final_specs, cable_usage = self._install_final_mv_branch(
                    remaining_load_points[0],
                    mv_main_bus,
                    cluster_id,
                    substation_vertex_id,
                    mv_vertices_dict,
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
                furthest_path
            )

            # Step 5: Install MV branch trunk + connections
            branch_specs, cable_usage = self._install_mv_branch_trunk(
                branch_load_points,
                mv_main_bus,
                cluster_id,
                substation_vertex_id,
                mv_vertices_dict,
                max_current,
                branch_counter,
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
        self, mv_load_points: List[dict], substation_vertex_id: int
    ) -> Dict[int, float]:
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
                "No database connection for MV routing distance calculation"
            )
            # Fallback: assign dummy distances
            for i, load_point in enumerate(mv_load_points):
                # 1km, 2km, 3km...
                mv_vertices_dict[load_point["vertex_id"]] = (i + 1) * 1000.0
            return mv_vertices_dict

        for load_point in mv_load_points:
            load_vertex = load_point["vertex_id"]

            try:
                _, length = self.database.get_path_to_bus_with_length(
                    load_vertex, substation_vertex_id
                )

                # Store distance in meters
                mv_vertices_dict[load_vertex] = length

            except Exception as e:
                self.logger.info(
                    f"Error calculating MV distance for vertex {load_vertex}: {e}"
                )
        # Log the created distance mapping
        self.logger.info(
            f"Created MV vertices dict: {len(mv_vertices_dict)} load points, "
            f"distances: {min(mv_vertices_dict.values()):.1f}-{max(mv_vertices_dict.values()):.1f}m"
        )

        return mv_vertices_dict

    def _find_furthest_mv_path(
        self,
        remaining_load_points: List[dict],
        mv_vertices_dict: Dict[int, float],
        substation_vertex_id: int,
    ) -> List[dict]:
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
            key=lambda lp: mv_vertices_dict.get(lp["vertex_id"], 0),
        )

        # Get the routing path from substation to furthest point
        if self.database:
            try:
                path_vertices = self.database.get_path_to_bus(
                    furthest_load_point["vertex_id"], substation_vertex_id
                )

                # Filter path to only include remaining load points (potential
                # branch nodes)
                path_load_points = []
                {lp["vertex_id"] for lp in remaining_load_points}

                for vertex_id in path_vertices:
                    # Find load point matching this vertex
                    matching_lp = next(
                        (
                            lp
                            for lp in remaining_load_points
                            if lp["vertex_id"] == vertex_id
                        ),
                        None,
                    )
                    if matching_lp:
                        path_load_points.append(matching_lp)

                # Return path in correct order (furthest first for
                # branch-by-branch)
                return path_load_points

            except Exception as e:
                self.logger.debug(f"Error finding MV path to furthest point: {e}")

        # Fallback: return just the furthest point
        return [furthest_load_point]

    def _determine_maximum_mv_branch(
        self, furthest_path: List[dict]
    ) -> Tuple[List[dict], float]:
        """
        Determine maximum MV branch that can be served by available MV cables.

        Uses actual cable specifications from equipment_data table instead of
        hard-coded limits. Finds the maximum current capacity available and
        aggregates loads until that limit is reached.

        Args:
            furthest_path: Path from substation to furthest MV load point

        Returns:
            Tuple of (branch_load_points, max_current)
        """
        if not furthest_path:
            return [], 0.0

        # Get maximum available MV cable capacity from selector
        max_cable = self.cable_selector.get_available_cables("MV")[0]

        mv_max_current_a = max_cable.max_i_a

        if mv_max_current_a == 0:
            self.logger.error("No MV cables found in equipment database")
            return [], 0.0

        branch_load_points = []

        # Accumulate load points starting from furthest
        for load_point in furthest_path:
            branch_load_points.append(load_point)

            # Calculate total load for current branch
            total_load_kw = sum(float(lp["load_kw"]) for lp in branch_load_points)

            required_current_a = (total_load_kw * 1000) / (
                np.sqrt(3) * 12470 * POWER_FACTOR
            )

            # Check if we've exceeded maximum available MV cable capacity
            if required_current_a >= mv_max_current_a and len(branch_load_points) > 1:
                # Remove the last load point that pushed us over the limit
                branch_load_points.remove(load_point)
                self.logger.debug(
                    f"MV branch size limited by available cable capacity: "
                    f"{required_current_a:.1f}A > {mv_max_current_a:.1f}A"
                )
                break
            elif (
                required_current_a >= mv_max_current_a and len(branch_load_points) == 1
            ):
                # Even single load point exceeds capacity - will need special
                # handling
                self.logger.warning(
                    f"Single MV load point {
                        load_point['name']} requires {
                        required_current_a:.1f}A "
                    f"(exceeds maximum available MV cable capacity of {
                        mv_max_current_a:.1f}A)"
                )
                break

        # Calculate final current for the selected branch
        if branch_load_points:
            total_branch_load_kw = sum(
                float(lp["load_kw"]) for lp in branch_load_points
            )
            max_current = (total_branch_load_kw * 1000) / (
                np.sqrt(3) * 12470 * POWER_FACTOR
            )
        else:
            max_current = 0.0

        self.logger.debug(
            f"MV branch determined: {len(branch_load_points)} load points, "
            f"{total_branch_load_kw:.0f}kW, {max_current:.1f}A "
            f"(max available: {mv_max_current_a:.1f}A)"
        )

        return branch_load_points, max_current

    def _install_mv_branch_trunk(
        self,
        branch_load_points: List[dict],
        mv_main_bus: str,
        cluster_id: str,
        substation_vertex_id: int,
        mv_vertices_dict: Dict[int, float],
        max_current: float,
        branch_id: int,
    ) -> Tuple[List[ComponentSpec], Dict[str, float]]:
        line_specs: List[ComponentSpec] = []
        cable_usage: Dict[str, float] = {}

        if not branch_load_points or not mv_vertices_dict:
            self.logger.warning("Missing load points or MV distance map")
            return line_specs, cable_usage

        # Trunk cable selection (unchanged logic)
        trunk_eq, trunk_parallel = self.find_optimal_mv_cable(
            required_current_a=max_current,
            distance_km=max(mv_vertices_dict.values()) / 1000.0,
        )

        # Order by distance from substation
        pts = sorted(
            branch_load_points,
            key=lambda lp: mv_vertices_dict.get(lp["vertex_id"], float("inf")),
        )

        # ---- Substation → first node ----
        if pts:
            first = pts[0]
            v_first = int(first["vertex_id"])
            d_first_km = mv_vertices_dict.get(v_first, 0.0) / 1000.0
            line_specs.append(
                LineSpec(
                    name=f"MV_Trunk_B{branch_id}_Main_{cluster_id}",
                    cable_equipment=trunk_eq,
                    bus1=mv_main_bus,
                    bus2=f"mv_node_{first['connection_point']}_{cluster_id}",
                    length_km=max(d_first_km, 1e-6),
                    parallel=trunk_parallel,
                    coordinates=self.route_or_fallback(substation_vertex_id, v_first),
                )
            )
            cable_usage[trunk_eq.name] = cable_usage.get(trunk_eq.name, 0.0) + max(
                d_first_km, 1e-6
            )

        # ---- Trunk between consecutive nodes ----
        for i in range(1, len(pts)):
            prev, curr = pts[i - 1], pts[i]
            v_prev, v_curr = int(prev["vertex_id"]), int(curr["vertex_id"])
            d_prev = mv_vertices_dict.get(v_prev, 0.0)
            d_curr = mv_vertices_dict.get(v_curr, 0.0)

            # skip zero-length hops in length only; still give a tiny geometry
            # if viz needs it?
            if v_prev == v_curr or d_prev == d_curr:
                continue

            seg_km = abs(d_curr - d_prev) / 1000.0
            line_specs.append(
                LineSpec(
                    name=f"MV_Trunk_B{branch_id}_S{i}_{cluster_id}",
                    cable_equipment=trunk_eq,
                    bus1=f"mv_node_{prev['connection_point']}_{cluster_id}",
                    bus2=f"mv_node_{curr['connection_point']}_{cluster_id}",
                    length_km=seg_km,
                    parallel=trunk_parallel,
                    coordinates=self.route_or_fallback(v_prev, v_curr),
                )
            )
            cable_usage[trunk_eq.name] = cable_usage.get(trunk_eq.name, 0.0) + seg_km

        # ---- Transformer service drops (fixed 5 m) ----
        for lp in pts:
            if lp.get("type") != "transformer":
                continue

            trafo_id = lp.get("transformer_id", lp.get("name"))
            load_kw = float(lp.get("load_kw", 400))
            service_current_a = (load_kw * 1000.0) / (np.sqrt(3) * 12470 * POWER_FACTOR)

            svc_eq, svc_eq = self._select_smallest_mv_cable(
                required_current_a=service_current_a
            )

            # trunk node vertex
            conn_vid = int(lp["connection_point"])
            trafo_vid = int(
                lp.get("vertex_id", lp["connection_point"])
            )  # transformer vertex
            svc_coords = self.route_or_fallback(
                conn_vid, trafo_vid, stub_m=5.0
            )  # <-- always yields 2 points

            svc_len_km = 0.005  # exactly 5 m
            line_specs.append(
                LineSpec(
                    name=f"MV_Service_T{trafo_id}_{cluster_id}",
                    cable_equipment=svc_eq,
                    bus1=f"mv_node_{lp['connection_point']}_{cluster_id}",
                    bus2=f"trafo_{trafo_id}_{cluster_id}_MV",
                    length_km=svc_len_km,
                    parallel=1,
                    coordinates=svc_coords,
                )
            )
            cable_usage[svc_name] = cable_usage.get(svc_name, 0.0) + svc_len_km

        self.logger.debug(
            "MV branch %s: %d lines, trunk %s, total %.3f km",
            branch_id,
            len(line_specs),
            trunk_eq.name,
            sum(cable_usage.values()),
        )
        return line_specs, cable_usage

    def _install_final_mv_branch(
        self,
        final_load_point: dict,
        mv_main_bus: str,
        cluster_id: str,
        substation_vertex_id: int,
        mv_vertices_dict: Dict[int, float],
    ) -> Tuple[List[ComponentSpec], Dict[str, float]]:
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

        Returns:
            Tuple of (line_specs, cable_usage)
        """
        line_specs = []
        cable_usage = {}

        connection_point = final_load_point["connection_point"]
        load_kw = float(final_load_point["load_kw"])
        load_type = final_load_point.get("type", "unknown")

        # Calculate required current for this single load point
        required_current_a = (load_kw * 1000) / (np.sqrt(3) * 12470 * POWER_FACTOR)

        # Get distance from substation using vertex_id
        vertex_id = final_load_point.get("vertex_id", connection_point)
        distance_km = mv_vertices_dict.get(vertex_id) / 1000.0

        # Select appropriate MV cable for this load
        cable_eq, parallel_count = self.find_optimal_mv_cable(
            required_current_a=required_current_a, distance_km=distance_km
        )
        cable_equipment = cable_eq

        # Create direct connection from substation to final load point
        coordinates = self.route_or_fallback(substation_vertex_id, connection_point)

        final_line_spec = LineSpec(
            name=f"MV_Final_{connection_point}_{cluster_id}",
            cable_equipment=cable_equipment,
            bus1=mv_main_bus,
            bus2=f"mv_node_{connection_point}_{cluster_id}",
            length_km=distance_km,
            parallel=parallel_count,
            coordinates=coordinates,
        )
        line_specs.append(final_line_spec)

        # Track cable usage
        cable_usage[cable_equipment.name] = (
            cable_usage.get(cable_equipment.name, 0) + distance_km
        )

        # Handle the actual load connection (transformer or MV building)
        if load_type == "transformer":
            transformer_id = final_load_point.get(
                "transformer_id", final_load_point["name"]
            )

            # Short service connection from MV node to transformer
            service_distance_km = 0.005  # 5m typical service connection

            # Calculate current requirement for transformer service
            transformer_load_kw = load_kw
            service_current_a = (transformer_load_kw * 1000) / (
                np.sqrt(3) * 12470 * 0.9
            )

            # Select appropriate service cable
            service_cable_name, service_cable_equipment = (
                self._select_smallest_mv_cable(required_current_a=service_current_a)
            )

            # Generate coordinates for transformer service
            transformer_vertex_id = final_load_point.get("vertex_id", connection_point)
            coordinates = self.route_or_fallback(
                connection_point, transformer_vertex_id
            )

            transformer_service_spec = LineSpec(
                name=f"MV_Service_T{transformer_id}_{cluster_id}",
                cable_equipment=service_cable_equipment,
                bus1=f"mv_node_{connection_point}_{cluster_id}",
                # Connect to MV side of transformer
                bus2=f"trafo_{transformer_id}_{cluster_id}_mv",
                length_km=service_distance_km,
                parallel=1,
                coordinates=coordinates,
            )
            line_specs.append(transformer_service_spec)

            # Track service cable usage
            cable_usage[service_cable_name] = (
                cable_usage.get(service_cable_name, 0) + service_distance_km
            )

        elif load_type == "mv_building":
            # MV building connections will be handled by _install_mv_consumer_cables
            # The infrastructure is now in place for them to connect to
            pass

        self.logger.debug(
            f"Final MV load point installed: {
                load_kw:.0f}kW, {
                required_current_a:.1f}A, "
            f"cable: {cable_equipment.name}, distance: {distance_km:.3f}km"
        )

        return line_specs, cable_usage

    def _select_smallest_mv_cable(
        self, required_current_a: float
    ) -> Tuple[CableEquipment, int]:
        """
        Select the smallest available MV cable that can handle the required current.

        This is used for MV transformer service connections where we want minimal cable sizing.
        Uses only cables available in the equipment_data table.

        Args:
            required_current_a: Required current capacity in Amperes

        Returns:
            Tuple of (cable_name, cable_equipment)
        """
        # Use the standard MV cable selection which already handles fallbacks
        cable_equipment, parallel_count = self.find_optimal_mv_cable(
            required_current_a=required_current_a,
            distance_km=0.005,  # Short service connection
        )

        if cable_equipment:
            return cable_equipment, parallel_count

        return None, 0

    def _get_consumer_simultaneous_load_dict(
        self,
        consumer_list: List[int],
        buildings_df: pd.DataFrame,
        consumer_df: pd.DataFrame,
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
        transformer_vertex: int,
    ) -> List[int]:
        """Find path to furthest node from transformer."""
        if not connection_nodes:
            return []

        # Find node with maximum distance to transformer
        furthest_node = max(connection_nodes, key=lambda x: vertices_dict.get(x, 0))

        # For simplicity, return direct path (in real implementation,
        # this would use graph algorithms to find actual path)
        return [furthest_node]

    def _determine_maximum_load_branch(
        self,
        node_path: List[int],
        buildings_df: pd.DataFrame,
        consumer_df: pd.DataFrame,
    ) -> Tuple[List[int], float]:
        """Determine maximum load branch that can be served by heaviest cable."""
        if not node_path:
            return [], 0.0

        # Calculate simultaneous load for nodes in path
        sim_load = utils.simultaneousPeakLoad(buildings_df, consumer_df, node_path)

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
    ) -> Tuple[List[ComponentSpec], Dict[str, float]]:
        """Install cables for the final remaining node."""
        # Load and current
        sim_load = utils.simultaneousPeakLoad(buildings_df, consumer_df, [final_node])
        max_current = sim_load / (VN * V_BAND_LOW * np.sqrt(3))

        # Cable selection
        cable_equipment, parallel_count = self._find_minimal_cable(max_current)

        # Length (km)
        if final_node == transformer_vertex:
            distance_km = 0.001  # 1 m minimum
        else:
            # keep your existing distance method (assumed to return km)
            distance_km = self._calculate_node_distance(
                final_node, transformer_vertex, vertices_dict
            )
            distance_km = max(distance_km, 0.001)  # avoid exact zero

        # Geometry: GUARANTEED coordinates (routed, straight, or 5 m stub)
        coordinates = self.route_or_fallback(final_node, transformer_vertex, stub_m=5.0)

        # Line spec
        line_spec = LineSpec(
            name=f"Line_Final_{final_node}_{cluster_id}",
            cable_equipment=cable_equipment,
            bus1=f"Bus_Node_{final_node}_{cluster_id}",
            bus2=lv_bus,
            length_km=distance_km,
            parallel=parallel_count,
            coordinates=coordinates,
        )

        return [line_spec], {cable_equipment.name: distance_km}

    def _install_branch_cables(
        self,
        branch_nodes: List[int],
        lv_bus: str,
        cluster_id: str,
        transformer_vertex: int,
        vertices_dict: Dict[int, float],
        load_dict: Dict[int, float],
        max_current: float,
        branch_id: int,
    ) -> Tuple[List[ComponentSpec], Dict[str, float]]:
        """Install cables for a branch of nodes."""
        line_specs = []
        cable_usage = {}

        # Select cable for main branch
        cable_eq, parallel_count = self._find_minimal_cable(max_current)
        cable_equipment = cable_eq

        # Connect branch nodes in sequence
        for i, node in enumerate(branch_nodes[:-1]):
            next_node = branch_nodes[i + 1]
            # Calculate distance between consecutive nodes
            distance = self._calculate_node_distance(node, next_node, vertices_dict)

            # Generate coordinates for the line segment
            coordinates = self.route_or_fallback(node, next_node)

            line_spec = LineSpec(
                name=f"Line_Branch{branch_id}_Seg{i}_{cluster_id}",
                cable_equipment=cable_equipment,
                bus1=f"Bus_Node_{node}_{cluster_id}",
                bus2=f"Bus_Node_{next_node}_{cluster_id}",
                length_km=distance,
                parallel=parallel_count,
                coordinates=coordinates,
            )
            line_specs.append(line_spec)

            cable_usage[cable_equipment.name] = (
                cable_usage.get(cable_equipment.name, 0) + distance
            )

        # Connect branch start to LV bus
        if branch_nodes:
            start_node = branch_nodes[-1]
            if start_node == transformer_vertex:
                distance = 0.001  # Direct connection
            else:
                # Calculate distance between start node and transformer
                distance = self._calculate_node_distance(
                    start_node, transformer_vertex, vertices_dict
                )

            # Generate coordinates for the line
            coordinates = self.route_or_fallback(start_node, transformer_vertex)

            line_spec = LineSpec(
                name=f"Line_Branch{branch_id}_Main_{cluster_id}",
                cable_equipment=cable_equipment,
                bus1=f"Bus_Node_{start_node}_{cluster_id}",
                bus2=lv_bus,
                length_km=distance,
                parallel=1,
                coordinates=coordinates,
            )
            line_specs.append(line_spec)
            cable_usage[cable_equipment.name] = (
                cable_usage.get(cable_equipment.name, 0) + distance
            )

        return line_specs, cable_usage

    def _find_minimal_cable(
        self, max_current: float, distance_km: float = 0
    ) -> Tuple[object, int]:
        """Find the minimum cable that can handle the given current using voltage-aware selection."""

        cable, parallel_count = self.cable_selector.find_optimal_cable(
            required_current_a=max_current,
            voltage_level="LV",  # LV networks in this context
            distance_km=distance_km,
            application_area=None,  # Could be enhanced with settlement type detection
        )

        return cable, parallel_count

    def find_optimal_mv_cable(
        self,
        required_current_a: float,
        distance_km: float = 0,
        application_area: Optional[int] = None,
    ) -> Tuple[Optional[CableEquipment], int]:
        """
        Find optimal MV cable using voltage-aware selection.

        This method specifically handles MV cable selection for 12.47kV networks.

        Args:
            required_current_a: Required current capacity in Amperes
            distance_km: Cable length in kilometers
            application_area: Optional settlement type (1=rural, 2=suburban, 3=urban)

        Returns:
            Tuple of (cable_name, parallel_count) or (None, 0) if no suitable cable found
        """

        cable, parallel_count = self.cable_selector.find_optimal_cable(
            required_current_a=required_current_a,
            voltage_level="MV",  # MV networks
            distance_km=distance_km,
            application_area=application_area,
        )

        return cable, parallel_count

    def _get_node_coordinates(self, node_id: int) -> Optional[Tuple[float, float]]:
        """Get coordinates for a node from database."""
        if not self.database:
            return None

        try:
            coord = self.database.get_node_geom(node_id)
            if coord:
                return (float(coord[0]), float(coord[1]))
        except Exception as e:
            self.logger.debug(f"Could not get coordinates for node {node_id}: {e}")

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

        Returns:
            List of LineSpec for consumer connections
        """
        line_specs = []

        # Get consumers that need to be connected through each connection point
        # This uses the database method to find which consumers connect through
        # which connection points
        if not self.database:
            self.logger.warning("No database connection for consumer cable routing")
            return line_specs

        for connection_point in connection_nodes:
            # Get consumers that connect through this connection point
            consumer_vertices = self.database.get_vertices_from_connection_points(
                [connection_point]
            )

            # Filter to only consumers in our cluster
            branch_consumers = [v for v in consumer_vertices if v in consumer_list]

            for consumer_vertex in branch_consumers:
                # Get the path from consumer to transformer to find the
                # connection point
                path_nodes = self.database.get_path_to_bus(
                    consumer_vertex, transformer_vertex
                )

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
                current_a = (sim_load_mw * 1000) / (VN * np.sqrt(3))  # Convert MW to A

                # Select appropriate cable for house connection
                cable_eq, parallel_count = self._find_minimal_cable(
                    max_current=current_a,
                    distance_km=distance_km,
                )

                cable_equipment = cable_eq

                # Generate coordinates for the consumer connection line
                coordinates = self.route_or_fallback(consumer_vertex, connection_point)

                # Create line specification from connection point to consumer
                line_spec = LineSpec(
                    name=f"Line_Consumer_{consumer_vertex}_{cluster_id}",
                    cable_equipment=cable_equipment,
                    bus1=f"Bus_Node_{connection_point}_{cluster_id}",
                    bus2=f"Bus_Consumer_{consumer_vertex}_{cluster_id}",
                    length_km=distance_km,
                    parallel=parallel_count,
                    coordinates=coordinates,
                )
                line_specs.append(line_spec)

                self.logger.debug(
                    f"Connected consumer {consumer_vertex} to connection point {connection_point} "
                    f"with {cable_equipment.name} cable ({distance_km:.3f}km)"
                )

        return line_specs

    def _install_mv_consumer_cables(
        self, cluster_id: str, mv_buildings: List[dict]
    ) -> List[ComponentSpec]:
        """
        Install MV consumer cables from trunk connection points to MV building buses.
        Creates one MV_Building bus + load per building, then a routed MV_Consumer line
        from mv_node_{connection_point} to mv_building_{osm_id}.
        """
        line_specs: List[ComponentSpec] = []

        def _get_vid(b: dict) -> int:
            # prefer 'vertex_id', fallback to 'vertice_id'; never fallback to
            # osm_id
            if "vertex_id" in b:
                return int(b["vertex_id"])
            if "vertice_id" in b:
                return int(b["vertice_id"])
            raise KeyError(
                f"mv_building {
                    b.get('osm_id')} missing 'vertex_id'/'vertice_id'"
            )

        def _coords_from_nodes(
            node_ids: List[int],
        ) -> Optional[List[Tuple[float, float]]]:
            if not node_ids or len(node_ids) < 2:
                return None
            coords: List[Tuple[float, float]] = []
            for nid in node_ids:
                # (lon, lat) in 4326 per your query
                c = self.database.get_node_geom(nid)
                if c:
                    coords.append((float(c[0]), float(c[1])))
            return coords if len(coords) >= 2 else None

        # 1) Buses + loads (one per building)
        for b in mv_buildings:
            osm_id = b["osm_id"]
            vid = _get_vid(b)
            peak_kw = float(b["peak_load_kw"])

            line_specs.append(
                BusSpec(
                    name=f"mv_building_{osm_id}_{cluster_id}",
                    coordinates=self._get_node_coordinates(vid),
                    voltage_kv=12.47,
                )
            )
            line_specs.append(
                LoadSpec(
                    name=f"mv_load_{osm_id}_{cluster_id}",
                    bus=f"mv_building_{osm_id}_{cluster_id}",
                    kw=peak_kw,
                    kvar=peak_kw * 0.3,
                    kv=12.47,
                    n_phases=3,
                    conn="delta",
                    building_id=str(osm_id),
                )
            )

        # 2) Service cables (connection_point -> building bus)
        for b in mv_buildings:
            osm_id = b["osm_id"]
            conn_pt = int(b["connection_point"])  # MV trunk node
            vid = _get_vid(b)  # building vertex (centroid)
            peak_kw = float(b["peak_load_kw"])

            # Routed path + length (meters) from building to connection point
            try:
                node_seq, length_m = self.database.get_path_to_bus_with_length(
                    vid, conn_pt
                )
            except Exception:
                node_seq, length_m = None, None

            # Distance in km; clamp zero to 1 m to avoid exact-zero lines in
            # sims
            distance_km = max(((length_m or 0.0) / 1000.0), 0.001)

            # MV sizing at 12.47 kV,
            current_a = (peak_kw * 1000.0) / (np.sqrt(3) * 12470 * POWER_FACTOR)

            cable_eq, parallel = self.find_optimal_mv_cable(
                required_current_a=current_a, distance_km=distance_km
            )
            if not cable_eq:
                continue

            # Build coordinates from the routed node sequence; if missing, skip
            # coords
            coordinates = _coords_from_nodes(node_seq) if node_seq else None

            line_specs.append(
                LineSpec(
                    name=f"MV_Consumer_{osm_id}_{cluster_id}",
                    cable_equipment=cable_eq,
                    bus1=f"mv_node_{conn_pt}_{cluster_id}",
                    bus2=f"mv_building_{osm_id}_{cluster_id}",
                    length_km=distance_km,
                    parallel=parallel,
                    coordinates=coordinates,
                )
            )

        return line_specs

    def _calculate_node_distance(
        self, node1: int, node2: int, vertices_dict: Dict
    ) -> float:
        """Calculate distance between two nodes using routing costs from vertices_dict.

        vertices_dict contains: vertex_id -> distance_from_transformer (in meters)
        """
        try:
            # Get routing distances from transformer for both nodes
            dist1 = vertices_dict.get(node1)
            dist2 = vertices_dict.get(node2)

            if dist1 is None or dist2 is None:
                self.logger.warning(
                    f"Missing distance data for nodes {node1} or {node2}, using default"
                )
                return 0.1  # Default 100m

            # Calculate difference in routing distance and convert to km
            # This gives approximate distance between the two nodes
            distance_m = abs(float(dist1) - float(dist2))
            distance_km = distance_m / 1000.0

            # Ensure minimum realistic distance
            return max(distance_km, 0.01)  # At least 10m

        except (TypeError, ValueError) as e:
            self.logger.warning(
                f"Error calculating distance between nodes {node1} and {node2}: {e}"
            )
            return 0.1  # Default 100m

    def _update_cable_usage(
        self, local_usage: Dict[str, float], new_usage: Dict[str, float]
    ) -> None:
        """Update local cable usage dictionary."""
        for cable, length in new_usage.items():
            local_usage[cable] = local_usage.get(cable, 0) + length
