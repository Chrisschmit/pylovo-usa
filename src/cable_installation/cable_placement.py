"""
Cable placement algorithms extracted from grid_generator.py

This module contains the core cable placement algorithms that determine optimal
cable routing and sizing for distribution networks. The algorithms use a
branch-by-branch approach starting from the furthest nodes and working toward
transformers.
"""

import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .. import utils
from ..config_loader import *
from ..database.database_client import DatabaseClient
from ..electrical_backend.component_specs import (BusSpec, ComponentSpec,
                                                  LineSpec, LoadSpec)
from ..equipment_schema import CableEquipment
from .cable_selection import CableSelector


class CablePlacementAlgorithm:
    """
    Implements cable placement algorithms for electrical distribution networks.

    Extracted from GridGenerator._install_cables_for_cluster with refactoring
    to work with component specifications instead of direct pandapower creation.
    """

    def __init__(
        self,
        dbc: Optional[DatabaseClient] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize cable placement algorithm with voltage-aware cable selection."""
        self.logger = logger or logging.getLogger(__name__)

        # Store database reference for coordinate generation
        self.dbc = dbc

        # Initialize voltage-aware cable selector if database is available
        if dbc:
            self.cable_selector = CableSelector(dbc, logger=self.logger)
            self.use_voltage_aware_selection = True
        else:
            self.cable_selector = None
            self.use_voltage_aware_selection = False
            self.logger.warning(
                "No database provided - falling back to static cable selection"
            )

    def route_or_fallback(
        self, a: int, b: int, stub_m: float = 0.5
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
        if self.dbc and a != b:
            try:
                nodes = self.dbc.get_path_to_bus(a, b)
            except Exception:
                nodes = None

        if nodes and len(nodes) >= 2:
            coords: List[Tuple[float, float]] = []
            for nid in nodes:
                c = self.dbc.get_node_geom(nid)
                if c:
                    coords.append((float(c[0]), float(c[1])))
            if len(coords) >= 2:
                return coords

        # 2) Straight segment if both endpoints exist and are distinct
        a_c = self.dbc.get_node_geom(a) if self.dbc else None
        b_c = self.dbc.get_node_geom(b) if self.dbc else None
        if a_c and b_c and a != b:
            return [(float(a_c[0]), float(a_c[1])),
                    (float(b_c[0]), float(b_c[1]))]

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

    def _offset_polyline_for_viz(
        self,
        coords: Optional[List[Tuple[float, float]]],
        deviation_deg: float = 5e-6,
        sign: float = 1.0,
    ) -> Optional[List[Tuple[float, float]]]:
        """
        Offset interior polyline vertices by a tiny amount for visualization.
        Keeps endpoints fixed so topology/connectivity remain correct.
        """
        if not coords or len(coords) < 3 or deviation_deg <= 0:
            return coords
        out: List[Tuple[float, float]] = [coords[0]]
        dx = sign * deviation_deg
        dy = sign * deviation_deg
        for x, y in coords[1:-1]:
            out.append((float(x) + dx, float(y) + dy))
        out.append(coords[-1])
        return out

    def plan_trunk_and_branches(
        self,
        *,
        level: str,
        cluster_id: str,
        hub_vertex_id: int,
        load_points: List[Dict],
        distance_lookup: Callable[[int], float],
        path_to_hub: Callable[[int, int], List[int]],
        calc_branch_current: Callable[[List[Dict]], float],
        select_trunk: Callable[[float, float], Tuple[Optional[CableEquipment], int]],
        select_service: Callable[[float], Tuple[Optional[CableEquipment], int]],
        route: Callable[[int, int], List[Tuple[float, float]]],
        logger: logging.Logger,
    ) -> List[List[Dict]]:
        """
        Generic greedy branch-by-branch planner used for both MV and LV.

        High-level algorithm (skeleton):
        - While load points remain:
          - Pick furthest by distance_lookup
          - Get routed path to hub via path_to_hub and filter to remaining load points
          - Grow branch from furthest inward while trunk selection remains feasible
          - Emit trunk and service components
          - Remove processed load points

        This skeleton prepares the structure; emitting of LineSpec/BusSpec is kept
        minimal for now and will be wired in subsequent steps.
        """

        planned_branches: List[List[Dict]] = []

        remaining = load_points.copy()
        branch_idx = 0

        def _vid(lp: Dict) -> int:
            # Common helper to get vertex id across MV/LV shapes
            if "vertex_id" in lp:
                return int(lp["vertex_id"])
            if "connection_point" in lp:
                return int(lp["connection_point"])  # LV nodes may pass this
            # As a last resort, treat lp itself as vertex id if it's an
            # int-like
            try:
                return int(lp)
            except Exception:
                raise KeyError(f"Cannot extract vertex_id from {lp}")

        while remaining:
            # Furthest first
            furthest_lp = max(
                remaining, key=lambda lp: float(
                    distance_lookup(_vid(lp)) or 0.0)
            )

            # Path vertices from furthest to hub
            try:
                path_vertices = list(
                    path_to_hub(
                        _vid(furthest_lp),
                        hub_vertex_id))
            except Exception:
                path_vertices = []

            # Filter path to remaining load points, ordered from furthest
            # inward
            path_lps: List[Dict] = []
            vid_to_lp = {_vid(lp): lp for lp in remaining}
            for v in path_vertices:
                if v in vid_to_lp:
                    path_lps.append(vid_to_lp[v])

            if not path_lps:
                # Fallback: treat only the furthest point as a branch of one
                path_lps = [furthest_lp]

            # Grow feasible branch
            branch: List[Dict] = []
            for lp in path_lps:
                trial_branch = branch + [lp]
                I_req = float(calc_branch_current(trial_branch) or 0.0)
                # Conservative trunk distance: max to hub across branch points
                try:
                    max_d_km = (
                        max(distance_lookup(_vid(x))
                            for x in trial_branch) / 1000.0
                    )
                except Exception:
                    max_d_km = 0.0

                trunk_eq, trunk_par = select_trunk(I_req, max_d_km)
                if trunk_eq is None or trunk_par <= 0:
                    break
                branch = trial_branch

            # Record planned branch for emission in caller
            planned_branches.append(branch)

            try:
                logger.debug(
                    "Planned %s branch %s with %d points (I≈%.1f A)",
                    level,
                    branch_idx,
                    len(branch),
                    float(calc_branch_current(branch) or 0.0),
                )
            except Exception:
                pass

            # Remove processed points
            for lp in branch:
                if lp in remaining:
                    remaining.remove(lp)

            branch_idx += 1

            # Safety valve to avoid infinite loops in skeleton state
            if branch_idx > 10_000:
                logger.warning(
                    "Aborting planner loop due to excessive iterations")
                break

        return planned_branches

    def create_lv_network_components(
        self,
        cluster_id: str,
        lv_bus: str,
        vertex_distance_mapping: Dict[int, float],
        transformer_vertex: int,
        buildings_df: pd.DataFrame,
        consumer_df: pd.DataFrame,
        connection_nodes: List[int],
        n_phases: int = 3,
    ) -> List[ComponentSpec]:
        """
        Create component specifications for LV network using branch-by-branch algorithm.

        This is the main algorithm extracted from _install_cables_for_cluster.

        Args:
            cluster_id: Unique identifier for this cluster (e.g. "K123_S456_B789")
            lv_bus: Name of the LV bus (transformer secondary side)
            vertex_distance_mapping: Mapping of vertex_id -> distance_to_transformer
            transformer_vertex: Vertex ID of transformer location
            buildings_df: DataFrame with building information
            consumer_df: DataFrame with consumer categories
            connection_nodes: List of connection point vertices (excluding buildings)
            n_phases: Number of phases (1 or 3) based on transformer configuration

        Returns:
            List of ComponentSpec objects for buses, lines, and loads
        """
        component_specs = []

        # Determine phase configuration based on transformer type
        if n_phases == 1:
            line_phases = "A"  # Single-phase on phase A
        else:
            line_phases = "ABC"  # 3-phase

        # Calculate load data for all consumers
        consumer_list = buildings_df.vertice_id.to_list()
        consumer_list = list(dict.fromkeys(consumer_list))

        simultaneous_load_kw = self._get_consumer_simultaneous_load_dict(
            consumer_list, buildings_df
        )

        # Create buses for all connection nodes
        for node_id in connection_nodes:
            bus_spec = BusSpec(
                name=f"Bus_Node_{node_id}_{cluster_id}",
                coordinates=self._get_node_coordinates(node_id),
                vertex_id=node_id,
            )
            component_specs.append(bus_spec)

        # Create buses and loads for all consumers
        for consumer_id in consumer_list:
            # Bus for each consumer
            bus_spec = BusSpec(
                name=f"Bus_Consumer_{consumer_id}_{cluster_id}",
                coordinates=self._get_node_coordinates(consumer_id),
                vertex_id=consumer_id,
            )
            component_specs.append(bus_spec)

            # Compute proper LV load parameters based on transformer
            # configuration
            kw_val = simultaneous_load_kw[consumer_id]
            kvar_val = float(kw_val) * float(np.tan(np.arccos(POWER_FACTOR)))

            # Set voltage and connection based on phase configuration
            if n_phases == 1:
                # Single-phase: use L-N voltage for wye connection
                load_kv = 0.120
                conn_type = "wye"
            else:
                # 3-phase: use L-N voltage for wye connection
                load_kv = 0.208
                conn_type = "wye"

            load_spec = LoadSpec(
                name=f"lv_load_{consumer_id}_{cluster_id}",
                bus=f"Bus_Consumer_{consumer_id}_{cluster_id}",
                kw=kw_val,
                kvar=kvar_val,
                kv=load_kv,
                n_phases=n_phases,
                conn=conn_type,
                vertex_id=consumer_id,
            )
            component_specs.append(load_spec)

        # First, install consumer cables from connection points to buildings
        consumer_specs = self._install_consumer_cables(
            cluster_id=cluster_id,
            connection_nodes=connection_nodes,
            consumer_list=consumer_list,
            transformer_vertex=transformer_vertex,
            vertex_distance_mapping=vertex_distance_mapping,
            power_demand=simultaneous_load_kw,
            n_phases=n_phases,
        )
        component_specs.extend(consumer_specs)

        # Use unified planner for LV trunk/branches (temporary integration)
        def lv_distance_lookup(vid: int) -> float:
            return float(vertex_distance_mapping.get(int(vid), 0.0))

        def lv_path_to_hub(vid: int, hub: int) -> List[int]:
            if not self.dbc:
                return [vid, hub]
            try:
                return self.dbc.get_path_to_bus(int(vid), int(hub))
            except Exception:
                return [vid, hub]

        def lv_calc_branch_current(branch_load_points: List[Dict]) -> float:
            node_ids: List[int] = []
            for lp in branch_load_points:
                if isinstance(lp, dict):
                    node_ids.append(
                        int(lp.get("vertex_id", lp.get("connection_point", lp)))
                    )
                else:
                    node_ids.append(int(lp))
            sim_load_mw = utils.simultaneousPeakLoad(
                buildings_df, consumer_df, node_ids
            )
            return float(sim_load_mw / (VN * V_BAND_LOW * np.sqrt(3)))

        def lv_select_trunk(
            I_req: float, distance_km: float
        ) -> Tuple[Optional[CableEquipment], int]:
            return self.cable_selector.find_optimal_cable(
                required_current_a=float(I_req or 0.0),
                voltage_level="LV",
                distance_km=float(distance_km or 0.0),
                n_phases=n_phases,
            )

        def lv_select_service(
                I_req: float) -> Tuple[Optional[CableEquipment], int]:
            return self.cable_selector.find_optimal_cable(
                required_current_a=float(I_req or 0.0),
                voltage_level="LV",
                distance_km=0.005,
                n_phases=n_phases,
            )

        lv_load_points = [{"vertex_id": v} for v in connection_nodes]

        branches = self.plan_trunk_and_branches(
            level="LV",
            cluster_id=cluster_id,
            hub_vertex_id=transformer_vertex,
            load_points=lv_load_points,
            distance_lookup=lv_distance_lookup,
            path_to_hub=lv_path_to_hub,
            calc_branch_current=lv_calc_branch_current,
            select_trunk=lv_select_trunk,
            select_service=lv_select_service,
            route=lambda a, b: self.route_or_fallback(a, b),
            logger=self.logger,
        )

        # Emit LV trunk lines for each planned branch
        for branch_id, branch in enumerate(branches):
            if not branch:
                continue
            I_req = lv_calc_branch_current(branch)
            try:
                max_d_km = (
                    max(lv_distance_lookup(
                        int(lp["vertex_id"])) for lp in branch)
                    / 1000.0
                )
            except Exception:
                max_d_km = 0.0
            trunk_eq, trunk_par = lv_select_trunk(I_req, max_d_km)
            if not trunk_eq:
                continue

            # Log parallel cable usage for LV trunk
            if trunk_par > 1:
                self.logger.info(
                    f"LV PARALLEL CABLES: Branch {branch_id} requires {trunk_par} parallel cables "
                    f"of type {
                        trunk_eq.name} (I_req={
                        I_req:.1f}A, distance={
                        max_d_km:.3f}km, "
                    f"single cable capacity={trunk_eq.max_i_a:.1f}A)"
                )
            else:
                self.logger.debug(
                    f"LV single cable: Branch {branch_id} using 1x {
                        trunk_eq.name} "
                    f"(I_req={I_req:.1f}A, capacity={trunk_eq.max_i_a:.1f}A)"
                )

            pts = sorted(
                branch, key=lambda lp: lv_distance_lookup(int(lp["vertex_id"]))
            )
            first_vid = int(pts[0]["vertex_id"]) if pts else transformer_vertex
            d_first_km = max(lv_distance_lookup(first_vid) / 1000.0, 0.001)
            coords = self.route_or_fallback(transformer_vertex, first_vid)
            coords = self._offset_polyline_for_viz(
                coords, deviation_deg=5e-6, sign=1.0 if (branch_id % 2 == 0) else -1.0
            )
            component_specs.append(
                LineSpec(
                    name=f"LV_Trunk_B{branch_id}_Main_{cluster_id}",
                    cable_equipment=trunk_eq,
                    bus1=lv_bus,
                    bus2=f"Bus_Node_{first_vid}_{cluster_id}",
                    length_km=d_first_km,
                    parallel=trunk_par,
                    coordinates=coords,
                    phases=line_phases,
                    from_vertex_id=transformer_vertex,
                    to_vertex_id=first_vid,
                )
            )

            for i in range(1, len(pts)):
                v_prev = int(pts[i - 1]["vertex_id"])
                v_curr = int(pts[i]["vertex_id"])
                d_prev = lv_distance_lookup(v_prev)
                d_curr = lv_distance_lookup(v_curr)
                seg_km = max(abs(d_curr - d_prev) / 1000.0, 0.001)
                coords = self.route_or_fallback(v_prev, v_curr)
                coords = self._offset_polyline_for_viz(
                    coords,
                    deviation_deg=5e-6,
                    sign=1.0 if ((branch_id + i) % 2 == 0) else -1.0,
                )
                component_specs.append(
                    LineSpec(
                        name=f"LV_Trunk_B{branch_id}_S{i}_{cluster_id}",
                        cable_equipment=trunk_eq,
                        bus1=f"Bus_Node_{v_prev}_{cluster_id}",
                        bus2=f"Bus_Node_{v_curr}_{cluster_id}",
                        length_km=seg_km,
                        parallel=trunk_par,
                        coordinates=coords,
                        phases=line_phases,
                        from_vertex_id=v_prev,
                        to_vertex_id=v_curr,
                    )
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
            if self.dbc:
                try:
                    coord = self.dbc.get_node_geom(
                        load_point["vertex_id"]
                    )  # transfomerposition or for mv_buildings: connection_point
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
                vertex_id=load_point["vertex_id"],
            )
            component_specs.append(bus_spec)

        # Apply MV trunk+branch with unified planner
        if not substation_vertex_id:
            self.logger.warning("Missing substation vertex for MV planning")
            return component_specs

        mv_distance_mapping = self._create_mv_distance_mapping(
            mv_load_points, substation_vertex_id
        )

        def mv_distance_lookup(vid: int) -> float:
            return float(mv_distance_mapping.get(int(vid), 0.0))

        def mv_path_to_hub(vid: int, hub: int) -> List[int]:
            if not self.dbc:
                return [vid, hub]
            try:
                return self.dbc.get_path_to_bus(int(vid), int(hub))
            except Exception:
                return [vid, hub]

        MV_BASE_V = BASE_VOLTAGE_V.get("MV", 12470)

        def mv_calc_branch_current(branch_lps: List[Dict]) -> float:
            total_kw = 0.0
            for lp in branch_lps:
                try:
                    total_kw += float(lp.get("load_kw", 0.0))
                except Exception:
                    continue
            # I = P / (sqrt(3) * V_ll * pf)
            return float((total_kw * 1000.0) /
                         (np.sqrt(3) * MV_BASE_V * POWER_FACTOR))

        def mv_select_trunk(
            I_req: float, distance_km: float
        ) -> Tuple[Optional[CableEquipment], int]:
            return self.cable_selector.find_optimal_cable(
                required_current_a=float(I_req or 0.0),
                voltage_level="MV",
                distance_km=float(distance_km or 0.0),
            )

        def mv_select_service(
                I_req: float) -> Tuple[Optional[CableEquipment], int]:
            return self.cable_selector.find_optimal_cable(
                required_current_a=float(I_req or 0.0),
                voltage_level="MV",
                distance_km=0.005,
            )

        mv_branches = self.plan_trunk_and_branches(
            level="MV",
            cluster_id=cluster_id,
            hub_vertex_id=substation_vertex_id,
            load_points=mv_load_points,
            distance_lookup=mv_distance_lookup,
            path_to_hub=mv_path_to_hub,
            calc_branch_current=mv_calc_branch_current,
            select_trunk=mv_select_trunk,
            select_service=mv_select_service,
            route=lambda a, b: self.route_or_fallback(a, b),
            logger=self.logger,
        )

        # Emit MV trunk segments and transformer service drops per planned
        # branch
        for branch_id, branch in enumerate(mv_branches):
            if not branch:
                continue

            I_req = mv_calc_branch_current(branch)
            try:
                max_d_km = (
                    max(mv_distance_lookup(
                        int(lp["vertex_id"])) for lp in branch)
                    / 1000.0
                )
            except Exception:
                max_d_km = 0.0
            trunk_eq, trunk_par = mv_select_trunk(I_req, max_d_km)
            if not trunk_eq:
                continue

            # Log parallel cable usage for MV trunk
            if trunk_par > 1:
                self.logger.info(
                    f"MV PARALLEL CABLES: Branch {branch_id} requires {trunk_par} parallel cables "
                    f"of type {
                        trunk_eq.name} (I_req={
                        I_req:.1f}A, distance={
                        max_d_km:.3f}km, "
                    f"single cable capacity={trunk_eq.max_i_a:.1f}A)"
                )
            else:
                self.logger.debug(
                    f"MV single cable: Branch {branch_id} using 1x {
                        trunk_eq.name} "
                    f"(I_req={I_req:.1f}A, capacity={trunk_eq.max_i_a:.1f}A)"
                )

            pts = sorted(
                branch, key=lambda lp: mv_distance_lookup(int(lp["vertex_id"]))
            )

            # Substation -> first node
            first_vid = int(
                pts[0]["vertex_id"]) if pts else substation_vertex_id
            d_first_km = max(mv_distance_lookup(first_vid) / 1000.0, 0.001)
            coords = self.route_or_fallback(substation_vertex_id, first_vid)
            coords = self._offset_polyline_for_viz(
                coords, deviation_deg=5e-6, sign=1.0 if (branch_id % 2 == 0) else -1.0
            )
            component_specs.append(
                LineSpec(
                    name=f"MV_Trunk_B{branch_id}_Main_{cluster_id}",
                    cable_equipment=trunk_eq,
                    bus1=mv_main_bus,
                    bus2=f"mv_node_{pts[0]['connection_point']}_{cluster_id}",
                    length_km=d_first_km,
                    parallel=trunk_par,
                    coordinates=coords,
                    phases="ABC",  # Will be updated by phase allocator
                    from_vertex_id=substation_vertex_id,
                    to_vertex_id=first_vid,
                )
            )

            # Between consecutive nodes
            for i in range(1, len(pts)):
                v_prev = int(pts[i - 1]["vertex_id"])
                v_curr = int(pts[i]["vertex_id"])
                d_prev = mv_distance_lookup(v_prev)
                d_curr = mv_distance_lookup(v_curr)
                seg_km = max(abs(d_curr - d_prev) / 1000.0, 0.001)
                coords = self.route_or_fallback(v_prev, v_curr)
                coords = self._offset_polyline_for_viz(
                    coords,
                    deviation_deg=5e-6,
                    sign=1.0 if ((branch_id + i) % 2 == 0) else -1.0,
                )
                component_specs.append(
                    LineSpec(
                        name=f"MV_Trunk_B{branch_id}_S{i}_{cluster_id}",
                        cable_equipment=trunk_eq,
                        bus1=f"mv_node_{pts[i -
                                            1]['connection_point']}_{cluster_id}",
                        bus2=f"mv_node_{
                            pts[i]['connection_point']}_{cluster_id}",
                        length_km=seg_km,
                        parallel=trunk_par,
                        coordinates=coords,
                        phases="ABC",  # Will be updated by phase allocator
                        from_vertex_id=v_prev,
                        to_vertex_id=v_curr,
                    )
                )

            # Transformer service drops (5 m) from trunk node to transformer
            # vertex
            for lp in pts:
                if lp.get("type") != "transformer":
                    continue
                trafo_id = lp.get("transformer_id", lp.get("name"))
                load_kw = float(lp.get("load_kw", 0.0))
                service_I = (load_kw * 1000.0) / \
                    (np.sqrt(3) * MV_BASE_V * POWER_FACTOR)
                svc_eq, svc_par = mv_select_service(service_I)
                if not svc_eq:
                    continue
                conn_vid = int(lp["connection_point"])  # trunk node vertex
                trafo_vid = int(lp.get("vertex_id", lp["connection_point"]))
                coords = self.route_or_fallback(conn_vid, trafo_vid)
                component_specs.append(
                    LineSpec(
                        name=f"MV_Service_T{trafo_id}_{cluster_id}",
                        cable_equipment=svc_eq,
                        bus1=f"mv_node_{lp['connection_point']}_{cluster_id}",
                        bus2=f"trafo_{trafo_id}_{cluster_id}_mv",
                        length_km=0.005,
                        parallel=1,
                        coordinates=coords,
                        phases="ABC",  # Will be updated by phase allocator
                        from_vertex_id=conn_vid,
                        to_vertex_id=trafo_vid,
                    )
                )

        # After feeders are in place, add MV building buses + loads and their
        # MV_Consumer_* service lines (connection point -> building centroid)
        if mv_buildings:
            mv_building_specs = self._install_mv_consumer_cables(
                cluster_id=cluster_id,
                mv_buildings=mv_buildings,
            )
            component_specs.extend(mv_building_specs)

        return component_specs

    def _create_mv_distance_mapping(
        self, mv_load_points: List[dict], substation_vertex_id: int
    ) -> Dict[int, float]:
        """
        Create distance mapping for MV load points (equivalent to vertex_distance_mapping for LV).

        Maps each MV load point vertex to its routing distance from the substation.
        This enables the branch-by-branch algorithm to work at MV scale.

        Args:
            mv_load_points: List of MV transformers and buildings
            substation_vertex_id: Substation location vertex

        Returns:
            Dict mapping vertex_id -> distance_from_substation (in meters)
        """
        mv_distance_mapping = {}

        if not self.dbc:
            self.logger.warning(
                "No database connection for MV routing distance calculation"
            )
            # Fallback: assign dummy distances
            for i, load_point in enumerate(mv_load_points):
                # 1km, 2km, 3km...
                mv_distance_mapping[load_point["vertex_id"]] = (i + 1) * 1000.0
            return mv_distance_mapping

        for load_point in mv_load_points:
            load_vertex = load_point["vertex_id"]

            try:
                _, length = self.dbc.get_path_to_bus_with_length(
                    load_vertex, substation_vertex_id
                )

                # Store distance in meters
                mv_distance_mapping[load_vertex] = length

            except Exception as e:
                self.logger.info(
                    f"Error calculating MV distance for vertex {load_vertex}: {e}"
                )
        # Log the created distance mapping
        self.logger.info(
            f"Created MV vertices dict: {
                len(mv_distance_mapping)} load points, "
            f"distances: {min(mv_distance_mapping.values()):.1f}-{max(mv_distance_mapping.values()):.1f}m"
        )

        return mv_distance_mapping

    def _get_consumer_simultaneous_load_dict(
        self,
        consumer_list: List[int],
        buildings_df: pd.DataFrame,
    ) -> Dict[int, float]:
        """
        Calculate simultaneous load for each consumer.

        Args:
            consumer_list: List of consumer vertex IDs
            buildings_df: Building data with peak loads and building types

        Returns:
            Dict mapping consumer_id -> simultaneous_load_kW
        """
        simultaneous_load_kw = {consumer: 0 for consumer in consumer_list}

        for row in buildings_df.itertuples():
            # Look up simultaneity factor for this building type
            simultaneity_factor = CONSUMER_CATEGORIES.loc[
                CONSUMER_CATEGORIES.definition == row.type, "sim_factor"
            ].item()

            # Calculate simultaneous load in kW using diversity factor
            simultaneous_load_kw[row.vertice_id] = utils.oneSimultaneousLoad(
                row.peak_load_in_kw, row.houses_per_building, simultaneity_factor
            )

        return simultaneous_load_kw

    def _get_node_coordinates(
            self, node_id: int) -> Optional[Tuple[float, float]]:
        """Get coordinates for a node from database."""
        if not self.dbc:
            return None

        try:
            coord = self.dbc.get_node_geom(node_id)
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
        vertex_distance_mapping: Dict,
        power_demand: Dict,
        n_phases: int = 3,
    ) -> List[ComponentSpec]:
        """
        Install cables from connection points to consumer buildings (house connections).

        This creates the final segment from the street connection point to each building.

        Args:
            cluster_id: Cluster identifier
            connection_nodes: List of connection point vertices
            consumer_list: List of consumer/building vertices
            transformer_vertex: Transformer location vertex
            vertex_distance_mapping: Mapping of vertex to distance from transformer
            power_demand: Power demand dictionary for each consumer

        Returns:
            List of LineSpec for consumer connections
        """
        line_specs = []

        # Determine phase configuration for consumer connections
        if n_phases == 1:
            consumer_phases = "A"  # Single-phase on phase A
        else:
            consumer_phases = "ABC"  # 3-phase

        # Get consumers that need to be connected through each connection point
        # This uses the database method to find which consumers connect through
        # which connection points
        if not self.dbc:
            self.logger.warning(
                "No database connection for consumer cable routing")
            return line_specs

        for connection_point in connection_nodes:
            # Get consumers that connect through this connection point
            consumer_vertices = self.dbc.get_vertices_from_connection_points(
                [connection_point]
            )

            # Filter to only consumers in our cluster
            branch_consumers = [
                v for v in consumer_vertices if v in consumer_list]

            for consumer_vertex in branch_consumers:
                # Get the path from consumer to transformer to find the
                # connection point
                path_nodes = self.dbc.get_path_to_bus(
                    consumer_vertex, transformer_vertex
                )

                if len(path_nodes) < 2:
                    continue

                # path_nodes[0] is the consumer, path_nodes[1] is the
                # connection point
                if path_nodes[1] != connection_point:
                    continue

                # Calculate distance and required current
                distance_km = self._calculate_node_distance(
                    consumer_vertex, connection_point, vertex_distance_mapping
                )

                # Get power demand and calculate current
                sim_load_mw = power_demand.get(consumer_vertex, 0)
                current_a = (sim_load_mw * 1000) / \
                    (VN * np.sqrt(3))  # Convert MW to A

                # Select appropriate cable for house connection
                cable_eq, parallel_count = self.cable_selector.find_optimal_cable(
                    required_current_a=current_a,
                    voltage_level="LV",
                    distance_km=distance_km,
                    n_phases=n_phases,
                )

                cable_equipment = cable_eq

                # Generate coordinates for the consumer connection line
                coordinates = self.route_or_fallback(
                    consumer_vertex, connection_point)

                # Create line specification from connection point to consumer
                line_spec = LineSpec(
                    name=f"Line_Consumer_{consumer_vertex}_{cluster_id}",
                    cable_equipment=cable_equipment,
                    bus1=f"Bus_Node_{connection_point}_{cluster_id}",
                    bus2=f"Bus_Consumer_{consumer_vertex}_{cluster_id}",
                    length_km=distance_km,
                    parallel=parallel_count,
                    coordinates=coordinates,
                    phases=consumer_phases,
                    from_vertex_id=connection_point,
                    to_vertex_id=consumer_vertex,
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
                c = self.dbc.get_node_geom(nid)
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
                    vertex_id=vid,
                )
            )
            line_specs.append(
                LoadSpec(
                    name=f"mv_load_{osm_id}_{cluster_id}",
                    bus=f"mv_building_{osm_id}_{cluster_id}",
                    kw=peak_kw,
                    kvar=peak_kw * 0.3,
                    kv=12.47,  # L-L for 3φ wye
                    n_phases=3,
                    conn="wye",
                    building_id=str(osm_id),
                    vertex_id=vid,
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
                node_seq, length_m = self.dbc.get_path_to_bus_with_length(
                    vid, conn_pt)
            except Exception:
                node_seq, length_m = None, None

            # Distance in km; clamp zero to 1 m to avoid exact-zero lines in
            # sims
            distance_km = max(((length_m or 0.0) / 1000.0), 0.001)

            # MV sizing at 12.47 kV,
            current_a = (peak_kw * 1000.0) / \
                (np.sqrt(3) * 12470 * POWER_FACTOR)

            cable_eq, parallel = self.cable_selector.find_optimal_cable(
                required_current_a=current_a,
                voltage_level="MV",
                distance_km=distance_km,
            )
            if not cable_eq:
                continue

            # Log parallel cable usage for MV consumer connections
            if parallel > 1:
                self.logger.info(
                    f"MV CONSUMER PARALLEL CABLES: Building {osm_id} requires {parallel} parallel "
                    f"{
                        cable_eq.name} cables (I_req={
                        current_a:.1f}A, distance={
                        distance_km:.3f}km, "
                    f"load={
                        peak_kw:.1f}kW, single cable capacity={
                        cable_eq.max_i_a:.1f}A)"
                )

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
                    phases="ABC",  # Will be updated by phase allocator
                    from_vertex_id=conn_pt,
                    to_vertex_id=vid,
                )
            )

        return line_specs

    def _calculate_node_distance(
        self, node1: int, node2: int, vertex_distance_mapping: Dict
    ) -> float:
        """Calculate distance between two nodes using routing costs from vertex_distance_mapping.

        vertex_distance_mapping contains: vertex_id -> distance_from_transformer (in meters)
        """
        try:
            # Get routing distances from transformer for both nodes
            dist1 = vertex_distance_mapping.get(node1)
            dist2 = vertex_distance_mapping.get(node2)

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
