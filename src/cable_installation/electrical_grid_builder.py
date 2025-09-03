"""
Unified Grid Builder for pylovo-usa

This module implements the new architecture for grid construction that decouples
algorithms from electrical simulation backends. It orchestrates the complete
grid construction process from MV substations down to LV loads.

Based on the architecture defined in decoupled_grid_architecture_merged.md
"""

import logging
from typing import Any, Dict, List, Optional

from ..database.database_client import DatabaseClient
from ..electrical_backend.base_backend import IElectricalBackend
from ..electrical_backend.component_specs import BusSpec, ComponentSpec, TransformerSpec
from .cable_placement import CablePlacementAlgorithm


class ElectricalGridBuilder:
    """
    Unified orchestrator for the electrical grid construction using backend-agnostic approach.

    This class coordinates the complete grid generation process:
    1. Infrastructure placement (already done by clustering algorithms)
    2. MV substation network construction
    3. MV-LV distribution transformers
    4. LV network construction using cable placement algorithms
    5. Backend-agnostic electrical simulation
    """

    def __init__(
        self,
        backend: IElectricalBackend,
        database: DatabaseClient,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize unified grid builder.

        Args:
            backend: Electrical simulation backend (e.g., AltDSSBackend)
            database: Database client for grid data access
            logger: Optional logger instance
        """
        self.backend = backend
        self.database = database
        self.logger = logger or logging.getLogger(__name__)

        # Initialize cable placement algorithm with database for voltage-aware
        # selection<
        self.cable_algorithm = CablePlacementAlgorithm(
            database=database, logger=self.logger
        )

    def build_complete_grid_for_cluster(
        self, kcid: int, scid: int, regional_identifier: int = 1
    ) -> bool:
        """
        Build complete hierarchical grid for a single (kcid, scid) cluster.

        This method is the integration point for the existing parallel processing workflow.
        It builds a complete MV-LV hierarchical grid for one substation cluster.

        Args:
            kcid: K-means cluster ID
            scid: Substation cluster ID (was bcid in old system)

        Returns:
            True if construction succeeded, False otherwise
        """
        try:
            self.logger.info(f"Starting hierarchical construction for K{kcid}_S{scid}")

            # Store regional_identifier for use in helper methods
            self.current_regional_identifier = regional_identifier

            # Build complete MV-LV hierarchy
            all_component_specs = []

            # Phase 1: Build MV substation and feeders
            self.logger.debug(f"Building MV substation network for K{kcid}_S{scid}")
            mv_specs = self._build_mv_substation_network(kcid, scid)
            all_component_specs.extend(mv_specs)

            # Create MV components and extract MV-only parameters
            for spec in mv_specs:
                self.backend.create_component(spec)

            # Solve MV network and extract MV parameters
            if self.backend.solve_power_flow():
                # lOG CIRCUIT METRICS AFTER MV SOLVE
                metrics = self.backend.get_circuit_metrics()
                self.logger.info(f"Metrics after MV grid solve: {metrics}")
            # Phase 2: Build all LV networks under this MV substation
            self.logger.debug(f"Building LV networks for K{kcid}_S{scid}")
            lv_specs = self._build_lv_networks_top_down(kcid, scid)
            all_component_specs.extend(lv_specs)

            # Create remaining LV components in the backend
            self.logger.info(f"Creating {len(lv_specs)} LV components in backend")

            for spec in lv_specs:
                self.backend.create_component(spec)

            # Stage C: Recalculate voltage bases after LV transformers are added
            # This ensures proper kVBase assignment for new 0.416kV buses
            self.logger.info(
                "Recalculating voltage bases after LV transformer placement"
            )
            self.backend.dss("CalcVoltageBases")

            if self.backend.solve_power_flow():

                metrics = self.backend.get_circuit_metrics()
                self.logger.info(f"Metrics after MV+LV grid solve: {metrics}")

                self.logger.info(
                    f"✓ Grid construction and power flow successful for K{kcid}_S{scid}"
                )

                # Export results and save to database
                grid_data = self.backend.export_to_format()
                self._save_cluster_results(regional_identifier, kcid, scid, grid_data)

                # Save line components to database for visualization
                self._save_line_components_to_database(all_component_specs, kcid, scid)

                return True
            else:
                self.logger.error(f"✗ Power flow did not converge for K{kcid}_S{scid}")
                return False

        except Exception as e:
            self.logger.error(
                f"Grid construction failed for K{kcid}_S{scid}: {
                    str(e)}",
                exc_info=True,
            )
            return False
        finally:

            self.backend.cleanup()

    def _build_mv_substation_network(self, kcid: int, scid: int) -> List[ComponentSpec]:
        """
        Build MV substation network with feeders to LV transformers.

        Args:
            kcid: K-means cluster ID
            scid: Substation cluster ID

        Returns:
        ications for MV network
        """
        component_specs = []

        # Get substation data with pre-selected equipment
        substation_data = self.database.get_substation_for_scid(kcid, scid)

        if not substation_data:
            raise ValueError(f"No substation found for K{kcid}_S{scid}")

        substation_equipment = self.database.get_equipment_by_id(
            substation_data["equipment_id"]
        )

        # Initialize circuit with proper single source
        source_bus = f"Source_K{kcid}_S{scid}"
        primary_kv = substation_equipment.primary_voltage_kv

        # Initialize AltDSS circuit with consistent bus naming
        circuit_name = f"Grid_K{kcid}_S{scid}"
        self.backend.initialize_circuit(circuit_name, source_bus, primary_kv)

        # Edit the existing Vsource (created by initialize_circuit) to set MVA levels
        # This avoids creating duplicate sources
        self.backend.dss(
            f"Edit Vsource.source basekv={primary_kv} pu=1.0 phases=3 bus1={source_bus} "
            f"MVASC3=1000 MVASC1=900"
        )

        # MV main bus (no need to "create" - it exists when transformer
        # references it)
        mv_main_bus = f"MV_Main_K{kcid}_S{scid}"

        # Create substation transformer (primary_kv -> secondary_kv)
        substation_tx_spec = TransformerSpec(
            name=f"SubTx_K{kcid}_S{scid}",
            bus1=source_bus,  # primary_kv side
            bus2=mv_main_bus,  # secondary_kv side
            equipment=substation_equipment,
            # Pre-selected during placement, contains number of phases
            kva=substation_equipment.s_max_kva if substation_equipment else None,
        )
        component_specs.append(substation_tx_spec)

        # Get LV transformers and MV buildings under this substation
        lv_transformers = self.database.get_lv_transformers_for_scid(kcid, scid)
        mv_buildings = self.database.get_mv_buildings_for_scid(kcid, scid)

        # Create complete MV network using cable placement algorithm
        self.logger.debug(
            f"Creating MV network: {
                len(lv_transformers)} LV transformers, {
                len(mv_buildings)} MV buildings"
        )

        mv_network_specs = self.cable_algorithm.create_mv_network_components(
            cluster_id=f"MV_K{kcid}_S{scid}",
            mv_main_bus=mv_main_bus,
            lv_transformers=lv_transformers,
            mv_buildings=mv_buildings,
            equipment_lookup=self.database.get_equipment_by_id,
            substation_vertex_id=substation_data.get("substation_vertice_id"),
        )

        component_specs.extend(mv_network_specs)

        self.logger.debug(
            f"Created {
                len(component_specs)} MV components for K{kcid}_S{scid}"
        )
        return component_specs

    def _build_lv_networks_top_down(self, kcid: int, scid: int) -> List[ComponentSpec]:
        """
        Build LV networks from each distribution transformer to buildings.

        This method uses the extracted cable placement algorithms to create
        LV networks under each MV-LV transformer.

        Args:
            kcid: K-means cluster ID
            scid: Substation cluster ID

        Returns:
            List of component specifications for all LV networks
        """
        component_specs = []

        # Canonical LV bus name (use everywhere for TX, lines, loads)
        def lv_bus_name(bcid_int: int) -> str:
            return f"lv_bus_b{bcid_int}"

        # Get all bcid clusters under this scid
        bcid_clusters = self.database.get_bcids_for_scid(kcid, scid)

        for bcid in bcid_clusters:
            self.logger.debug(f"Building LV network for bcid {bcid}")

            # Get LV transformer data with pre-selected equipment
            lv_tx_data = self.database.get_lv_transformer_for_bcid(kcid, scid, bcid)

            if not lv_tx_data:
                self.logger.warning(f"No LV transformer found for bcid {bcid}")
                continue

            # CRITICAL: Use pre-selected transformer equipment
            lv_equipment = self.database.get_equipment_by_id(lv_tx_data["equipment_id"])

            # Create LV bus (400V side of transformer) using canonical name
            lv_bus = lv_bus_name(bcid)
            lv_bus_spec = BusSpec(name=lv_bus, voltage_kv=0.4)
            component_specs.append(lv_bus_spec)

            # Create MV-LV distribution transformer
            mv_cluster_id = f"MV_K{kcid}_S{scid}"
            mv_bus = f"trafo_{bcid}_{mv_cluster_id}_mv"
            lv_tx_spec = TransformerSpec(
                name=f"DistTx_B{bcid}",
                bus1=mv_bus,  # 20kV side
                bus2=lv_bus,  # 400V side
                equipment=lv_equipment,  # Pre-selected during LV placement!
                kva=lv_equipment.s_max_kva if lv_equipment else None,
            )
            component_specs.append(lv_tx_spec)

            # Get network data for cable placement algorithm
            # Pass the regional_identifier from the current processing context
            network_data = self._prepare_lv_network_data(
                self.current_regional_identifier, kcid, scid, bcid
            )

            if not network_data:
                self.logger.warning(f"No network data available for bcid {bcid}")
                continue

            (
                vertices_dict,
                transformer_vertex,
                buildings_df,
                consumer_df,
                connection_nodes,
            ) = network_data

            # Apply cable placement algorithm to create LV network
            cluster_id = f"K{kcid}_S{scid}_B{bcid}"

            lv_network_specs = self.cable_algorithm.create_lv_network_components(
                cluster_id=cluster_id,
                lv_bus=lv_bus,
                vertices_dict=vertices_dict,
                transformer_vertex=transformer_vertex,
                buildings_df=buildings_df,
                consumer_df=consumer_df,
                connection_nodes=connection_nodes,
                equipment_lookup=self.database.get_equipment_by_id,
            )

            component_specs.extend(lv_network_specs)

        self.logger.debug(
            f"Created LV networks for {
                len(bcid_clusters)} bcid clusters"
        )
        return component_specs

    def _prepare_lv_network_data(
        self, regional_identifier: int, kcid: int, scid: int, bcid: int
    ) -> Optional[tuple]:
        """
        Prepare network data for LV cable placement algorithm.

        This replicates the data preparation from GridGenerator.prepare_vertices_list
        but uses the new database interface.

        Returns:
            Tuple of (vertices_dict, transformer_vertex, buildings_df, consumer_df, connection_nodes)
            or None if data is not available
        """
        try:
            # Get vertices and distance information
            vertices_dict, transformer_vertex = self.database.get_vertices_from_bcid(
                regional_identifier=regional_identifier, kcid=kcid, bcid=bcid, scid=scid
            )

            # Get building information
            buildings_df = self.database.get_buildings_from_bcid(
                regional_identifier=regional_identifier, kcid=kcid, bcid=bcid, scid=scid
            )

            # Get consumer categories
            consumer_df = self.database.get_consumer_categories()

            # Calculate connection nodes (non-building vertices)
            vertices_list = list(vertices_dict.keys())
            consumer_list = buildings_df.vertice_id.to_list()
            consumer_list = list(dict.fromkeys(consumer_list))  # Remove duplicates
            connection_nodes = [v for v in vertices_list if v not in consumer_list]

            return (
                vertices_dict,
                transformer_vertex,
                buildings_df,
                consumer_df,
                connection_nodes,
            )

        except Exception as e:
            self.logger.error(
                f"Failed to prepare LV network data for bcid {bcid}: {
                    str(e)}"
            )
            return None

    def _save_cluster_results(
        self, regional_identifier: int, kcid: int, scid: int, grid_data: Dict[str, Any]
    ) -> None:
        """
        Save grid construction results for a single cluster to database.

        Args:
            kcid: K-means cluster ID
            scid: Substation cluster ID
            grid_data: Complete grid data from backend export
        """
        try:

            self.database.save_grid_cluster(
                regional_identifier=regional_identifier,
                kcid=kcid,
                scid=scid,
                grid_data=grid_data,
            )

        except Exception as e:
            self.logger.error(f"Failed to save cluster grid results: {str(e)}")

    def _save_line_components_to_database(
        self, component_specs: List[ComponentSpec], kcid: int, scid: int
    ) -> None:
        """
        Save line components to database for visualization with MV/LV distinction.

        Args:
            component_specs: All component specifications from grid construction
            kcid: K-means cluster ID
            scid: Substation cluster ID
        """
        try:
            line_count = 0

            for spec in component_specs:
                # Only process line specifications
                if not hasattr(spec, "component_type") or spec.component_type != "line":
                    continue

                from ..electrical_backend.component_specs import LineSpec

                if not isinstance(spec, LineSpec):
                    continue

                # Skip lines without geometry
                if not spec.coordinates or len(spec.coordinates) < 2:
                    self.logger.warning(
                        f"Skipping line {spec.name} - no coordinates (bus1={spec.bus1}, bus2={spec.bus2})"
                    )
                    continue

                # Determine voltage level from bus names or equipment
                grid_level = self._determine_line_voltage_level(spec)

                # Get equipment information
                equipment_id = (
                    spec.cable_equipment.name
                    if spec.cable_equipment
                    else "unknown_cable"
                )

                # Extract bus numbers from bus names (simplified)
                # Simple hash-based bus numbering
                from_bus = hash(spec.bus1) % 10000
                to_bus = hash(spec.bus2) % 10000

                # Save to appropriate table based on voltage level
                if grid_level == "MV":
                    self.logger.debug(
                        f"Inserting MV line: {
                            spec.name}, from {from_bus} to {to_bus}, {
                            spec.length_km:.3f}km"
                    )
                    self.database.insert_mv_line(
                        geom=spec.coordinates,
                        kcid=kcid,
                        scid=scid,
                        line_name=spec.name,
                        equipment_id=equipment_id,
                        from_bus=from_bus,
                        to_bus=to_bus,
                        length_km=spec.length_km,
                    )
                elif grid_level == "LV":
                    # Extract bcid from line name or spec context
                    bcid = self._extract_bcid_from_line_name(spec.name)
                    if bcid is not None:
                        self.logger.debug(
                            f"Inserting LV line: {
                                spec.name}, from {from_bus} to {to_bus}, {
                                spec.length_km:.3f}km"
                        )
                        self.database.insert_lv_line(
                            geom=spec.coordinates,
                            kcid=kcid,
                            scid=scid,
                            bcid=bcid,
                            line_name=spec.name,
                            equipment_id=equipment_id,
                            from_bus=from_bus,
                            to_bus=to_bus,
                            length_km=spec.length_km,
                        )
                    else:
                        self.logger.warning(
                            f"Could not determine bcid for LV line {
                                spec.name}"
                        )

                line_count += 1

            self.logger.info(f"✓ Saved {line_count} line components to database")

        except Exception as e:
            self.logger.error(
                f"Failed to save line components: {
                    str(e)}",
                exc_info=True,
            )

    def _determine_line_voltage_level(self, line_spec) -> str:
        """
        Determine if a line is MV (20kV) or LV (400V) based on bus names or equipment.

        Args:
            line_spec: LineSpec component

        Returns:
            'MV' or 'LV'
        """
        # Check bus names for voltage indicators
        bus1_name = line_spec.bus1.upper()
        bus2_name = line_spec.bus2.upper()

        # MV indicators
        mv_indicators = ["MV_", "SOURCE", "69KV", "20KV", "SUB", "MAIN"]
        if any(
            indicator in bus1_name or indicator in bus2_name
            for indicator in mv_indicators
        ):
            return "MV"

        # LV indicators
        lv_indicators = ["LV_", "400V", "BUILDING", "_B"]
        if any(
            indicator in bus1_name or indicator in bus2_name
            for indicator in lv_indicators
        ):
            return "LV"

        # Check equipment voltage level if available
        if hasattr(line_spec, "cable_equipment") and line_spec.cable_equipment:
            if hasattr(line_spec.cable_equipment, "voltage_level"):
                return line_spec.cable_equipment.voltage_level

        # Default assumption - most lines are LV
        return "LV"

    def _extract_bcid_from_line_name(self, line_name: str) -> Optional[int]:
        """
        Extract bcid from line name like 'Line_K1_S0_B123_...'

        Args:
            line_name: Line identifier

        Returns:
            bcid integer or None if not found
        """
        import re

        # Look for pattern like _B123_ or _B123
        match = re.search(r"_B(\d+)", line_name)
        if match:
            return int(match.group(1))

        # If line name doesn't contain bcid, it might be an MV line
        return None
