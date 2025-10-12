"""
Unified Grid Builder for pylovo-usa

This module implements the new architecture for grid construction that decouples
algorithms from electrical simulation backends. It orchestrates the complete
grid construction process from MV substations down to LV loads.

Based on the architecture defined in decoupled_grid_architecture_merged.md
"""

import logging
from typing import Any

from ..config_loader import *
from ..database.database_client import DatabaseClient
from ..electrical_backend.backend_interface import IElectricalBackend
from ..electrical_backend.component_specs import BusSpec, ComponentSpec, TransformerSpec
from ..electrical_backend.phase_allocator import PhaseAllocator
from .cable_placement import CablePlacementAlgorithm
from .grid_statistics import calculate_and_save_statistics


class ElectricalGridBuilder:
    """
    Unified orchestrator for the electrical grid construction using backend-agnostic approach.

    This class coordinates the generation of the electrical representation of the grid process:
    """

    def __init__(
        self,
        backend: IElectricalBackend,
        dbc: DatabaseClient,
        logger: logging.Logger | None = None,
    ):
        """
        Initialize unified grid builder.

        Args:
            backend: Electrical simulation backend (e.g., OpenDssBackend)
            dbc: Database client for grid data access
            logger: Optional logger instance
        """
        self.backend = backend
        self.dbc = dbc
        self.logger = logger or logging.getLogger(__name__)

        # Initialize cable placement algorithm with dbc for voltage-aware
        # selection<
        self.cable_algorithm = CablePlacementAlgorithm(dbc=dbc, logger=self.logger)

    def build_complete_grid_for_cluster(self, kcid: int, scid: int, regional_identifier: int = 1) -> bool:
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

            self.current_regional_identifier = regional_identifier
            all_component_specs = []

            self.logger.debug(f"Building MV substation network for K{kcid}_S{scid}")
            mv_specs = self._build_mv_substation_network(kcid, scid)
            all_component_specs.extend(mv_specs)

            self.logger.debug(f"Building LV networks for K{kcid}_S{scid}")
            lv_specs = self._build_lv_networks_top_down(kcid, scid)
            all_component_specs.extend(lv_specs)
            self.logger.info("Starting phase allocation optimization...")
            phase_allocator = PhaseAllocator(
                logger=self.logger,
                max_imbalance_pct=30.0,
                raise_on_imbalance=False,
                optimize_retries=6,
                retry_threshold_pct=20.0,
            )
            all_component_specs = phase_allocator.allocate(all_component_specs)

            lv_imbalance = phase_allocator.get_phase_imbalance()
            mv_report = phase_allocator.get_mv_balance()
            self.logger.info(
                f"Phase allocation completed. LV imbalance: {lv_imbalance:.1f}% | "
                f"MV imbalance: {mv_report.get('imbalance_pct', 0.0):.1f}%"
            )

            calculate_and_save_statistics(
                component_specs=all_component_specs,
                kcid=kcid,
                scid=scid,
                output_dir="statistics",
                generate_plots=True,
                logger=self.logger,
            )
            created_components = set()
            skipped_duplicates = 0

            for spec in all_component_specs:
                spec_name = getattr(spec, "name", None)
                if spec_name and spec_name in created_components:
                    self.logger.warning(f"Skipping duplicate component: {spec_name}")
                    skipped_duplicates += 1
                    continue

                try:
                    self.backend.create_component(spec)
                    if spec_name:
                        created_components.add(spec_name)
                except Exception as e:
                    if "Duplicate" in str(e) or "redefined" in str(e):
                        self.logger.warning(f"OpenDSS duplicate detected, skipping: {spec_name}")
                        skipped_duplicates += 1
                        if spec_name:
                            created_components.add(spec_name)
                    else:
                        raise

            if skipped_duplicates > 0:
                self.logger.info(f"Skipped {skipped_duplicates} duplicate components")

            self.backend.dss("CalcVoltageBases")

            if self.backend.solve_power_flow():
                metrics = self.backend.get_circuit_metrics()
                self.logger.info(f"Metrics after grid solve: {metrics}")

                self.logger.info(f"✓ Grid construction and power flow successful for K{kcid}_S{scid}")

                grid_data = self.backend.export_to_format()
                self._save_cluster_results(regional_identifier, kcid, scid, grid_data)
                self._save_line_components_to_dbc(all_component_specs, kcid, scid)
                return True
            else:
                self.logger.error(f"✗ Power flow did not converge for K{kcid}_S{scid}")
                return False

        except Exception as e:
            self.logger.error(f"Grid construction failed for K{kcid}_S{scid}: {str(e)}", exc_info=True)
            return False
        finally:
            self.backend.cleanup()

    def _build_mv_substation_network(self, kcid: int, scid: int) -> list[ComponentSpec]:
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
        substation, substation_vertice_id = self.dbc.get_substation_for_scid(kcid, scid)

        if not substation:
            raise ValueError(f"No substation found for K{kcid}_S{scid}")

        # Equipment is now included directly in the returned data

        # Initialize circuit with proper single source
        source_bus = f"Source_K{kcid}_S{scid}"

        # Initialize OpenDss circuit with consistent bus naming
        circuit_name = f"Grid_K{kcid}_S{scid}"
        self.backend.initialize_circuit(circuit_name, source_bus, substation.primary_voltage_kv)

        # MV main bus
        mv_main_bus = f"MV_Main_K{kcid}_S{scid}"

        # Create substation transformer (primary_kv -> secondary_kv)
        substation_tx_spec = TransformerSpec(
            name=f"SubTx_K{kcid}_S{scid}",
            bus1=source_bus,  # primary_kv side
            bus2=mv_main_bus,  # secondary_kv side
            equipment=substation,
            # Pre-selected during placement, contains number of phases
            kva=substation.s_max_kva if substation else None,
            primary_phases="ABC",
            secondary_phases="ABC",
            vertex_id=substation_vertice_id,
        )
        component_specs.append(substation_tx_spec)

        # Get LV transformers and MV buildings under this substation
        lv_transformers = self.dbc.get_lv_transformers_for_scid(kcid, scid)
        mv_buildings = self.dbc.get_mv_buildings_for_scid(kcid, scid)

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
            substation_vertex_id=substation_vertice_id,
        )

        component_specs.extend(mv_network_specs)

        self.logger.debug(
            f"Created {
                len(component_specs)} MV components for K{kcid}_S{scid}"
        )
        return component_specs

    def _build_lv_networks_top_down(self, kcid: int, scid: int) -> list[ComponentSpec]:
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
        bcid_clusters = self.dbc.get_bcids_for_scid(kcid, scid)

        for bcid in bcid_clusters:
            self.logger.debug(f"Building LV network for bcid {bcid}")

            # Get LV transformer data with pre-selected equipment
            lv_tx_data = self.dbc.get_lv_transformer_for_bcid(kcid, scid, bcid)

            if not lv_tx_data:
                self.logger.warning(f"No LV transformer found for bcid {bcid}")
                continue

            # CRITICAL: Use pre-selected transformer equipment (now included
            # directly)
            lv_equipment = lv_tx_data["equipment"]

            # Get transformer vertex from lv_tx_data
            transformer_vertex = lv_tx_data.get("transformer_vertice_id")

            # Create LV bus using canonical name
            lv_bus = lv_bus_name(bcid)
            lv_bus_spec = BusSpec(
                name=lv_bus,
                voltage_kv=0.24,
                vertex_id=transformer_vertex,
            )
            component_specs.append(lv_bus_spec)

            # Create MV-LV distribution transformer
            mv_cluster_id = f"MV_K{kcid}_S{scid}"
            mv_bus = f"trafo_{bcid}_{mv_cluster_id}_mv"
            lv_tx_spec = TransformerSpec(
                name=f"DistTx_B{bcid}",
                bus1=mv_bus,  # MV side bus
                bus2=lv_bus,  # LV side bus
                equipment=lv_equipment,  # Pre-selected during LV placement!
                kva=lv_equipment.s_max_kva if lv_equipment else None,
                primary_phases="ABC",
                secondary_phases="split_phase",
                vertex_id=transformer_vertex,
            )
            component_specs.append(lv_tx_spec)

            #
            network_data = self._prepare_lv_network_data(self.current_regional_identifier, kcid, scid, bcid)

            if not network_data:
                self.logger.warning(f"No network data available for bcid {bcid}")
                continue

            # Apply cable placement algorithm to create LV network
            cluster_id = f"K{kcid}_S{scid}_B{bcid}"

            (
                vertex_distance_mapping,
                transformer_vertex,
                buildings_df,
                consumer_df,
                connection_nodes,
            ) = network_data

            lv_network_specs = self.cable_algorithm.create_lv_network_components(
                cluster_id=cluster_id,
                lv_bus=lv_bus,
                vertex_distance_mapping=vertex_distance_mapping,
                transformer_vertex=transformer_vertex,
                buildings_df=buildings_df,
                consumer_df=consumer_df,
                connection_nodes=connection_nodes,
                n_phases=lv_equipment.n_phases,
            )

            component_specs.extend(lv_network_specs)

        self.logger.debug(
            f"Created LV networks for {
                len(bcid_clusters)} bcid clusters"
        )
        return component_specs

    def _prepare_lv_network_data(self, regional_identifier: int, kcid: int, scid: int, bcid: int) -> tuple | None:
        """
        Prepare network data for LV cable placement algorithm.

        This replicates the data preparation from GridGenerator.prepare_vertices_list
        but uses the new dbc interface.

        Returns:
            Tuple of (vertex_distance_mapping, transformer_vertex, buildings_df, consumer_df, connection_nodes)
            or None if data is not available
        """
        try:
            # Get vertices and distance information
            vertex_distance_mapping, transformer_vertex = self.dbc.get_vertices_from_bcid(
                regional_identifier=regional_identifier,
                kcid=kcid,
                bcid=bcid,
                scid=scid,
            )

            # Get building information
            buildings_df = self.dbc.get_buildings_from_bcid(
                regional_identifier=regional_identifier, kcid=kcid, bcid=bcid, scid=scid
            )

            # Get consumer categories
            consumer_df = self.dbc.get_consumer_categories()

            # Calculate connection nodes (non-building vertices)
            vertices_list = list(vertex_distance_mapping.keys())
            consumer_list = buildings_df.vertice_id.to_list()
            consumer_list = list(dict.fromkeys(consumer_list))  # Remove duplicates
            connection_nodes = [v for v in vertices_list if v not in consumer_list]

            return (
                vertex_distance_mapping,
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

    def _save_cluster_results(self, regional_identifier: int, kcid: int, scid: int, grid_data: dict[str, Any]) -> None:
        """
        Save grid construction results for a single cluster to dbc.

        Args:
            kcid: K-means cluster ID
            scid: Substation cluster ID
            grid_data: Complete grid data from backend export
        """
        try:
            self.dbc.save_grid_cluster(
                regional_identifier=regional_identifier,
                kcid=kcid,
                scid=scid,
                grid_data=grid_data,
            )

        except Exception as e:
            self.logger.error(f"Failed to save cluster grid results: {str(e)}")

    def _save_line_components_to_dbc(self, component_specs: list[ComponentSpec], kcid: int, scid: int) -> None:
        """
        Save line components to dbc for visualization with MV/LV distinction.

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
                        f"Skipping line {
                            spec.name} - no coordinates (bus1={
                            spec.bus1}, bus2={
                            spec.bus2})"
                    )
                    continue

                # Determine voltage level from bus names or equipment
                grid_level = self._determine_line_voltage_level(spec)

                # Get equipment information
                equipment_id = spec.cable_equipment.name if spec.cable_equipment else "unknown_cable"

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
                    self.dbc.insert_mv_line(
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
                        self.dbc.insert_lv_line(
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

            self.logger.info(f"✓ Saved {line_count} line components to dbc")

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
        if any(indicator in bus1_name or indicator in bus2_name for indicator in mv_indicators):
            return "MV"

        # LV indicators
        lv_indicators = ["LV_", "400V", "BUILDING", "_B"]
        if any(indicator in bus1_name or indicator in bus2_name for indicator in lv_indicators):
            return "LV"

        # Check equipment voltage level if available
        if hasattr(line_spec, "cable_equipment") and line_spec.cable_equipment:
            if hasattr(line_spec.cable_equipment, "voltage_level"):
                return line_spec.cable_equipment.voltage_level

        # Default assumption - most lines are LV
        return "LV"

    def _extract_bcid_from_line_name(self, line_name: str) -> int | None:
        """
        Extract bcid from line name patterns like:
        - 'LV_Trunk_B5_Main_K1_S1_B1' -> bcid=1
        - 'Line_Consumer_123_K1_S0_B456' -> bcid=456

        Args:
            line_name: Line identifier

        Returns:
            bcid integer or None if not found
        """
        import re

        # Look for the cluster ID pattern K{kcid}_S{scid}_B{bcid} at the end
        # This ensures we get the bcid from the cluster ID, not branch numbers
        match = re.search(r"K\d+_S\d+_B(\d+)", line_name)
        if match:
            return int(match.group(1))

        # If line name doesn't contain bcid, it might be an MV line
        return None
