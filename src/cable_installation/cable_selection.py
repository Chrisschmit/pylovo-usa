"""
Enhanced Cable Selection for MV/LV Networks

This module provides voltage-level-aware cable selection that integrates with the
equipment_data table and supports the hierarchical MV-LV grid construction approach.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..database.database_client import DatabaseClient
from ..equipment_schema import CableEquipment


class VoltageAwareCableSelector:
    """
    Enhanced cable selection that considers voltage levels, settlement types,
    and integrates with equipment_data table instead of static config.
    """

    def __init__(self, database: DatabaseClient,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize voltage-aware cable selector.

        Args:
            database: Database client for equipment queries
            logger: Optional logger instance
        """
        self.database = database
        self.logger = logger or logging.getLogger(__name__)

        # Cache for cable data to avoid repeated database queries
        self._cable_cache: Dict[str, List[CableEquipment]] = {}

    def get_available_cables(self, voltage_level: str,
                             application_area: Optional[int] = None) -> List[CableEquipment]:
        """
        Get available cables filtered by voltage level and optional settlement type.

        Args:
            voltage_level: 'MV' or 'LV'
            application_area: Optional settlement type filter (1=rural, 2=suburban, 3=urban)

        Returns:
            List of CableEquipment objects matching criteria
        """
        cache_key = f"{voltage_level}_{application_area}"

        if cache_key not in self._cable_cache:
            query = """
            SELECT * FROM equipment_data
            WHERE type = 'Cable'
              AND voltage_level = %s
            """
            params = [voltage_level]

            if application_area is not None:
                query += " AND (application_area IS NULL OR application_area = %s)"
                params.append(application_area)

            query += " ORDER BY max_i_a, name"

            self.database.cur.execute(query, params)
            rows = self.database.cur.fetchall()

            # Convert to CableEquipment objects
            cables = []
            columns = [desc[0] for desc in self.database.cur.description]

            for row in rows:
                cable_dict = dict(zip(columns, row))
                try:
                    from ..equipment_schema import \
                        create_equipment_from_database_row
                    cable = create_equipment_from_database_row(cable_dict)
                    if isinstance(cable, CableEquipment):
                        cables.append(cable)
                except Exception as e:
                    self.logger.warning(
                        f"Failed to create cable equipment for {
                            cable_dict.get(
                                'name', 'unknown')}: {e}")

            self._cable_cache[cache_key] = cables
            self.logger.debug(
                f"Cached {
                    len(cables)} cables for {voltage_level} level")

        return self._cable_cache[cache_key]

    def find_optimal_cable(
        self,
        required_current_a: float,
        voltage_level: str,
        distance_km: float = 0,
        application_area: Optional[int] = None,
        voltage_drop_limit_pct: float = 4.5,
        base_voltage_v: float = 400
    ) -> Tuple[Optional[CableEquipment], int]:
        """
        Find optimal cable considering voltage level, current capacity, and voltage drop.

        Args:
            required_current_a: Required current capacity in Amperes
            voltage_level: 'MV' or 'LV'
            distance_km: Cable length in kilometers (for voltage drop check)
            application_area: Optional settlement type (1=rural, 2=suburban, 3=urban)
            voltage_drop_limit_pct: Maximum allowed voltage drop percentage
            base_voltage_v: Base voltage for voltage drop calculation

        Returns:
            Tuple of (selected_cable, parallel_count) or (None, 0) if no suitable cable found
        """
        available_cables = self.get_available_cables(
            voltage_level, application_area)

        if not available_cables:
            self.logger.warning(
                f"No cables available for {voltage_level} level")
            return None, 0

        # Set appropriate base voltage for voltage level
        if voltage_level == 'MV':
            base_voltage_v = 20000  # 20kV for MV
        elif voltage_level == 'LV':
            base_voltage_v = 400   # 400V for LV

        # Try increasing numbers of parallel cables
        for parallel_count in range(1, 6):  # Max 5 parallel cables
            current_per_cable = required_current_a / parallel_count

            # Filter cables by current capacity
            suitable_cables = [
                cable for cable in available_cables
                if cable.max_i_a >= current_per_cable
            ]

            if not suitable_cables:
                continue

            # Apply voltage drop constraint if distance is specified
            if distance_km > 0:
                voltage_drop_suitable = []

                for cable in suitable_cables:
                    # Calculate cable impedance (Z = sqrt(R^2 + X^2))
                    impedance_ohm_per_km = np.sqrt(
                        float(cable.r_ohm_per_km)**2 +
                        float(cable.x_ohm_per_km)**2
                    )

                    # Voltage drop = I * Z * L / parallel_count
                    voltage_drop_v = (
                        required_current_a * impedance_ohm_per_km * distance_km) / parallel_count
                    voltage_drop_pct = (voltage_drop_v / base_voltage_v) * 100

                    if voltage_drop_pct <= voltage_drop_limit_pct:
                        voltage_drop_suitable.append(cable)

                if voltage_drop_suitable:
                    # Select cable with smallest current rating (most
                    # economical)
                    optimal_cable = min(
                        voltage_drop_suitable, key=lambda c: c.max_i_a)
                    self.logger.debug(
                        f"Selected {optimal_cable.name} for {voltage_level} "
                        f"({parallel_count} parallel, {required_current_a}A, {distance_km}km)"
                    )
                    return optimal_cable, parallel_count
            else:
                # No voltage drop constraint - select smallest suitable cable
                optimal_cable = min(suitable_cables, key=lambda c: c.max_i_a)
                self.logger.debug(
                    f"Selected {optimal_cable.name} for {voltage_level} "
                    f"({parallel_count} parallel, {required_current_a}A)"
                )
                return optimal_cable, parallel_count

        self.logger.warning(
            f"No suitable cable found for {voltage_level} level "
            f"({required_current_a}A, {distance_km}km)"
        )
        return None, 0

    def get_cable_cost(self, cable: CableEquipment,
                       length_km: float, parallel_count: int = 1) -> float:
        """
        Calculate total cable cost including parallel installation.

        Note: parallel_count is used for cost calculation but not stored in database.

        Args:
            cable: Selected cable equipment
            length_km: Cable length in kilometers
            parallel_count: Number of parallel cables (for cost calculation only)

        Returns:
            Total installation cost
        """
        if cable.cost is None:
            self.logger.warning(f"No cost data for cable {cable.name}")
            return 0.0

        return float(cable.cost) * length_km * parallel_count

    def create_pandapower_std_types(
            self, net, voltage_level: str, application_area: Optional[int] = None) -> None:
        """
        Create pandapower standard types for specified voltage level.

        Args:
            net: Pandapower network
            voltage_level: 'MV' or 'LV'
            application_area: Optional settlement type filter
        """
        available_cables = self.get_available_cables(
            voltage_level, application_area)

        for cable in available_cables:
            try:
                import pandapower as pp

                # Extract cross-section from name if possible (fallback to
                # rating)
                q_mm2 = 100  # Default value
                if '_' in cable.name and cable.name.split(
                        '_')[-1].replace('/', '').isdigit():
                    try:
                        q_mm2 = int(cable.name.split('_')[-1].replace('/', ''))
                    except BaseException:
                        q_mm2 = cable.max_i_a // 4  # Rough estimate

                pp_name = cable.name.replace('_', ' ')

                # Create standard type
                pp.create_std_type(
                    net,
                    {
                        "r_ohm_per_km": float(cable.r_ohm_per_km),
                        "x_ohm_per_km": float(cable.x_ohm_per_km),
                        "max_i_ka": cable.max_i_a / 1000.0,  # Convert A to kA
                        "c_nf_per_km": float(cable.capacitance_nf_per_km or 0),
                        "q_mm2": q_mm2
                    },
                    name=pp_name,
                    element="line"
                )

            except Exception as e:
                self.logger.error(
                    f"Failed to create std_type for {
                        cable.name}: {e}")

        self.logger.info(
            f"Created {
                len(available_cables)} {voltage_level} cable standard types")
