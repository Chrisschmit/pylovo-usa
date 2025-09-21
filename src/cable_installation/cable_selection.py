"""
Enhanced Cable Selection for MV/LV Networks

This module provides voltage-level-aware cable selection that integrates with the
equipment_data table and supports the hierarchical MV-LV grid construction approach.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config_loader import BASE_VOLTAGE_V, VOLTAGE_DROP_LIMIT_PCT
from ..database.database_client import DatabaseClient
from ..equipment_schema import (CableEquipment,
                                create_equipment_from_database_row)


class CableSelector:
    """
    Cable selection that considers voltage levels, settlement types,
    and integrates with equipment_data table.
    """

    def __init__(
        self, database: DatabaseClient, logger: Optional[logging.Logger] = None
    ):
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

    # Currently default only 3 phase cables are supported
    def get_available_cables(
        self,
        voltage_level: str,
        application_area: Optional[int] = None,
        n_phases: int = 3,
    ) -> List[CableEquipment]:
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
              AND n_phases = %s
            """
            params = [voltage_level, n_phases]

            if application_area is not None:
                query += " AND (application_area IS NULL OR application_area = %s)"
                params.append(application_area)

            query += " ORDER BY max_i_a DESC, name"

            self.database.cur.execute(query, params)
            rows = self.database.cur.fetchall()

            # Convert to CableEquipment objects
            cables = []
            columns = [desc[0] for desc in self.database.cur.description]

            for row in rows:
                cable_dict = dict(zip(columns, row))
                try:

                    cable = create_equipment_from_database_row(cable_dict)
                    if isinstance(cable, CableEquipment):
                        cables.append(cable)
                except Exception as e:
                    self.logger.warning(
                        f"Failed to create cable equipment for {
                            cable_dict.get(
                                'name', 'unknown')}: {e}"
                    )

            self._cable_cache[cache_key] = cables
            self.logger.debug(
                f"Cached {
                    len(cables)} cables for {voltage_level} level"
            )

        return self._cable_cache[cache_key]

    def find_optimal_cable(
        self,
        required_current_a: float,
        voltage_level: str,
        distance_km: float = 0,
        application_area: Optional[int] = None,
        n_phases: int = 3,
    ) -> Tuple[Optional[CableEquipment], int]:
        """
        Find optimal cable considering voltage level, current capacity, and voltage drop.

        Args:
            required_current_a: Required current capacity in Amperes
            voltage_level: 'MV' or 'LV'
            distance_km: Cable length in kilometers (for voltage drop check)
            application_area: Optional settlement type (1=rural, 2=suburban, 3=urban)

        Returns:
            Tuple of (selected_cable, parallel_count) or (None, 0) if no suitable cable found
        """
        available_cables = self.get_available_cables(
            voltage_level, application_area, n_phases
        )

        if not available_cables:
            self.logger.warning(
                f"No cables available for {voltage_level} level")
            return None, 0

        base_voltage_v = float(BASE_VOLTAGE_V.get(voltage_level, 416))
        voltage_drop_limit_pct = float(
            VOLTAGE_DROP_LIMIT_PCT.get(
                voltage_level, 4.5))

        # Try increasing numbers of parallel cables until a suitable cable is
        # found
        parallel_count = 1
        while True:
            current_per_cable = required_current_a / parallel_count

            # Filter cables by current capacity
            suitable_cables = [
                cable
                for cable in available_cables
                if cable.max_i_a >= current_per_cable
            ]

            if not suitable_cables:
                parallel_count += 1
                continue

            # Apply voltage drop constraint if distance is specified
            if distance_km > 0:
                voltage_drop_suitable = []

                for cable in suitable_cables:
                    # Calculate cable impedance (Z = sqrt(R^2 + X^2))
                    impedance_ohm_per_km = np.sqrt(
                        float(cable.r_ohm_per_km) ** 2 +
                        float(cable.x_ohm_per_km) ** 2
                    )

                    # Voltage drop = I * Z * L / parallel_count
                    voltage_drop_v = (
                        required_current_a * impedance_ohm_per_km * distance_km
                    ) / parallel_count
                    voltage_drop_pct = (voltage_drop_v / base_voltage_v) * 100

                    if voltage_drop_pct <= voltage_drop_limit_pct:
                        voltage_drop_suitable.append(cable)

                if voltage_drop_suitable:
                    # Select cable with smallest current rating (most
                    # economical)
                    optimal_cable = min(
                        voltage_drop_suitable, key=lambda c: c.max_i_a)
                    return optimal_cable, parallel_count
            else:
                # No voltage drop constraint - select smallest suitable cable
                optimal_cable = min(suitable_cables, key=lambda c: c.max_i_a)

                return optimal_cable, parallel_count
