"""
Equipment Data Schema Definitions for pylovo-usa.

This module provides dataclass definitions for electrical equipment from the equipment_data table.
Used by both LV and MV infrastructure placement components.

Key components:
- Equipment dataclasses: TransformerEquipment, CableEquipment
- Infrastructure result classes: InfrastructureCluster
- Configuration classes: PlacementConfig
- Load aggregator strategy classes: LVLoadAggregator, MVLoadAggregator
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from src import utils


@dataclass
class PlacementConfig:
    """Configuration parameters for infrastructure placement."""
    grid_level: str  # "LV" or "MV"
    distance_caps: Dict[int, float]  # settlement_type -> max_distance_meters
    capacity_limits: Dict[int, int]  # settlement_type -> max_units
    min_distance_cap: float
    shrink_factor: float
    planning_power_factor: float

    def get_distance_cap(self, settlement_type: int) -> float:
        """Get distance cap for given settlement type."""
        return self.distance_caps.get(settlement_type, 160.0)

    def get_capacity_limit(self, settlement_type: int) -> int:
        """Get capacity limit for given settlement type."""
        return self.capacity_limits.get(settlement_type, 12)


@dataclass
class Equipment(ABC):
    """Base class for all electrical equipment from equipment_data table."""
    # Common fields for all equipment types
    name: str                           # Equipment name (PRIMARY KEY)
    type: str                  # 'Substation', 'Transformer', 'Line'
    application_area: int               # Settlement type equipment belongs to
    ovh_ung: str                       # Installation type: 'Overhead' or 'Underground'
    n_phases: int                      # Number of phases (1, 2, or 3)
    # Nominal voltage class (e.g.'HV-MV', 'MV-LV')
    voltage_level: str
    cost: float                        # Investment cost

    @classmethod
    @abstractmethod
    def from_database_row(cls, row: dict) -> 'Equipment':
        """Create equipment instance from database row."""


@dataclass
class TransformerEquipment(Equipment):
    """Transformer/Substation equipment with power transformation capabilities."""
    # Transformer-specific fields
    s_max_kva: int                     # Rated apparent power in kVA
    primary_voltage_kv: float          # Primary side voltage in kilovolts
    secondary_voltage_kv: float        # Secondary side voltage in kilovolts
    reactance_pu: float               # Per-unit reactance
    no_load_losses_kw: float          # No-load (core) losses in kilowatts
    short_circuit_res_ohm: float      # Short-circuit resistance in ohms

    @property
    def rated_power_kva(self) -> int:
        """Alias for s_max_kva for backward compatibility."""
        return self.s_max_kva

    @classmethod
    def from_database_row(cls, row: dict) -> 'TransformerEquipment':
        """Create transformer equipment from database row."""
        return cls(
            name=row['name'],
            type=row['type'],
            application_area=row['application_area'],
            ovh_ung=row['ovh_ung'],
            n_phases=int(row['n_phases']),
            voltage_level=row['voltage_level'],
            cost=float(row['cost']),
            s_max_kva=int(row['s_max_kva']),
            primary_voltage_kv=float(row['primary_voltage_kv']),
            secondary_voltage_kv=float(row['secondary_voltage_kv']),
            reactance_pu=float(
                row['reactance_pu']) if row['reactance_pu'] else 0.0,
            no_load_losses_kw=float(
                row['no_load_losses_kw']) if row['no_load_losses_kw'] else 0.0,
            short_circuit_res_ohm=float(
                row['short_circuit_res_ohm']) if row['short_circuit_res_ohm'] else 0.0
        )


@dataclass
class CableEquipment(Equipment):
    """Cable/Line equipment for electrical connections."""
    # Cable-specific fields
    r_ohm_per_km: float               # Resistance per km in ohms
    x_ohm_per_km: float               # Inductive reactance per km in ohms
    z_ohm_per_km: float               # Impedance per km in ohms
    capacitance_nf_per_km: float      # Capacitance per km in nanofarads
    max_i_a: int                      # Maximum current in amperes
    line_voltage: float               # Nominal line voltage in kilovolts

    @property
    def max_current_ka(self) -> float:
        """Maximum current in kA for compatibility."""
        return self.max_i_a / 1000.0

    @property
    def cable_impedance_ohm_per_km(self) -> float:
        """Calculate cable impedance for voltage drop calculations."""
        return (self.r_ohm_per_km**2 + self.x_ohm_per_km**2)**0.5

    @classmethod
    def from_database_row(cls, row: dict) -> 'CableEquipment':
        """Create cable equipment from database row."""
        return cls(
            name=row['name'],
            type=row['type'],
            application_area=row['application_area'],
            ovh_ung=row['ovh_ung'],
            n_phases=row['n_phases'],
            voltage_level=row['voltage_level'],
            cost=float(row['cost']),
            r_ohm_per_km=float(
                row['r_ohm_per_km']) if row['r_ohm_per_km'] else 0.0,
            x_ohm_per_km=float(
                row['x_ohm_per_km']) if row['x_ohm_per_km'] else 0.0,
            z_ohm_per_km=float(
                row['z_ohm_per_km']) if row['z_ohm_per_km'] else 0.0,
            capacitance_nf_per_km=float(
                row['capacitance_nf_per_km']) if row['capacitance_nf_per_km'] else 0.0,
            max_i_a=int(row['max_i_a']) if row['max_i_a'] else 0,
            line_voltage=float(
                row['line_voltage']) if row['line_voltage'] else 0.0
        )


@dataclass
class InfrastructureCluster:
    """Result of infrastructure placement for a single cluster."""
    cluster_id: int
    node_vertices: List[int]
    # Only transformers/substations for infrastructure placement
    equipment: TransformerEquipment
    optimal_vertex: int
    aggregate_load: float
    total_cost: float


class LoadAggregator(ABC):
    """Abstract base class for different load aggregation strategies."""

    @abstractmethod
    def calculate_aggregate_load(
        self,
        nodes: List[int],
        buildings_df: pd.DataFrame,
        consumer_df: pd.DataFrame,
        **kwargs
    ) -> float:
        """Calculate aggregate load for given nodes."""


class LVLoadAggregator(LoadAggregator):
    """Load aggregator for LV transformer sizing - aggregates building loads."""

    def calculate_aggregate_load(
        self,
        nodes: List[int],
        buildings_df: pd.DataFrame,
        consumer_df: pd.DataFrame,
        **kwargs
    ) -> float:
        """Calculate simultaneous peak load for buildings in kW."""
        return float(utils.simultaneousPeakLoad(
            buildings_df, consumer_df, nodes))


class MVLoadAggregator(LoadAggregator):
    """Load aggregator for MV substation sizing - aggregates LV transformer + MV building loads."""

    def __init__(self, database_client):
        self.dbc = database_client

    def calculate_aggregate_load(
        self,
        connection_points: List[int],
        buildings_df: pd.DataFrame,
        consumer_df: pd.DataFrame,
        **kwargs
    ) -> float:
        """
        Calculate aggregate load for MV substation.
        Nodes contain mix of LV transformer vertices and MV building vertices.
        """
        total_load_kw = 0.0

        for node in connection_points:
            # Check if this node is an LV transformer vertex
            lv_transformer_power = self._get_lv_transformer_power(
                node, kwargs.get('kcid'))
            if lv_transformer_power:
                # Add LV transformer rating (converted from kVA to kW)
                total_load_kw += lv_transformer_power * \
                    kwargs.get('power_factor', 0.9)
            else:
                # This is an MV building - get its direct load
                building_load = self._get_building_load(node, buildings_df)
                total_load_kw += building_load

        return total_load_kw

    def _get_lv_transformer_power(
            self, connection_point: int, kcid: int) -> Optional[float]:
        """Get LV transformer power rating if vertex is a transformer."""
        try:
            return self.dbc.get_lv_transformer_power_at_vertex(
                connection_point, kcid)
        except BaseException as e:
            print(e)
            return None

    def _get_building_load(self, vertex_id: int,
                           buildings_df: pd.DataFrame) -> float:
        """Get building load for MV-connected building."""
        building = buildings_df[buildings_df['vertice_id'] == vertex_id]
        if not building.empty:
            return float(building.iloc[0]['peak_load_in_kw'])
        return 0.0


# Pre-configured placement configurations
LV_CONFIG = PlacementConfig(
    grid_level="LV",
    distance_caps={1: 120.0, 2: 160.0, 3: 220.0},
    capacity_limits={1: 12, 2: 12, 3: 20},
    min_distance_cap=40.0,
    shrink_factor=0.70,
    planning_power_factor=0.90
)

MV_CONFIG = PlacementConfig(
    grid_level="MV",
    distance_caps={1: 800.0, 2: 1200.0, 3: 2000.0},
    capacity_limits={1: 10, 2: 50, 3: 2500},
    min_distance_cap=200.0,
    shrink_factor=0.75,
    planning_power_factor=0.85
)


def create_equipment_from_database_row(row: dict) -> Equipment:
    """
    Factory function to create appropriate equipment type from database row.

    Args:
        row: Dictionary with equipment data from equipment_data table

    Returns:
        TransformerEquipment or CableEquipment based on type field
    """
    equipment_type = row['type'].lower()

    if equipment_type in ['transformer', 'substation']:
        return TransformerEquipment.from_database_row(row)
    elif equipment_type in ['line', 'cable']:
        return CableEquipment.from_database_row(row)
    else:
        raise ValueError(f"Unknown equipment type: {row['type']}")
