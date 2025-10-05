"""
Component specification classes for backend-agnostic grid construction.

This module defines specification classes that describe electrical components
without being tied to any specific electrical simulation backend. These specs
serve as an intermediate format between grid construction algorithms and backends.
"""

from dataclasses import dataclass

from ..equipment_schema import CableEquipment, TransformerEquipment


@dataclass
class ComponentSpec:
    """Base class for all component specifications."""

    name: str
    component_type: str = ""


@dataclass
class BusSpec(ComponentSpec):
    """Bus specification."""

    voltage_kv: float = 0.208  # US standard LV three-phase voltage (208Y/120V)
    coordinates: tuple[float, float] | None = None
    n_phases: int = 3
    vertex_id: int | None = None

    def __post_init__(self):
        self.component_type = "bus"


@dataclass
class TransformerSpec(ComponentSpec):
    """Transformer specification with pre-selected equipment."""

    bus1: str = ""  # Primary side bus
    bus2: str = ""  # Secondary side bus
    equipment: TransformerEquipment | None = None
    kva: float | None = None  # Override equipment rating if needed
    coordinates: tuple[float, float] | None = None
    primary_phases: str | None = None
    secondary_phases: str | None = None
    vertex_id: int | None = None

    def __post_init__(self):
        self.component_type = "transformer"


@dataclass
class LineSpec(ComponentSpec):
    """Line/Cable specification."""

    bus1: str = ""  # From bus
    bus2: str = ""  # To bus
    cable_equipment: CableEquipment | None = None  # From equipment_data
    length_km: float = 0.0
    parallel: int = 1  # Number of parallel cables
    coordinates: list | None = None
    phases: str | None = None
    from_vertex_id: int | None = None
    to_vertex_id: int | None = None

    def __post_init__(self):
        self.component_type = "line"


@dataclass
class LoadSpec(ComponentSpec):
    """Load specification."""

    bus: str = ""
    kw: float = 0.0
    kvar: float = 0.0
    kv: float = 0.208
    n_phases: int = 3
    conn: str = "wye"  # Connection type
    load_type: str = "residential"
    building_id: str | None = None
    coordinates: tuple[float, float] | None = None
    phase: str | None = None
    vertex_id: int | None = None

    def __post_init__(self):
        self.component_type = "load"
