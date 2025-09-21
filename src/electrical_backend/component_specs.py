"""
Component specification classes for backend-agnostic grid construction.

This module defines specification classes that describe electrical components
without being tied to any specific electrical simulation backend. These specs
serve as an intermediate format between grid construction algorithms and backends.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

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
    coordinates: Optional[Tuple[float, float]] = None
    n_phases: int = 3
    vertex_id: Optional[int] = None

    def __post_init__(self):
        self.component_type = "bus"


@dataclass
class TransformerSpec(ComponentSpec):
    """Transformer specification with pre-selected equipment."""

    bus1: str = ""  # Primary side bus
    bus2: str = ""  # Secondary side bus
    equipment: Optional[TransformerEquipment] = None
    kva: Optional[float] = None  # Override equipment rating if needed
    coordinates: Optional[Tuple[float, float]] = None
    primary_phases: Optional[str] = None
    secondary_phases: Optional[str] = None
    vertex_id: Optional[int] = None

    def __post_init__(self):
        self.component_type = "transformer"


@dataclass
class LineSpec(ComponentSpec):
    """Line/Cable specification."""

    bus1: str = ""  # From bus
    bus2: str = ""  # To bus
    cable_equipment: Optional[CableEquipment] = None  # From equipment_data
    length_km: float = 0.0
    parallel: int = 1  # Number of parallel cables
    coordinates: Optional[list] = None
    phases: Optional[str] = None
    from_vertex_id: Optional[int] = None
    to_vertex_id: Optional[int] = None

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
    building_id: Optional[str] = None
    coordinates: Optional[Tuple[float, float]] = None
    phase: Optional[str] = None
    vertex_id: Optional[int] = None

    def __post_init__(self):
        self.component_type = "load"
