"""
AltDSS Component Factory for pylovo-usa.

This module provides a centralized factory class for creating all AltDSS circuit components
directly on an AltDSS instance using the pythonic interface. It focuses purely on component
creation and does not manage the AltDSS instance lifecycle.

Key responsibilities:
- Bus creation and coordinate setting
- Transformer creation from equipment_data
- Line/cable creation with line codes
- Load creation (MV and LV)
- External grid/source creation
- Line code generation from equipment data
- Component tracking and summary reporting

Note:
- AltDSS instance lifecycle (initialization, cleanup) is handled by AltDSSGridBuilder
- This factory assumes an already initialized AltDSS instance
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from ..equipment_data_schema import CableEquipment, TransformerEquipment


class AltDSSComponentFactory:
    """
    Factory class for creating AltDSS components directly on an AltDSS instance.

    This class centralizes all component creation logic and creates components
    directly on the provided AltDSS instance using the pythonic interface.
    """

    def __init__(self, dss_instance: Any,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the AltDSS component factory with an AltDSS instance.

        Args:
            dss_instance: The AltDSS instance to create components on
            logger: Optional logger for debugging
        """
        self.dss = dss_instance
        self.logger = logger or logging.getLogger(__name__)
        # Cache for created line code objects
        self.line_codes: Dict[str, Any] = {}
        self._buses_created: set = set()  # Track created buses
        # Store coordinates for later
        self._pending_coordinates: Dict[str, Tuple[float, float]] = {}
        self._components_created: Dict[str, List[Any]] = {
            'buses': [],
            'transformers': [],
            'lines': [],
            'loads': [],
            'sources': [],
            'linecodes': [],
            'capacitors': [],
            'meters': []
        }

    # ===== COMPONENT CREATION UTILITIES =====

    def calculate_voltage_bases(self) -> None:
        """Calculate voltage bases after all transformers are defined."""
        self.dss("CalcVoltageBases")
        self.logger.debug("Calculated voltage bases")

    def _try_set_coordinates(self, bus_name: str) -> bool:
        """Try to set coordinates for a bus if it exists and has pending coordinates.

        Returns:
            True if coordinates were set, False otherwise
        """
        if bus_name not in self._pending_coordinates:
            return False

        try:
            x, y = self._pending_coordinates[bus_name]
            bus = self.dss.Bus[bus_name]
            if bus.Name:  # Bus exists
                bus.X = x
                bus.Y = y
                self.logger.debug(
                    f"Set coordinates for bus {bus_name}: ({x}, {y})")
                # Remove from pending since it's now set
                del self._pending_coordinates[bus_name]
                return True
        except Exception:
            # Bus doesn't exist yet, keep coordinates pending
            pass
        return False

    def apply_pending_coordinates(self) -> Dict[str, bool]:
        """
        Apply all pending coordinates after the circuit has been solved.
        This should be called after solve() to ensure all buses exist.

        Returns:
            Dictionary mapping bus names to whether coordinates were successfully set
        """
        results = {}
        # Make a copy of keys since we'll be modifying the dict
        pending_buses = list(self._pending_coordinates.keys())

        for bus_name in pending_buses:
            results[bus_name] = self._try_set_coordinates(bus_name)

        remaining = len(self._pending_coordinates)
        if remaining > 0:
            self.logger.warning(
                f"{remaining} buses still have pending coordinates after solve")
        else:
            self.logger.info("All pending coordinates have been applied")

        return results

    # ===== BUS CREATION AND COORDINATES =====

    def set_bus_coordinates(self, bus_name: str, x: float, y: float) -> None:
        """
        Set coordinates for a bus (for visualization).
        Note: Coordinates are stored and applied after buses are created by components.

        Args:
            bus_name: Name of the bus
            x: X coordinate
            y: Y coordinate
        """
        # Store coordinates for later application
        self._pending_coordinates[bus_name] = (x, y)

        if bus_name not in self._buses_created:
            self._buses_created.add(bus_name)
            self._components_created['buses'].append(bus_name)

    def create_bus(self, name: str,
                   coordinates: Optional[Tuple[float, float]] = None) -> str:
        """
        Create/register a bus and optionally set its coordinates.

        Note: In AltDSS, buses are implicitly created when referenced by components.
        This method primarily tracks bus creation and sets coordinates if provided.

        Args:
            name: Bus name
            coordinates: Optional (x, y) coordinates

        Returns:
            Bus name for chaining
        """
        if name not in self._buses_created:
            self._buses_created.add(name)
            self._components_created['buses'].append(name)
            self.logger.debug(f"Registered bus: {name}")

        if coordinates:
            self.set_bus_coordinates(name, coordinates[0], coordinates[1])

        return name

    def create_mv_bus(
            self, name: str, coordinates: Optional[Tuple[float, float]] = None) -> str:
        """
        Create a medium voltage (20kV) bus.

        Args:
            name: Bus name
            coordinates: Optional (x, y) coordinates

        Returns:
            Bus name
        """
        return self.create_bus(name, coordinates)

    def create_lv_bus(
            self, name: str, coordinates: Optional[Tuple[float, float]] = None) -> str:
        """
        Create a low voltage (0.4kV) bus.

        Args:
            name: Bus name
            coordinates: Optional (x, y) coordinates

        Returns:
            Bus name
        """
        return self.create_bus(name, coordinates)

    # ===== EXTERNAL GRID/SOURCE =====

    def create_external_grid(self, bus: str, voltage_pu: float = 1.0,
                             mva_sc3: float = 1000.0, mva_sc1: float = 1000.0,
                             name: str = "Source") -> Any:
        """
        Create or modify an external grid connection (voltage source).

        Args:
            bus: Bus name to connect to
            voltage_pu: Voltage in per unit
            mva_sc3: 3-phase short circuit MVA
            mva_sc1: 1-phase short circuit MVA
            name: Source name

        Returns:
            Created/modified source object
        """
        # Always modify the default source (simplest approach)
        self.dss(
            f"Edit Vsource.Source bus1={bus} pu={voltage_pu} MVAsc3={mva_sc3} MVAsc1={mva_sc1}")
        source = self.dss.Vsource["Source"]

        self._components_created['sources'].append(source)
        self.logger.info(f"Created external grid '{name}' at {bus}")
        return source

    # ===== TRANSFORMER CREATION =====

    def create_transformer_from_equipment(self, name: str, equipment: TransformerEquipment,
                                          bus1: str, bus2: str,
                                          kva: Optional[float] = None,
                                          conns: Optional[List[str]] = None) -> Any:
        """
        Create a transformer using equipment data.

        Args:
            name: Transformer name
            equipment: TransformerEquipment object from database
            bus1: Primary side bus
            bus2: Secondary side bus
            kva: Optional kVA rating (uses equipment rating if None)
            conns: Optional connection types (default ["delta", "wye"])

        Returns:
            Created transformer object
        """
        kva_rating = kva or equipment.s_max_kva

        # Default connections if not specified
        if conns is None:
            conns = ["delta", "wye"]  # Typical US distribution

        # Create transformer using pythonic interface
        transformer = self.dss.Transformer.new(
            name,
            Phases=equipment.n_phases,
            Windings=2,
            Buses=[bus1, bus2],
            Conns=conns,
            kVs=[equipment.primary_voltage_kv, equipment.secondary_voltage_kv],
            kVAs=[kva_rating, kva_rating],
            pctRs=[0.5, 0.5],  # Default resistances
            XHL=equipment.reactance_pu * 100 if equipment.reactance_pu else 7.0
            # Skip no-load losses for now - can be set later if needed
        )

        self._components_created['transformers'].append(transformer)
        self.logger.info(
            f"Created transformer {name}: {
                equipment.primary_voltage_kv}kV -> {
                equipment.secondary_voltage_kv}kV, {kva_rating}kVA")
        return transformer

    def create_mv_lv_transformer(self, name: str, equipment: TransformerEquipment,
                                 mv_bus: str, lv_bus: str) -> Any:
        """
        Create an MV-LV transformer (20kV -> 0.4kV).

        Args:
            name: Transformer name
            equipment: TransformerEquipment object
            mv_bus: MV side bus
            lv_bus: LV side bus

        Returns:
            Created transformer object
        """
        return self.create_transformer_from_equipment(
            name, equipment, mv_bus, lv_bus,
            conns=["wye", "wye"]  # Typical for MV-LV
        )

    def create_substation_transformer(self, name: str, equipment: TransformerEquipment,
                                      hv_bus: str, mv_bus: str) -> Any:
        """
        Create a substation transformer (69kV -> 20kV).

        Args:
            name: Transformer name
            equipment: TransformerEquipment object
            hv_bus: HV side bus
            mv_bus: MV side bus

        Returns:
            Created transformer object
        """
        return self.create_transformer_from_equipment(
            name, equipment, hv_bus, mv_bus,
            conns=["delta", "wye"]  # Typical for substation
        )

    # ===== LINE/CABLE CREATION =====

    def create_line_code(self, cable: CableEquipment) -> Any:
        """
        Create an AltDSS line code from cable equipment data.

        Args:
            cable: CableEquipment object from database

        Returns:
            Created line code object (or existing if already created)
        """
        code_name = f"LC_{cable.name}"

        if code_name in self.line_codes:
            return self.line_codes[code_name]

        # Create line code using pythonic interface
        line_code = self.dss.LineCode.new(
            code_name,
            NPhases=cable.n_phases,
            R1=cable.r_ohm_per_km,
            X1=cable.x_ohm_per_km,
            R0=cable.r_ohm_per_km * 3,  # Zero sequence approximation
            X0=cable.x_ohm_per_km * 3,
            C1=cable.capacitance_nf_per_km /
            1000 if cable.capacitance_nf_per_km else 10,  # Convert to uF
            C0=cable.capacitance_nf_per_km / 1000 *
            0.5 if cable.capacitance_nf_per_km else 5,
            Units="km",
            NormAmps=cable.max_i_a,
            EmergAmps=cable.max_i_a * 1.25
        )

        self.line_codes[code_name] = line_code
        self._components_created['linecodes'].append(line_code)
        self.logger.debug(f"Created line code: {code_name}")
        return line_code

    def create_line_from_equipment(self, name: str, cable: CableEquipment,
                                   bus1: str, bus2: str, length_km: float,
                                   units: str = "km") -> Any:
        """
        Create a line/cable using equipment data.

        Args:
            name: Line name
            cable: CableEquipment object from database
            bus1: From bus
            bus2: To bus
            length_km: Line length in kilometers
            units: Length units (default "km")

        Returns:
            Created line object
        """
        # First ensure line code exists
        line_code = self.create_line_code(cable)

        # Create line using pythonic interface
        line = self.dss.Line.new(
            name,
            Bus1=bus1,
            Bus2=bus2,
            LineCode=line_code,
            Length=length_km,
            Units=units,
            Phases=cable.n_phases
        )

        self._components_created['lines'].append(line)
        self.logger.debug(
            f"Created line {name}: {bus1} -> {bus2}, {length_km}km")
        return line

    def create_mv_line(self, name: str, cable: CableEquipment,
                       from_bus: str, to_bus: str, length_km: float) -> Any:
        """
        Create an MV distribution line.

        Args:
            name: Line name
            cable: CableEquipment object (should be MV-rated)
            from_bus: From bus
            to_bus: To bus
            length_km: Line length

        Returns:
            Created line object
        """
        if cable.voltage_level not in ['MV', 'MV-LV']:
            self.logger.warning(
                f"Cable {
                    cable.name} may not be suitable for MV application")

        return self.create_line_from_equipment(
            name, cable, from_bus, to_bus, length_km)

    def create_lv_line(self, name: str, cable: CableEquipment,
                       from_bus: str, to_bus: str, length_km: float) -> Any:
        """
        Create an LV distribution line.

        Args:
            name: Line name
            cable: CableEquipment object (should be LV-rated)
            from_bus: From bus
            to_bus: To bus
            length_km: Line length

        Returns:
            Created line object
        """
        if cable.voltage_level != 'LV':
            self.logger.warning(
                f"Cable {
                    cable.name} may not be suitable for LV application")

        return self.create_line_from_equipment(
            name, cable, from_bus, to_bus, length_km)

    # ===== LOAD CREATION =====

    def create_load(self, name: str, bus: str, kw: float, kvar: float,
                    kv: float, n_phases: int = 3, conn: str = "wye",
                    model: int = 1, pf: Optional[float] = None) -> Any:
        """
        Create an AltDSS load.

        Args:
            name: Load name
            bus: Bus to connect to
            kw: Active power in kW
            kvar: Reactive power in kvar (ignored if pf is specified)
            kv: Voltage in kV
            n_phases: Number of phases
            conn: Connection type ("wye" or "delta")
            model: Load model (1=constant PQ, 2=constant Z, 3=constant P)
            pf: Optional power factor (overrides kvar)

        Returns:
            Created load object
        """
        # Build load parameters
        load_params = {
            "Bus1": bus,
            "Phases": n_phases,
            "kV": kv,
            "kW": kw,
            "Conn": conn,
            "Model": model
        }

        if pf is not None:
            load_params["pf"] = pf
        else:
            load_params["kvar"] = kvar

        # Create load using pythonic interface
        load = self.dss.Load.new(name, **load_params)

        self._components_created['loads'].append(load)
        self.logger.debug(f"Created load {name}: {kw}kW at {bus}")
        return load

    def create_mv_load(self, name: str, bus: str, kw: float,
                       pf: float = 0.9) -> Any:
        """
        Create an MV-connected load (for buildings >100kW).

        Args:
            name: Load name
            bus: MV bus to connect to
            kw: Active power in kW
            pf: Power factor (default 0.9)

        Returns:
            Created load object
        """
        if not (0 < pf <= 1):
            raise ValueError(f"Power factor must be between 0 and 1, got {pf}")

        if abs(pf - 1.0) < 1e-6:  # Essentially unity power factor
            kvar = 0
        else:
            kvar = kw * ((1 - pf**2)**0.5) / pf

        return self.create_load(name, bus, kw, kvar, kv=20.0, n_phases=3,
                                conn="delta", model=1)

    def create_lv_load(self, name: str, bus: str, kw: float,
                       pf: float = 0.95, n_phases: int = 3) -> Any:
        """
        Create an LV-connected load.

        Args:
            name: Load name
            bus: LV bus to connect to
            kw: Active power in kW
            pf: Power factor (default 0.95)
            n_phases: Number of phases (1 or 3)

        Returns:
            Created load object
        """
        if not (0 < pf <= 1):
            raise ValueError(f"Power factor must be between 0 and 1, got {pf}")

        if abs(pf - 1.0) < 1e-6:  # Essentially unity power factor
            kvar = 0
        else:
            kvar = kw * ((1 - pf**2)**0.5) / pf

        kv = 0.4 if n_phases == 3 else 0.23  # Line-to-line or line-to-neutral

        return self.create_load(name, bus, kw, kvar, kv=kv, n_phases=n_phases,
                                conn="wye", model=1)

    def create_building_load(self, building_id: int, bus: str,
                             peak_load_kw: float, load_type: str = "residential",
                             voltage_level: str = "LV") -> Any:
        """
        Create a load for a building based on its characteristics.

        Args:
            building_id: Building identifier
            bus: Bus to connect to
            peak_load_kw: Peak load in kW
            load_type: Type of load (residential, commercial, industrial)
            voltage_level: "LV" or "MV"

        Returns:
            Created load object
        """
        name = f"Load_B{building_id}"

        # Set power factor based on load type
        pf_map = {
            'residential': 0.95,
            'commercial': 0.90,
            'industrial': 0.85
        }
        pf = pf_map.get(load_type, 0.90)

        if voltage_level == "MV":
            return self.create_mv_load(name, bus, peak_load_kw, pf)
        else:
            # Determine phases based on load size
            n_phases = 3 if peak_load_kw > 10 else 1
            return self.create_lv_load(name, bus, peak_load_kw, pf, n_phases)

    # ===== ADDITIONAL COMPONENTS =====

    def create_capacitor(self, name: str, bus: str, kvar: float,
                         kv: float, n_phases: int = 3) -> Any:
        """
        Create a capacitor bank for power factor correction.

        Args:
            name: Capacitor name
            bus: Bus to connect to
            kvar: Reactive power in kvar
            kv: Voltage in kV
            n_phases: Number of phases

        Returns:
            Created capacitor object
        """
        capacitor = self.dss.Capacitor.new(
            name,
            Bus1=bus,
            Phases=n_phases,
            kV=kv,
            kvar=kvar
        )

        self._components_created['capacitors'].append(capacitor)
        self.logger.debug(f"Created capacitor {name}: {kvar}kvar at {bus}")
        return capacitor

    def create_energy_meter(self, name: str, element: str,
                            terminal: int = 1) -> Any:
        """
        Create an energy meter for monitoring.

        Args:
            name: Meter name
            element: Element to monitor (e.g., "Line.MainFeeder")
            terminal: Terminal number

        Returns:
            Created meter object
        """
        meter = self.dss.EnergyMeter.new(
            name,
            Element=element,
            Terminal=terminal
        )

        self._components_created['meters'].append(meter)
        self.logger.debug(f"Created energy meter {name} monitoring {element}")
        return meter

    # ===== UTILITY METHODS =====

    def get_component_summary(self) -> Dict[str, int]:
        """
        Get summary of created components.

        Returns:
            Dictionary with component counts
        """
        return {
            comp_type: len(components)
            for comp_type, components in self._components_created.items()
        }

    def reset(self):
        """Reset the factory state for a new circuit."""
        self._buses_created.clear()
        self._pending_coordinates.clear()
        self._components_created = {
            'buses': [],
            'transformers': [],
            'lines': [],
            'loads': [],
            'sources': [],
            'linecodes': [],
            'capacitors': [],
            'meters': []
        }
        self.line_codes.clear()
        self.logger.info("Factory reset for new circuit")
