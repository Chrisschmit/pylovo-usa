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
from typing import Any

from ..equipment_schema import CableEquipment, TransformerEquipment


class AltDSSComponentFactory:
    """
    Factory class for creating AltDSS components directly on an AltDSS instance.

    This class centralizes all component creation logic and creates components
    directly on the provided AltDSS instance using the pythonic interface.
    """

    def __init__(self, dss_instance: Any, logger: logging.Logger | None = None):
        """
        Initialize the AltDSS component factory with an AltDSS instance.

        Args:
            dss_instance: The AltDSS instance to create components on
            logger: Optional logger for debugging
        """
        self.dss = dss_instance
        self.logger = logger or logging.getLogger(__name__)
        # Cache for created line code objects
        self.line_codes: dict[str, Any] = {}
        # Store bus voltage bases for tracking
        self._components_created: dict[str, list[Any]] = {
            "buses": [],
            "transformers": [],
            "lines": [],
            "loads": [],
            "sources": [],
            "linecodes": [],
            "capacitors": [],
            "meters": [],
        }

    # ===== COMPONENT CREATION UTILITIES =====

    # ===== TRANSFORMER CREATION =====

    def create_transformer_from_equipment(
        self,
        name: str,
        equipment: TransformerEquipment,
        bus1: str,
        bus2: str,
        conns: list[str],
    ) -> Any:
        """
        Create a transformer using equipment data.

        Args:
            name: Transformer name
            equipment: TransformerEquipment object from database
            bus1: Primary side bus
            bus2: Secondary side bus
            conns: Connection types

        Returns:
            Created transformer object
        """

        # Create transformer using pythonic interface
        transformer = self.dss.Transformer.new(
            name,
            Phases=equipment.n_phases,
            Windings=2,
            Buses=[bus1, bus2],
            Conns=conns,
            kVs=[equipment.primary_voltage_kv, equipment.secondary_voltage_kv],
            kVAs=[equipment.s_max_kva, equipment.s_max_kva],
            pctRs=[0.5, 0.5],  # Default resistances
            XHL=equipment.reactance_pu * 100 if equipment.reactance_pu else 7.0,
        )

        self._components_created["transformers"].append(transformer)
        self.logger.debug(
            f"Created transformer {name}: {
                equipment.primary_voltage_kv}kV -> {
                equipment.secondary_voltage_kv}kV, {equipment.s_max_kva}kVA"
        )
        return transformer

    def create_mv_lv_transformer(self, name: str, equipment: TransformerEquipment, bus1: str, bus2: str) -> Any:
        """
        Create an MV-LV transformer (12.47kV -> 0.4kV).

        Args:
            name: Transformer name
            equipment: TransformerEquipment object
            bus1: MV side bus
            bus2: LV side bus

        Returns:
            Created transformer object
        """
        return self.create_transformer_from_equipment(name, equipment, bus1, bus2, conns=["delta", "wye"])

    def create_substation_transformer(self, name: str, equipment: TransformerEquipment, bus1: str, bus2: str) -> Any:
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
            name,
            equipment,
            bus1,
            bus2,
            conns=["delta", "wye"],  # Typical for substation
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
            C1=cable.capacitance_nf_per_km,
            C0=0.5 * cable.capacitance_nf_per_km,
            Units="km",
            NormAmps=cable.max_i_a,
            EmergAmps=cable.max_i_a * 1.25,
        )

        self.line_codes[code_name] = line_code
        self._components_created["linecodes"].append(line_code)
        self.logger.debug(f"Created line code: {code_name}")
        return line_code

    def create_line_from_equipment(
        self,
        name: str,
        cable: CableEquipment,
        bus1: str,
        bus2: str,
        length_km: float,
        units: str = "km",
    ) -> Any:
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
            Phases=cable.n_phases,
        )

        self._components_created["lines"].append(line)
        self.logger.debug(f"Created line {name}: {bus1} -> {bus2}, {length_km}km")
        return line

    def create_mv_line(
        self,
        name: str,
        cable: CableEquipment,
        from_bus: str,
        to_bus: str,
        length_km: float,
    ) -> Any:
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
        if cable.voltage_level not in ["MV", "MV-LV"]:
            self.logger.warning(
                f"Cable {
                    cable.name} may not be suitable for MV application"
            )

        return self.create_line_from_equipment(name, cable, from_bus, to_bus, length_km)

    def create_lv_line(
        self,
        name: str,
        cable: CableEquipment,
        from_bus: str,
        to_bus: str,
        length_km: float,
    ) -> Any:
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
        if cable.voltage_level != "LV":
            self.logger.warning(
                f"Cable {
                    cable.name} may not be suitable for LV application"
            )

        return self.create_line_from_equipment(name, cable, from_bus, to_bus, length_km)

    # ===== LOAD CREATION =====

    def create_load(
        self,
        name: str,
        bus: str,
        kw: float,
        kvar: float,
        kv: float,
        n_phases: int = 3,
        conn: str = "wye",
        model: int = 1,
        pf: float | None = None,
    ) -> Any:
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
            "Model": model,
        }

        if pf is not None:
            load_params["pf"] = pf
        else:
            load_params["kvar"] = kvar

        # Create load using pythonic interface
        load = self.dss.Load.new(name, **load_params)

        self._components_created["loads"].append(load)
        self.logger.debug(f"Created load {name}: {kw}kW at {bus}")
        return load

    def create_mv_load(self, name: str, bus: str, kw: float, pf: float = 0.9) -> Any:
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
            kvar = kw * ((1 - pf**2) ** 0.5) / pf

        return self.create_load(name, bus, kw, kvar, kv=20.0, n_phases=3, conn="delta", model=1)

    def create_lv_load(self, name: str, bus: str, kw: float, pf: float = 0.95, n_phases: int = 3) -> Any:
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
            kvar = kw * ((1 - pf**2) ** 0.5) / pf

        kv = 0.4 if n_phases == 3 else 0.23  # Line-to-line or line-to-neutral

        return self.create_load(name, bus, kw, kvar, kv=kv, n_phases=n_phases, conn="wye", model=1)

    def create_building_load(
        self,
        building_id: int,
        bus: str,
        peak_load_kw: float,
        load_type: str = "residential",
        voltage_level: str = "LV",
    ) -> Any:
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
        pf_map = {"residential": 0.95, "commercial": 0.90, "industrial": 0.85}
        pf = pf_map.get(load_type, 0.90)

        if voltage_level == "MV":
            return self.create_mv_load(name, bus, peak_load_kw, pf)
        else:
            # Determine phases based on load size
            n_phases = 3 if peak_load_kw > 10 else 1
            return self.create_lv_load(name, bus, peak_load_kw, pf, n_phases)

    # ===== ADDITIONAL COMPONENTS =====

    def create_capacitor(self, name: str, bus: str, kvar: float, kv: float, n_phases: int = 3) -> Any:
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
        capacitor = self.dss.Capacitor.new(name, Bus1=bus, Phases=n_phases, kV=kv, kvar=kvar)

        self._components_created["capacitors"].append(capacitor)
        self.logger.debug(f"Created capacitor {name}: {kvar}kvar at {bus}")
        return capacitor

    def create_energy_meter(self, name: str, element: str, terminal: int = 1) -> Any:
        """
        Create an energy meter for monitoring.

        Args:
            name: Meter name
            element: Element to monitor (e.g., "Line.MainFeeder")
            terminal: Terminal number

        Returns:
            Created meter object
        """
        meter = self.dss.EnergyMeter.new(name, Element=element, Terminal=terminal)

        self._components_created["meters"].append(meter)
        self.logger.debug(f"Created energy meter {name} monitoring {element}")
        return meter

    # ===== UTILITY METHODS =====

    def get_component_summary(self) -> dict[str, int]:
        """
        Get summary of created components.

        Returns:
            Dictionary with component counts
        """
        return {comp_type: len(components) for comp_type, components in self._components_created.items()}

    def reset(self):
        """Reset the factory state for a new circuit."""
        self._components_created = {
            "buses": [],
            "transformers": [],
            "lines": [],
            "loads": [],
            "sources": [],
            "linecodes": [],
            "capacitors": [],
            "meters": [],
        }
        self.line_codes.clear()
        self.logger.info("Factory reset for new circuit")

    # ===== SINGLE-PHASE COMPONENT CREATION =====

    def create_single_phase_line(
        self, name: str, cable: CableEquipment, bus1: str, bus2: str, length_km: float, phase: str = "A"
    ) -> Any:
        """
        Create single-phase line on specified phase.

        Based on SINGLE_PHASE_LATERAL_IMPLEMENTATION_PLAN.md section 3.2.
        Maps phase letters to AltDSS numeric notation.

        Args:
            name: Line name
            cable: Cable equipment (may be 3-phase cable used for 1-phase)
            bus1: From bus name
            bus2: To bus name
            length_km: Line length in km
            phase: Phase assignment ("A", "B", or "C")

        Returns:
            Created single-phase line object
        """

        # Create single-phase line code if needed
        line_code = self.create_single_phase_line_code(cable, phase)

        # Create line with phase-specific bus connections
        line = self.dss.Line.new(
            name,
            Bus1=bus1,
            Bus2=bus2,
            LineCode=line_code,
            Length=length_km,
            Units="km",
            Phases=1,  # Single phase
        )

        self._components_created["lines"].append(line)
        self.logger.debug(f"Created single-phase line {name}: {bus1} -> {bus2}, {length_km}km")
        return line

    def create_single_phase_line_code(self, cable: CableEquipment, phase: str) -> Any:
        """
        Create single-phase line code from three-phase cable equipment.

        Args:
            cable: Cable equipment (may be 3-phase)
            phase: Phase assignment for naming

        Returns:
            Single-phase line code object
        """
        code_name = f"LC_{cable.name}_1P_{phase}"

        if code_name in self.line_codes:
            return self.line_codes[code_name]

        # Create single-phase line code using positive sequence parameters
        line_code = self.dss.LineCode.new(
            code_name,
            NPhases=1,
            R1=cable.r_ohm_per_km,  # Positive sequence resistance
            X1=cable.x_ohm_per_km,  # Positive sequence reactance
            C1=cable.capacitance_nf_per_km,  # Positive sequence capacitance
            Units="km",
            NormAmps=cable.max_i_a,
            EmergAmps=cable.max_i_a * 1.25,
        )

        self.line_codes[code_name] = line_code
        self._components_created["linecodes"].append(line_code)
        self.logger.debug(f"Created single-phase line code: {code_name}")
        return line_code

    def create_split_phase_transformer(
        self,
        name: str,
        equipment: TransformerEquipment,
        mv_bus: str,
        lv_bus: str,
    ) -> Any:
        """
        Create US residential split-phase transformer with center-tap.

        Based on SINGLE_PHASE_LATERAL_IMPLEMENTATION_PLAN.md section 3.1.
        Creates 3-winding transformer for true US 120/240V split-phase service.

        Args:
            name: Transformer name
            equipment: Transformer equipment data
            mv_bus: MV side bus name
            lv_bus: LV side bus name

        Returns:
            Created split-phase transformer object
        """

        # Calculate MV phase-to-neutral voltage
        mv_ph_neutral_kv = equipment.primary_voltage_kv / 1.732  # L-L to L-N

        # Create 3-winding split-phase transformer
        transformer = self.dss.Transformer.new(
            name,
            Phases=1,
            Windings=3,
            # Buses: MV phase-neutral, LV hot1-neutral, LV hot2-neutral
            Buses=[
                mv_bus,  # MV phase (implicit neutral)
                # LV hot leg 1 (implicit neutral)
                f"{lv_bus}.1.0",
                # LV hot leg 2 (implicit neutral)
                f"{lv_bus}.0.2",
            ],
            Conns=["wye", "wye", "wye"],  # All wye connections
            # Primary 7.2kV, two 120V secondaries
            kVs=[mv_ph_neutral_kv, 0.12, 0.12],
            kVAs=[equipment.s_max_kva, equipment.s_max_kva, equipment.s_max_kva],
            pctRs=[0.6, 1.2, 1.2],
            XHL=2.04,  # high‑to‑low reactance in percent
            XHT=2.04,  # high‑to‑tertiary reactance in percent
            XLT=1.36,  # Winding resistances
        )

        self._components_created["transformers"].append(transformer)
        self.logger.debug(
            f"Created split-phase transformer {name}: {mv_bus} -> {lv_bus} (120/240V), {
                equipment.s_max_kva}kVA"
        )
        return transformer

    def create_single_phase_load(
        self, name: str, bus: str, kw: float, kvar: float, kv: float, conn: str = "wye"
    ) -> Any:
        """
        Create single-phase load with proper AltDSS bus notation.

        Args:
            name: Load name
            bus: Bus name (will be modified with phase suffix)
            kw: Active power in kW
            kvar: Reactive power in kvar
            kv: Voltage in kV (should be 0.120 for US residential)
            conn: Connection type

        Returns:
            Created single-phase load object
        """
        # Create single-phase load with phase-specific bus connection
        load = self.dss.Load.new(
            name,
            Bus1=bus,
            Phases=1,
            kV=kv,  # Should be 0.120 for US residential L-N voltage
            kW=kw,
            kvar=kvar,
            Conn=conn,
            Model=1,
        )

        self._components_created["loads"].append(load)

        return load
