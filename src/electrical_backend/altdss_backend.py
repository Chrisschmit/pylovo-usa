"""
AltDSS backend implementation for pylovo-usa.

This module implements the IElectricalBackend interface using AltDSS as the electrical
simulation engine. It handles AltDSS instance lifecycle, component creation, and
provides a clean interface for grid construction algorithms.
"""

import logging
from typing import Any, Dict, Optional

from .altdss_component_factory import AltDSSComponentFactory
from .base_backend import IElectricalBackend
from .component_specs import BusSpec, ComponentSpec, LineSpec, LoadSpec, TransformerSpec

# Import AltDSS with fallback
try:
    import altdss
except ImportError:
    altdss = None


class AltDSSBackendError(Exception):
    """Exception raised by AltDSS backend operations."""


class AltDSSBackend(IElectricalBackend):
    """
    AltDSS implementation of electrical backend interface.

    This backend uses AltDSS for electrical simulation and provides US distribution
    standard settings. It manages the AltDSS instance lifecycle and coordinates
    with the AltDSSComponentFactory for component creation.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize AltDSS backend.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.dss = None
        self.component_factory = None
        self._circuit_name = None

        if altdss is None:
            raise AltDSSBackendError(
                "AltDSS not available. Please install altdss package."
            )

    def initialize_circuit(self, name: str, source_bus: str, primary_kv: float) -> None:
        """
        Initialize AltDSS circuit with US distribution standards.

        Based on logic from AltDSSGridBuilder._init_altdss_circuit()

        Args:
            name: Circuit name
            source_bus: Name of the source bus
            primary_kv: Primary voltage level
        """
        try:
            # Clear any existing circuit state
            altdss.altdss("Clear")

            # Create new circuit with specified primary voltage level
            altdss.altdss(
                f"New Circuit.{name} basekv={primary_kv} pu=1.0 phases=3 bus1={source_bus}"
            )

            # Set US distribution voltage bases
            # Primary, MV, LV line-to-line, LV line-to-neutral
            voltage_bases = [primary_kv, 12.47, 0.416, 0.208]
            bases_str = ",".join(str(v) for v in voltage_bases)
            altdss.altdss(f"Set VoltageBases=[{bases_str}]")

            # CRITICAL: Calculate voltage bases after setting them
            # This ensures proper kVBase assignment to all buses
            altdss.altdss("CalcVoltageBases")

            # Set US standard frequency
            altdss.altdss("Set DefaultBaseFrequency=60")

            # Store AltDSS instance - use the altdss.altdss object
            self.dss = altdss.altdss
            self._circuit_name = name

            # Initialize component factory
            self.component_factory = AltDSSComponentFactory(self.dss, self.logger)

            self.logger.info(f"✓ Initialized AltDSS circuit: {name}")
            self.logger.debug(f"Voltage bases: {voltage_bases} kV")

        except Exception as e:
            self.logger.error(f"Failed to initialize AltDSS circuit: {str(e)}")
            raise AltDSSBackendError(
                f"AltDSS initialization failed: {
                    str(e)}"
            ) from e

    def create_component(self, spec: ComponentSpec) -> Any:
        """
        Create AltDSS component from specification.

        Routes component creation to appropriate factory methods based on spec type.

        Args:
            spec: Component specification object

        Returns:
            AltDSS component object
        """
        if self.component_factory is None:
            raise AltDSSBackendError(
                "Backend not initialized. Call initialize_circuit() first."
            )

        try:
            if isinstance(spec, TransformerSpec):
                if spec.equipment.secondary_voltage_kv > 1:
                    return self.component_factory.create_substation_transformer(
                        name=spec.name,
                        equipment=spec.equipment,
                        bus1=spec.bus1,
                        bus2=spec.bus2,
                    )
                else:
                    return self.component_factory.create_mv_lv_transformer(
                        name=spec.name,
                        equipment=spec.equipment,
                        bus1=spec.bus1,
                        bus2=spec.bus2,
                    )

            elif isinstance(spec, LineSpec):
                return self.component_factory.create_line_from_equipment(
                    name=spec.name,
                    cable=spec.cable_equipment,
                    bus1=spec.bus1,
                    bus2=spec.bus2,
                    length_km=spec.length_km,
                )
            elif isinstance(spec, LoadSpec):
                return self.component_factory.create_load(
                    name=spec.name,
                    bus=spec.bus,
                    kw=spec.kw,
                    kvar=spec.kvar,
                    kv=spec.kv,
                    n_phases=spec.n_phases,
                    conn=spec.conn,
                )
            elif isinstance(spec, BusSpec):
                # No need to create a bus, AltDSS will create it implicitly
                return spec.name
            else:
                raise AltDSSBackendError(
                    f"Unknown component spec type: {
                        type(spec)}"
                )

        except Exception as e:
            self.logger.error(
                f"Failed to create component {
                    spec.name}: {
                    str(e)}"
            )
            raise AltDSSBackendError(
                f"Component creation failed: {
                    str(e)}"
            ) from e

    def solve_power_flow(self) -> bool:
        """
        Solve AltDSS power flow.

        Based on logic from AltDSSGridBuilder._analyze_and_validate()

        Returns:
            True if power flow converged, False otherwise
        """
        if self.dss is None:
            raise AltDSSBackendError("No AltDSS instance available for analysis")

        try:
            # CRITICAL: Calculate voltage bases before solving
            # This assigns proper kVBase to all buses based on connectivity
            self.logger.info("Calculating voltage bases...")
            self.dss("CalcVoltageBases")

            self.logger.debug("Solving power flow...")
            self.dss.Solution.Solve()

            converged = self.dss.Solution.Converged
            if converged:
                self.logger.info("✓ Power flow converged")
            else:
                self.logger.error("✗ Power flow did not converge")

            return converged

        except Exception as e:
            self.logger.error(f"Power flow solution failed: {str(e)}")
            return False

    def export_to_format(self) -> Dict[str, Any]:
        """
        Export to JSON with metadata.

        Based on logic from AltDSSGridBuilder._export_to_json()

        Returns:
            Dictionary containing complete AltDSS circuit in JSON format
        """
        if self.dss is None:
            raise AltDSSBackendError("No AltDSS instance available for export")

        try:
            # Export using AltDSS built-in JSON functionality
            json_str = self.dss.to_json()

            self.logger.info(f"✓ Exported circuit to JSON format")
            return json_str

        except Exception as e:
            self.logger.error(f"JSON export failed: {str(e)}")
            raise AltDSSBackendError(f"JSON export failed: {str(e)}") from e

    def cleanup(self) -> None:
        """
        Clean up AltDSS resources and reset state.

        Based on logic from AltDSSGridBuilder._cleanup_altdss()
        """
        if self.dss:
            try:
                self.dss("Clear")
                self.logger.debug("✓ Cleared AltDSS circuit")
            except Exception as e:
                self.logger.warning(f"Error clearing AltDSS circuit: {str(e)}")
            finally:
                self.dss = None

        if self.component_factory:
            try:
                self.component_factory.reset()
                self.logger.debug("✓ Reset component factory")
            except Exception as e:
                self.logger.warning(
                    f"Error resetting component factory: {
                        str(e)}"
                )
            finally:
                self.component_factory = None

        # Clear internal state
        self._circuit_name = None

        self.logger.debug("✓ AltDSS cleanup completed")

    def get_circuit_metrics(self) -> Dict[str, Any]:
        """
        Get key circuit metrics after solving.

        Based on logic from AltDSSGridBuilder._get_circuit_metrics()

        Returns:
            Dictionary with circuit performance metrics
        """
        if self.dss is None:
            return {}

        try:
            # Get total power and losses
            total_power = self.dss.TotalPower()
            total_losses = self.dss.Losses()

            metrics = {
                "converged": self.dss.Solution.Converged,
                "total_power_kw": total_power.real if total_power else 0,
                "total_losses_kw": total_losses.real if total_losses else 0,
                "num_buses": self.dss.NumBuses,
                "num_elements": self.dss.NumCircuitElements,
            }

            # Get voltage statistics
            bus_voltages = self.dss.BusVMagPU()
            if bus_voltages is not None and len(bus_voltages) > 0:
                metrics["min_voltage_pu"] = min(bus_voltages)
                metrics["max_voltage_pu"] = max(bus_voltages)
                metrics["avg_voltage_pu"] = sum(bus_voltages) / len(bus_voltages)

                # Log voltage validation info
                min_v, max_v, avg_v = (
                    metrics["min_voltage_pu"],
                    metrics["max_voltage_pu"],
                    metrics["avg_voltage_pu"],
                )
                self.logger.info(
                    f"Voltage range: {min_v:.3f} - {max_v:.3f} pu (avg: {avg_v:.3f})"
                )

                if min_v < 0.95 or max_v > 1.05:
                    self.logger.warning(
                        f"Voltage violations detected: min={
                            min_v:.3f}pu, max={
                            max_v:.3f}pu"
                    )

            return metrics

        except Exception as e:
            self.logger.warning(f"Error getting circuit metrics: {str(e)}")
            return {}
