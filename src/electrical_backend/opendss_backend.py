"""
OpenDSS backend implementation for pylovo-usa.

This module implements the IElectricalBackend interface using OpenDSS as the electrical
simulation engine. It handles OpenDSS instance lifecycle, component creation, and
provides a clean interface for grid construction algorithms.
"""

import json
import logging
import os
import re
import shutil
from datetime import datetime
from typing import Any

from .opendss_component_factory import OpenDSSComponentFactory
from .backend_interface import IElectricalBackend
from .component_specs import BusSpec, ComponentSpec, LineSpec, LoadSpec, TransformerSpec

# Import Altdss with fallback
try:
    import altdss
except ImportError:
    OpenDSS = None


class OpenDSSBackendError(Exception):
    """Exception raised by OpenDSS backend operations."""


class OpenDSSBackend(IElectricalBackend):
    """
    OpenDSS implementation of electrical backend interface.

    This backend uses OpenDSS for electrical simulation and provides US distribution
    standard settings. It manages the OpenDSS instance lifecycle and coordinates
    with the OpenDSSComponentFactory for component creation.
    """

    def __init__(self, logger: logging.Logger | None = None):
        """
        Initialize OpenDSS backend.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.dss = None
        self.component_factory = None
        self._circuit_name = None

        if altdss is None:
            raise OpenDSSBackendError("OpenDSS not available. Please install OpenDSS package.")

    def initialize_circuit(self, name: str, source_bus: str, primary_kv: float) -> None:
        """
        Initialize OpenDSS circuit with US distribution standards.

        Based on logic from OpenDSSGridBuilder._init_OpenDSS_circuit()

        Args:
            name: Circuit name
            source_bus: Name of the source bus
            primary_kv: Primary voltage level
        """
        try:
            # Clear any existing circuit state
            altdss.altdss("Clear")

            # Create new circuit with specified primary voltage level
            altdss.altdss(f"New Circuit.{name} basekv={primary_kv} pu=1.0 phases=3 bus1={source_bus}")

            # Set US distribution voltage bases with single-phase support
            voltage_bases = [
                primary_kv,  # Transmission (69 kV)
                12.47,  # MV three-phase (L-L)
                0.240,  # US residential 240V (L-L across split-phase)
            ]
            bases_str = ",".join(str(v) for v in voltage_bases)
            altdss.altdss(f"Set VoltageBases=[{bases_str}]")

            # CRITICAL: Calculate voltage bases after setting them
            # This ensures proper kVBase assignment to all buses
            altdss.altdss("CalcVoltageBases")

            # Set US standard frequency
            altdss.altdss("Set DefaultBaseFrequency=60")

            # Store OpenDSS instance - use the altdss.altdss object
            self.dss = altdss.altdss
            self._circuit_name = name

            # Initialize component factory
            self.component_factory = OpenDSSComponentFactory(self.dss, self.logger)

            # Edit the existing Vsource (created by initialize_circuit) to set MVA levels
            # This avoids creating duplicate sources
            self.dss(
                f"Edit Vsource.source basekv={primary_kv} pu=1.0 phases=3 bus1={source_bus} " f"MVASC3=1000 MVASC1=900"
            )
            self.logger.info(f"✓ Initialized OpenDSS circuit: {name}")
            self.logger.debug(f"Voltage bases: {voltage_bases} kV")

        except Exception as e:
            self.logger.error(f"Failed to initialize OpenDSS circuit: {str(e)}")
            raise OpenDSSBackendError(
                f"OpenDSS initialization failed: {
                    str(e)}"
            ) from e

    def create_component(self, spec: ComponentSpec) -> Any:
        """
        Create OpenDSS component from specification.

        Routes component creation to appropriate factory methods based on spec type.

        Args:
            spec: Component specification object

        Returns:
            OpenDSS component object
        """
        if self.component_factory is None:
            raise OpenDSSBackendError("Backend not initialized. Call initialize_circuit() first.")

        try:
            if isinstance(spec, TransformerSpec):
                # Check for single-phase split-phase transformer based on phase
                # allocation
                if spec.secondary_phases == "split_phase":
                    return self.component_factory.create_split_phase_transformer(
                        name=spec.name,
                        equipment=spec.equipment,
                        mv_bus=spec.bus1,
                        lv_bus=spec.bus2,
                    )

                elif spec.equipment.secondary_voltage_kv > 1:
                    return self.component_factory.create_substation_transformer(
                        name=spec.name,
                        equipment=spec.equipment,
                        bus1=f"{spec.bus1}.1.2.3",
                        bus2=f"{spec.bus2}.1.2.3.0",
                    )
                else:
                    # Standard MV-LV three-phase transformer
                    return self.component_factory.create_mv_lv_transformer(
                        name=spec.name,
                        equipment=spec.equipment,
                        bus1=spec.bus1,
                        bus2=spec.bus2,
                    )

            elif isinstance(spec, LineSpec):
                # Check for single-phase line based on phase allocation
                if spec.phases in ["L1", "L2", "A", "B", "C"]:
                    return self.component_factory.create_single_phase_line(
                        name=spec.name,
                        cable=spec.cable_equipment,
                        bus1=spec.bus1,
                        bus2=spec.bus2,
                        length_km=spec.length_km,
                        phase=spec.phases,
                    )
                else:
                    # Standard three-phase line
                    return self.component_factory.create_line_from_equipment(
                        name=spec.name,
                        cable=spec.cable_equipment,
                        bus1=spec.bus1,
                        bus2=spec.bus2,
                        length_km=spec.length_km,
                    )
            elif isinstance(spec, LoadSpec):
                # Check for single-phase load based on phase allocation
                if spec.n_phases == 1 and spec.phase in ["L1", "L2", "A", "B", "C"]:
                    return self.component_factory.create_single_phase_load(
                        name=spec.name,
                        bus=spec.bus,
                        kw=spec.kw,
                        kvar=spec.kvar,
                        kv=spec.kv,
                        conn=spec.conn,
                    )
                else:
                    # Standard three-phase load
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
                # No need to create a bus, OpenDSS will create it implicitly
                return spec.name
            else:
                raise OpenDSSBackendError(
                    f"Unknown component spec type: {
                        type(spec)}"
                )

        except Exception as e:
            self.logger.error(
                f"Failed to create component {
                    spec.name}: {
                    str(e)}"
            )
            raise OpenDSSBackendError(
                f"Component creation failed: {
                    str(e)}"
            ) from e

    def solve_power_flow(self) -> bool:
        """
        Solve OpenDSS power flow.

        Based on logic from OpenDSSGridBuilder._analyze_and_validate()

        Returns:
            True if power flow converged, False otherwise
        """
        if self.dss is None:
            raise OpenDSSBackendError("No OpenDSS instance available for analysis")

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

            try:
                report_filename = (
                    f"{self._circuit_name}_electrical_statistics.txt"
                    if self._circuit_name
                    else "electrical_statistics.txt"
                )
                self.generate_electrical_statistics_report(report_filename)
            except Exception as diag_err:
                self.logger.warning(f"Electrical statistics report generation failed: {str(diag_err)}")

            return converged

        except Exception as e:
            self.logger.error(f"Power flow solution failed: {str(e)}")
            return False

    # ---------------------------------------------------------------------
    # Diagnostics helpers
    # ---------------------------------------------------------------------
    def generate_electrical_statistics_report(self, output_filename: str = "electrical_statistics.txt") -> None:
        """
        Generate comprehensive electrical statistics report after power flow solve.

        Creates a detailed text report including:
        - Power flow convergence status
        - Total power, losses, and loss percentage
        - Voltage statistics (min/max/avg) and zero-voltage bus detection
        - Exported CSV data files (voltages, currents, powers, losses, loads)
        - Circuit JSON snapshot for reproducibility

        All outputs are saved to: statistics/<circuit_name>/

        Args:
            output_filename: Name for the statistics report text file
        """
        if self.dss is None:
            return

        circuit_name = self._circuit_name or "unknown"
        m = re.search(r"K\d+_S\d+", circuit_name or "")
        subfolder = m.group(0) if m else circuit_name
        stats_dir = os.path.abspath(os.path.join(os.getcwd(), "statistics", subfolder))
        os.makedirs(stats_dir, exist_ok=True)
        out_path = os.path.join(stats_dir, output_filename)
        out_dir = stats_dir
        converged = bool(self.dss.Solution.Converged)
        total_power = self.dss.TotalPower()
        total_losses = self.dss.Losses()
        try:
            raw_bus_names = self.dss.BusNames()
            bus_names = list(raw_bus_names) if raw_bus_names is not None else []
        except Exception:
            bus_names = []
        try:
            raw_bus_vmags = self.dss.BusVMagPU()
            bus_vmags = list(raw_bus_vmags) if raw_bus_vmags is not None else []
        except Exception:
            bus_vmags = []

        min_v = min(bus_vmags) if bus_vmags else None
        max_v = max(bus_vmags) if bus_vmags else None
        avg_v = (sum(bus_vmags) / len(bus_vmags)) if bus_vmags else None
        zero_voltage_buses: list[str] = []
        for name, vpu in zip(bus_names, bus_vmags, strict=False):
            try:
                if abs(float(vpu)) < 1e-8:
                    zero_voltage_buses.append(name)
            except Exception:
                continue

        # Export CSVs; then move/copy from repo root into statistics/<circuit>
        export_types = ["Voltages", "Currents", "Powers", "Losses", "Loads"]
        exported_files = []

        for export_type in export_types:
            try:
                self.dss(f"Export {export_type}")

                base_name = f"{export_type}.csv"
                root_base = os.path.join(os.getcwd(), base_name)
                stats_base = os.path.join(stats_dir, base_name)
                tagged_name = f"{circuit_name}_EXP_{export_type.upper()}.csv"
                stats_tagged = os.path.join(stats_dir, tagged_name)

                if os.path.exists(root_base):
                    # Move the root export into statistics as tagged file
                    shutil.move(root_base, stats_tagged)
                elif os.path.exists(stats_base):
                    # DSS may have written directly into stats_dir; create tagged copy
                    shutil.copy2(stats_base, stats_tagged)
                else:
                    # Nothing found; skip
                    continue

                exported_files.append(stats_tagged)
                self.logger.debug(f"✓ Exported {export_type} → {stats_tagged}")
            except Exception as e:
                self.logger.warning(f"Failed to export {export_type}: {str(e)}")
                continue

        json_snapshot_path = None
        try:
            json_str = self.dss.to_json()
            json_basename = (
                f"circuit_snapshot_{self._circuit_name}.json" if self._circuit_name else "circuit_snapshot.json"
            )
            json_snapshot_path = os.path.abspath(os.path.join(out_dir, json_basename))
            with open(json_snapshot_path, "w") as f:
                json.dump(json.loads(json_str), f, indent=2)
        except Exception:
            pass

        def _to_kw(value: Any) -> float | None:
            try:
                if hasattr(value, "real"):
                    return float(value.real)
                if isinstance(value, (list, tuple)) and len(value) > 0:
                    return float(value[0])
                return float(value)
            except Exception:
                return None

        with open(out_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("Electrical Statistics Report\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Timestamp: {datetime.utcnow().isoformat()}Z\n")
            f.write(f"Circuit: {self._circuit_name or 'unknown'}\n")
            f.write(f"Converged: {converged}\n")
            tp_kw = _to_kw(total_power)
            tl_w = _to_kw(total_losses)
            tl_kw = (tl_w / 1000.0) if tl_w is not None else None
            if tp_kw is not None:
                f.write(f"Total Power kW: {tp_kw:.3f}\n")
            if tl_kw is not None:
                f.write(f"Total Losses kW: {tl_kw:.3f}\n")

            f.write("\nVoltage stats (pu):\n")
            f.write(f"  min: {min_v if min_v is not None else 'n/a'}\n")
            f.write(f"  avg: {avg_v if avg_v is not None else 'n/a'}\n")
            f.write(f"  max: {max_v if max_v is not None else 'n/a'}\n")

            if zero_voltage_buses:
                f.write(f"\nZero-voltage buses ({len(zero_voltage_buses)}):\n")
                # cap list to avoid huge files
                preview = zero_voltage_buses[:200]
                for b in preview:
                    f.write(f"  - {b}\n")
                if len(zero_voltage_buses) > len(preview):
                    f.write(f"  ... and {len(zero_voltage_buses) - len(preview)} more\n")
            else:
                f.write("\nZero-voltage buses: none\n")

            f.write("\nExported files:\n")
            for p in exported_files:
                f.write(f"  - {p}\n")

            if json_snapshot_path:
                f.write(f"\nJSON snapshot: {json_snapshot_path}\n")

    def export_to_format(self) -> dict[str, Any]:
        """
        Export to JSON with metadata.

        Based on logic from OpenDSSGridBuilder._export_to_json()

        Returns:
            Dictionary containing complete OpenDSS circuit in JSON format
        """
        if self.dss is None:
            raise OpenDSSBackendError("No OpenDSS instance available for export")

        try:
            # Export using OpenDSS built-in JSON functionality
            json_str = self.dss.to_json()

            self.logger.info("✓ Exported circuit to JSON format")
            return json_str

        except Exception as e:
            self.logger.error(f"JSON export failed: {str(e)}")
            raise OpenDSSBackendError(f"JSON export failed: {str(e)}") from e

    def cleanup(self) -> None:
        """
        Clean up OpenDSS resources and reset state.

        Based on logic from OpenDSSGridBuilder._cleanup_OpenDSS()
        """
        if self.dss:
            try:
                self.dss("Clear")
                self.logger.debug("✓ Cleared OpenDSS circuit")
            except Exception as e:
                self.logger.warning(f"Error clearing OpenDSS circuit: {str(e)}")
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

        self.logger.debug("✓ OpenDSS cleanup completed")

    def get_circuit_metrics(self) -> dict[str, Any]:
        """
        Get key circuit metrics after solving.

        Based on logic from OpenDSSGridBuilder._get_circuit_metrics()

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
                "total_losses_kw": total_losses.real / 1000 if total_losses else 0,
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
                    f"Voltage range: {
                        min_v:.3f} - {
                        max_v:.3f} pu (avg: {
                        avg_v:.3f})"
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
