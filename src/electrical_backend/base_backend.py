"""
Backend interface for electrical simulation backends.

This module defines the abstract base class that all electrical simulation backends
must implement. It provides a uniform interface that decouples the grid construction
algorithms from the specific electrical simulation software.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class IElectricalBackend(ABC):
    """
    Abstract interface for electrical simulation backends.

    This interface defines the contract that all electrical backends (AltDSS, pandapower, etc.)
    must implement. It ensures backend-agnostic grid construction where algorithms work
    with component specifications rather than specific electrical objects.
    """

    @abstractmethod
    def initialize_circuit(self, name: str, source_bus: str,
                           primary_kv: float) -> None:
        """
        Initialize a new electrical circuit.

        Args:
            name: Circuit name
            source_bus: Name of the source bus
            primary_kv: Primary voltage level
        """

    @abstractmethod
    def create_component(self, spec: "ComponentSpec") -> Any:
        """
        Create electrical component from specification.

        Args:
            spec: Component specification object

        Returns:
            Backend-specific component object
        """

    @abstractmethod
    def solve_power_flow(self) -> bool:
        """
        Solve power flow and return convergence status.

        Returns:
            True if power flow converged, False otherwise
        """

    @abstractmethod
    def export_to_format(self) -> Dict[str, Any]:
        """
        Export circuit to configured format.

        Returns:
            Dictionary containing the exported circuit data
        """

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources and reset backend state."""

    @abstractmethod
    def get_circuit_metrics(self) -> Dict[str, Any]:
        """
        Get key circuit metrics after solving.

        Returns:
            Dictionary with circuit performance metrics
        """
