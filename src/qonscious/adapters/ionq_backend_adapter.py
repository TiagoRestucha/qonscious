from __future__ import annotations

from datetime import datetime, timezone
from functools import cached_property
from typing import TYPE_CHECKING

from qiskit_ionq import IonQProvider

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from .backend_adapter import BackendAdapter

if TYPE_CHECKING:
    from qiskit import QuantumCircuit

    from qonscious.results.result_types import ExperimentResult


class IonQBackendAdapter(BackendAdapter):
    """
    I adapt IonQ's simulators and real computers.
    For reference read: https://docs.ionq.com/sdks/qiskit
    """

    def __init__(self, backend):
        self.backend = backend

    @classmethod
    def aria(cls, api_key) -> Self:
        """Simply provide your IonQ API key to get an adaptor on the real Aria 1"""
        provider = IonQProvider(api_key)
        return cls(provider.get_backend("qpu.aria-1"))

    @classmethod
    def simulator(cls, api_key) -> Self:
        """Simply provide your IonQ API key to get an adaptor on the simulator backend."""
        provider = IonQProvider(api_key)
        return cls(provider.get_backend("simulator"))

    @classmethod
    def aria_simulator(cls, api_key) -> Self:
        """Simply provide your IonQ API key to get an adaptor on the simulator of Aria 1
        with its noise model."""
        provider = IonQProvider(api_key)
        simulator = provider.get_backend("simulator")
        simulator.set_options(noise_model="aria-1")
        return cls(simulator)

    @cached_property
    def _backend_configuration(self):
        "QPU configuration obtained as indicated in https://quantum.cloud.ibm.com/docs/en/guides/get-qpu-information"
        "Cached after first call - maybe we should not cache it"
        "cache can be cleared with ```del obj._backend_configuration```"
        return self.backend.configuration()

    @cached_property
    def _backend_properties(self):
        "QPU dynamic information obtained as indicated in https://quantum.cloud.ibm.com/docs/en/guides/get-qpu-information"
        "Cached after first call - maybe we should not cache it"
        "cache can be cleared with ```del obj._backend_properties```"
        return self.backend.properties()

    @property
    def n_qubits(self) -> int:
        return self.backend._num_qubits

    @property
    def t1s(self) -> dict[int, float]:
      
        return {}

    @property
    def t2s(self) -> dict[int, float]:
    
        return {}

    def run(self, circuit: QuantumCircuit, **kwargs) -> ExperimentResult:
        kwargs.setdefault("shots", 1024)
        created = datetime.now(timezone.utc).isoformat()
        running = datetime.now(timezone.utc).isoformat()
        job = self.backend.run([circuit], shots=kwargs["shots"])
        finished = datetime.now(timezone.utc).isoformat()
        counts = job.get_counts()
        return {
            "counts": counts,
            "shots": kwargs["shots"],
            "backend_properties": {"name": self.backend.name},
            "timestamps": {
                "created": created,
                "running": running,
                "finished": finished,
            },
            "raw_results": job.result(),
        }
