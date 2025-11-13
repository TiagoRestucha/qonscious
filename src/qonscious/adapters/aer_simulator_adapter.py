from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as Sampler
from qiskit_ibm_runtime import QiskitRuntimeService

from .base_sampler_adapter import BaseSamplerAdapter

if TYPE_CHECKING:
    from qiskit_aer.backends.backendconfiguration import AerBackendConfiguration
    from qiskit_aer.backends.backendproperties import AerBackendProperties


class AerSimulatorAdapter(BaseSamplerAdapter):
    def __init__(self, sampler: Sampler, simulator: AerSimulator, qubits_properties: list):
        self.sampler = sampler or Sampler()
        self.simulator = simulator or AerSimulator()
        self.qubits_properties = qubits_properties

    @classmethod
    def based_on(cls, token, backend_name) -> Self:
        service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
        backend_to_simulate = service.backend(backend_name)
        return cls(
            Sampler(),
            AerSimulator.from_backend(backend_to_simulate),
            [
                backend_to_simulate.properties().qubit_property(i)
                for i in range(backend_to_simulate.configuration().n_qubits)
            ],
        )

    def transpile(self, circuit: QuantumCircuit) -> QuantumCircuit:
        return transpile(circuit, self.simulator, optimization_level=0)

    @cached_property
    def _backend_properties(self) -> AerBackendProperties | None:
        return self.simulator.properties()

    @cached_property
    def _backend_configuration(self) -> AerBackendConfiguration:
        return self.simulator.configuration()

    @property
    def n_qubits(self) -> int:
        return self._backend_configuration.n_qubits

    @property
    def t1s(self) -> dict[int, float]:
        return {i: self.qubits_properties[i]["T1"][0] for i in range(len(self.qubits_properties))}

    @property
    def t2s(self) -> dict[int, float]:
        return {i: self.qubits_properties[i]["T2"][0] for i in range(len(self.qubits_properties))}
