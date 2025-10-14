from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from qiskit_aer.noise import (
    NoiseModel,
    ReadoutError,
    depolarizing_error,
    pauli_error,
    thermal_relaxation_error,
)
from qiskit_aer.primitives import SamplerV2 as Sampler

from .base_sampler_adapter import BaseSamplerAdapter

if TYPE_CHECKING:
    from qonscious.results.result_types import ExperimentResult


class CustomNoisyBackendAdapter(BaseSamplerAdapter):
    """
    Custom noisy backend that does not rely on Kraus/SuperOp representations.

    Features:
    - Depolarizing errors on one- and two-qubit gates
    - Thermal relaxation (T1/T2 decoherence)
    - Readout errors
    - Pauli errors (X, Y, Z)

    Args:
        n_qubits: Number of qubits (default: 5)
        t1_times: T1 relaxation times in microseconds per qubit (default: 50 µs)
        t2_times: T2 relaxation times in microseconds per qubit (default: 70 µs)
        gate_times: Gate durations in nanoseconds (single=50ns, two=300ns)
        depol_prob_1q: Depolarizing probability for single-qubit gates (default: 0.001)
        depol_prob_2q: Depolarizing probability for two-qubit gates (default: 0.01)
        readout_error_prob: Readout error probability (default: 0.02)
        thermal_population: Excited state population (default: 0.01)
    """

    def __init__(
        self,
        n_qubits: int = 5,
        t1_times: dict[int, float] | None = None,
        t2_times: dict[int, float] | None = None,
        gate_times: dict[str, float] | None = None,
        depol_prob_1q: float = 0.001,
        depol_prob_2q: float = 0.01,
        readout_error_prob: float = 0.02,
        thermal_population: float = 0.01,

    ):
        self._n_qubits = n_qubits
        
        # T1 and T2 in microseconds
        self._t1_times = t1_times or {i: 50.0 for i in range(n_qubits)}
        self._t2_times = t2_times or {i: 70.0 for i in range(n_qubits)}
        
        # Gate durations in nanoseconds
        self._gate_times = gate_times or {
            "single": 50,
            "two": 300,
        }
        
        # Noise parameters
        self.depol_prob_1q = depol_prob_1q
        self.depol_prob_2q = depol_prob_2q
        self.readout_error_prob = readout_error_prob
        self.thermal_population = thermal_population
        
        # Build noise model
        self.noise_model = self._build_noise_model()
        
        # Noisy simulator and sampler
        self.simulator = AerSimulator(noise_model=self.noise_model)
        self.sampler = Sampler()

    def _build_noise_model(self) -> NoiseModel:
        nm = NoiseModel()
        depol_1q = depolarizing_error(self.depol_prob_1q, 1)
        depol_2q = depolarizing_error(self.depol_prob_2q, 2)
        single_qubit_gates = ["x", "sx", "rx", "ry", "rz", "h", "s", "sdg", "t", "tdg", "z", "p", "id"]
        two_qubit_gates = ["cx", "cz", "swap"]

        # Single-qubit errors: thermal + depolarizing + pauli idle
        pauli_x_idle = pauli_error([("X", 0.005), ("I", 0.995)])
        for q in range(self._n_qubits):
            t1, t2 = self._t1_times[q] * 1000, self._t2_times[q] * 1000
            thermal_1q = thermal_relaxation_error(t1, t2, self._gate_times["single"], excited_state_population=self.thermal_population)  # type: ignore[arg-type]
            err_1q = thermal_1q.compose(depol_1q)
            for g in single_qubit_gates:
                nm.add_quantum_error(err_1q, g, [q])
            nm.add_quantum_error(pauli_x_idle, "id", [q])

        # Two-qubit errors: thermal + depolarizing
        for q1 in range(self._n_qubits):
            for q2 in range(q1 + 1, self._n_qubits):
                t1a, t2a = self._t1_times[q1] * 1000, self._t2_times[q1] * 1000
                t1b, t2b = self._t1_times[q2] * 1000, self._t2_times[q2] * 1000
                th_a = thermal_relaxation_error(t1a, t2a, self._gate_times["two"], excited_state_population=self.thermal_population)  # type: ignore[arg-type]
                th_b = thermal_relaxation_error(t1b, t2b, self._gate_times["two"], excited_state_population=self.thermal_population)  # type: ignore[arg-type]
                err_2q = th_a.tensor(th_b).compose(depol_2q)
                for g in two_qubit_gates:
                    nm.add_quantum_error(err_2q, g, [q1, q2])
                    nm.add_quantum_error(err_2q, g, [q2, q1])

        # Readout errors
        p01, p10 = self.readout_error_prob * 0.8, self.readout_error_prob
        ro = ReadoutError([[1 - p01, p01], [p10, 1 - p10]])
        for q in range(self._n_qubits):
            nm.add_readout_error(ro, [q])

        # Two-qubit readout crosstalk
        p = 0.01
        ro2 = ReadoutError([[1-p, p/3, p/3, p/3], [p/3, 1-p, p/3, p/3], [p/3, p/3, 1-p, p/3], [p/3, p/3, p/3, 1-p]])
        for q in range(self._n_qubits - 1):
            nm.add_readout_error(ro2, [q, q+1])

        return nm

    def transpile(self, circuit: QuantumCircuit) -> QuantumCircuit:
        return transpile(circuit, self.simulator, optimization_level=3)

    def run(self, circuit: QuantumCircuit, **kwargs) -> ExperimentResult:
        from datetime import datetime, timezone
        
        shots = kwargs.get("shots", 1024)
        timestamps = {"created": datetime.now(timezone.utc).isoformat()}
        
        tcirc = self.transpile(circuit)
        job = self.simulator.run(tcirc, shots=shots)
        timestamps["running"] = datetime.now(timezone.utc).isoformat()
        result = job.result()
        timestamps["finished"] = datetime.now(timezone.utc).isoformat()
        
        return {
            "counts": result.get_counts(0),
            "shots": shots,
            "backend_properties": {
                "name": "CustomNoisyBackendAdapter",
                "noise_model.n_qubits": str(self._n_qubits),
                "noise_model.depol_prob_1q": str(self.depol_prob_1q),
                "noise_model.depol_prob_2q": str(self.depol_prob_2q),
                "noise_model.readout_error_prob": str(self.readout_error_prob),
                "noise_model.thermal_population": str(self.thermal_population),
                "noise_model.t1_avg_us": str(np.mean(list(self._t1_times.values()))),
                "noise_model.t2_avg_us": str(np.mean(list(self._t2_times.values()))),
            },
            "timestamps": timestamps,
            "raw_results": result,
        }

    @property
    def n_qubits(self) -> int:
        return self._n_qubits

    @property
    def t1s(self) -> dict[int, float]:
        return self._t1_times

    @property
    def t2s(self) -> dict[int, float]:
        return self._t2_times

    def get_noise_model(self) -> NoiseModel:
        return self.noise_model

    def update_noise_parameters(
        self,
        depol_prob_1q: float | None = None,
        depol_prob_2q: float | None = None,
        readout_error_prob: float | None = None,
        thermal_population: float | None = None,
    ) -> None:
        params = {
            "depol_prob_1q": depol_prob_1q,
            "depol_prob_2q": depol_prob_2q,
            "readout_error_prob": readout_error_prob,
            "thermal_population": thermal_population,
        }
        for key, value in params.items():
            if value is not None:
                setattr(self, key, value)
        
        self.noise_model = self._build_noise_model()
        self.simulator = AerSimulator(noise_model=self.noise_model)
        self.sampler = Sampler()

    def print_noise_summary(self) -> None:
        avg_t1 = np.mean(list(self._t1_times.values()))
        avg_t2 = np.mean(list(self._t2_times.values()))
        
        summary = f"""
{'='*60}
Custom Noisy Backend Adapter - Noise Summary
{'='*60}
Number of qubits: {self._n_qubits}

Depolarizing Errors:
  Single-qubit gates: {self.depol_prob_1q:.4f}
  Two-qubit gates: {self.depol_prob_2q:.4f}

Thermal Relaxation:
  Average T1: {avg_t1:.2f} µs
  Average T2: {avg_t2:.2f} µs
  Thermal population: {self.thermal_population:.4f}

Readout Errors:
  Readout error probability: {self.readout_error_prob:.4f}

Gate Times:
  Single-qubit gates: {self._gate_times['single']} ns
  Two-qubit gates: {self._gate_times['two']} ns
{'='*60}"""
        print(summary)
