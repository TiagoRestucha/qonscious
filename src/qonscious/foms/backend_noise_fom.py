from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING
import math
import numpy as np

from qonscious.foms.figure_of_merit import FigureOfMerit

if TYPE_CHECKING:
    from qonscious.adapters.backend_adapter import BackendAdapter
    from qonscious.results.result_types import FigureOfMeritResult


class BackendNoiseFoM(FigureOfMerit):


    def evaluate(self, backend_adapter: BackendAdapter, **kwargs) -> FigureOfMeritResult:
        # --- Get noise parameters with safe fallbacks
        dep1q = float(getattr(backend_adapter, "depol_prob_1q", 0.0))
        dep2q = float(getattr(backend_adapter, "depol_prob_2q", 0.0))
        readout_p = float(getattr(backend_adapter, "readout_error_prob", 0.0))
        thermal_pop = float(getattr(backend_adapter, "thermal_population", 0.0))

        # T1/T2 maps and gate durations (convert to seconds when needed)
        t1_map = getattr(backend_adapter, "t1s", None) or getattr(backend_adapter, "_t1_times", None) or {}
        t2_map = getattr(backend_adapter, "t2s", None) or getattr(backend_adapter, "_t2_times", None) or {}
        gate_times = getattr(backend_adapter, "_gate_times", {"single": 50.0, "two": 300.0})

        def _values_in_seconds(values, default_seconds):
            if not values:
                return [default_seconds]
            as_list = [float(v) for v in values]
            mean_value = float(np.mean(as_list))
            if mean_value > 1.0:  # assume microseconds if larger than one second
                return [v * 1e-6 for v in as_list]
            return as_list

        def _gate_in_seconds(value, default_seconds):
            if value is None:
                return default_seconds
            if value > 1.0:  # assume nanoseconds
                return float(value) * 1e-9
            return float(value)

        t1_values_seconds = _values_in_seconds(list(t1_map.values()), 50.0 * 1e-6)
        t2_values_seconds = _values_in_seconds(list(t2_map.values()), 70.0 * 1e-6)
        gate_1q_seconds = _gate_in_seconds(gate_times.get("single"), 50.0 * 1e-9)
        gate_2q_seconds = _gate_in_seconds(gate_times.get("two"), 300.0 * 1e-9)

        avg_t1_seconds = float(np.mean(t1_values_seconds)) if t1_values_seconds else 50.0 * 1e-6
        avg_t2_seconds = float(np.mean(t2_values_seconds)) if t2_values_seconds else 70.0 * 1e-6

        # --- Component fidelities
        f_gate_1q = max(0.0, 1.0 - dep1q)
        f_gate_2q = max(0.0, 1.0 - dep2q)
        f_readout = max(0.0, 1.0 - readout_p)

        # Coherence via T2 (seconds)
        eps = 1e-12
        avg_t2_safe = max(avg_t2_seconds, eps)
        f_coh_1q = math.exp(-gate_1q_seconds / avg_t2_safe)
        f_coh_2q = math.exp(-gate_2q_seconds / avg_t2_safe)
        f_coherence = (f_coh_1q + f_coh_2q) / 2.0

        components = [f_gate_1q, f_gate_2q, f_readout, f_coherence]
        fom_total = float(np.prod(components) ** (1.0 / len(components)))

        properties = {
            "backend_qubits": int(getattr(backend_adapter, "n_qubits", 0)),
            "backend_name": getattr(type(backend_adapter), "__name__", ""),
            "T1": float(avg_t1_seconds),
            "T2": float(avg_t2_seconds),
            "overall_noise_score": float(fom_total),
            "single_qubit_gate_fidelity": float(f_gate_1q),
            "two_qubit_gate_fidelity": float(f_gate_2q),
            "readout_fidelity": float(f_readout),
            "coherence_factor": float(f_coherence),
            "single_qubit_gate_duration_seconds": float(gate_1q_seconds),
            "two_qubit_gate_duration_seconds": float(gate_2q_seconds),
            "thermal_population_fraction": float(thermal_pop), 
        }

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "figure_of_merit": self.__class__.__name__,
            "properties": properties,
            "experiment_result": None,
        }
