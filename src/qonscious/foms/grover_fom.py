# grade_fom.py
"""GRADE: Figure of Merit basada en Grover (simple, sin barriers)."""

from __future__ import annotations

import math
import random
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from qiskit import QuantumCircuit

from qonscious.foms.figure_of_merit import FigureOfMerit

if TYPE_CHECKING:
    from qonscious.adapters.backend_adapter import BackendAdapter
    from qonscious.results.result_types import ExperimentResult, FigureOfMeritResult


class GroverFigureOfMerit(FigureOfMerit):
    """
    Grover multi-objetivo: aplica fase -1 a varios estados marcados y mide.
    Estructura estilo CHSH: __init__, compute_required_shots, evaluate. Helpers internos.
    SIN barreras entre iteraciones.
    """

    def __init__(
        self,
        num_targets: int, #parametro obligatorio
        lambda_factor: float,
        mu_factor: float,
        shots: int = 1024,
        num_qubits: int | None = None,
        targets_int: list[int] | None = None,
    ) -> None:
        self.num_targets = int(num_targets)
        self.lambda_factor = float(lambda_factor)
        self.mu_factor = float(mu_factor)
        self.shots = int(shots)
        self.num_qubits = None if num_qubits is None else int(num_qubits)
        self.targets_int = None if targets_int is None else list(targets_int)


    def compute_required_shots(self) -> int:
        return 2000  # fijo, tomado del __init__

    def evaluate(self, backend_adapter: BackendAdapter) -> FigureOfMeritResult:
        """Ejecuta Grover con la config de self y devuelve métricas + resultado crudo."""
        # 1) Espacio y targets
        search_space, target_bitstrings = self._make_search_space_and_targets(
            self.num_targets,
            self.num_qubits,
            self.targets_int
        )
        M = len(target_bitstrings)                                # cantidad de targets
        n = len(target_bitstrings[0]) if M > 0 else 1           # qubits efectivos e.g. "000" = 3
        N = len(search_space)                                     # tamaño del espacio
        R = self._optimal_rounds(N, M)                            # iteraciones óptimas

        # 2) Circuito Grover (SIN barriers)
        qc = self._build_grover_circuit(n, target_bitstrings, R)

        # 3) Run
        run_result: ExperimentResult = backend_adapter.run(qc, shots=self.shots)
        if run_result is None:
            raise RuntimeError("backend_adapter.run devolvió None.")
        counts = (
            run_result.get("counts", {})
            if isinstance(run_result, dict)
            else getattr(run_result, "counts", {})
        )

        # 4) Score
        metrics = self._compute_score(counts, target_bitstrings, self.shots, self.lambda_factor,
                                       self.mu_factor)

        # 5) Empaquetar
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "figure_of_merit": self.__class__.__name__,
            "properties": {
                "num_qubits": n,
                "search_space_size": N,
                "targets_count": M,
                "grover_iterations": R,
                "target_states": target_bitstrings,
                "lambda_factor": self.lambda_factor,
                "mu_factor": self.mu_factor,
                "shots": self.shots,
                **metrics,
            },
            "experiment_result": run_result,
        }

    # --- Internos (posicionales, simples) ---
    def _build_grover_circuit(self, n: int, targets: list[str], R: int) -> QuantumCircuit:
        qc = QuantumCircuit(n, n, name="Grover")
        qc.h(range(n))
        oracle = self._build_oracle(targets, n)
        diffusion = self._build_diffusion(n)
        for _ in range(R):
            qc.compose(oracle, qubits=range(n), inplace=True)
            qc.compose(diffusion, qubits=range(n), inplace=True)
        qc.measure(range(n), range(n))
        return qc

    def _make_search_space_and_targets(
        self,
        num_targets: int,
        num_qubits: int | None,
        targets_int: list[int] | None,
    ) -> tuple[list[int], list[str]]:
        # Elegir n y N
        if num_qubits is not None:
            n = int(num_qubits)
            N = 2**n
            if N is not None and N <= 0:
                raise ValueError("search_space_size debe ser > 0")
            max_real = N
        else:
            n = max(1, math.ceil(math.log2(max(num_targets, 1))))
            N = 2**n
            max_real = N

        # Elegir targets
        real_space = list(range(max_real))
        if targets_int is None:
            if num_targets > len(real_space):
                raise ValueError(f"num_targets ({num_targets}) > tamaño del espacio real ({len(real_space)})")
            chosen = random.sample(real_space, k=num_targets)
        else:
            chosen = list(targets_int)
            for t in chosen:
                if not (0 <= t < max_real):
                    raise ValueError(f"target fuera de rango real: {t} ∉ [0,{max_real-1}]")

        targets_binary = [format(t, f"0{n}b") for t in chosen]
        search_space = list(range(N))
        return search_space, targets_binary

    def _build_oracle(self, marked: list[str], n: int) -> QuantumCircuit:
        qc = QuantumCircuit(n, name="Oracle")
        tgt = n - 1
        for bitstr in marked:
            bits_le = list(reversed(bitstr))#cambio de endian
            zeros = [i for i, b in enumerate(bits_le) if b == "0"]
            for i in zeros:
                qc.x(i)
            if n > 1:
                qc.h(tgt)
                qc.mcx(list(range(n - 1)), tgt)
                qc.h(tgt)
            else:
                qc.z(tgt)  # n=1: Z directo (evita H-Z-H= X)
            for i in zeros:
                qc.x(i)
        return qc

    def _build_diffusion(self, n: int) -> QuantumCircuit:
        dq = QuantumCircuit(n, name="Diffusion")
        dq.h(range(n))
        dq.x(range(n))
        if n > 1:
            dq.h(n - 1)
            dq.mcx(list(range(n - 1)), n - 1)
            dq.h(n - 1)
        else:
            dq.z(0)
        dq.x(range(n))
        dq.h(range(n))
        return dq

    def _optimal_rounds(self, N: int, M: int) -> int:
        R = math.floor((math.pi / 4) * math.sqrt(N/M))
        return max(0, R)

    def _compute_score(
        self,
        counts: dict[str, int],
        targets: list[str],
        shots: int,
        lambd: float,
        mu: float,
    ) -> dict[str, Any]:
        if shots <= 0:
            return {"score": 0.0, "P_T": 0.0, "sigma_T": 0.0, "P_N": 1.0}
        P = {s: c / shots for s, c in counts.items()}
        #calcular
        P_T = sum(P.get(s, 0.0) for s in targets)
        P_N = 1.0 - P_T
        M = len(targets)
        if M:
            p_list = [P.get(s, 0.0) for s in targets]
            p_bar = P_T / M
            sigma_T = (sum((p - p_bar) ** 2 for p in p_list) / M) ** 0.5
        else:
            sigma_T = 0.0
        raw = P_T - (lambd * sigma_T) - (mu * P_N)
        score = 0.0 if (mu * P_N >= P_T) else max(0.0, raw)
        return {"score": score, "P_T": P_T, "sigma_T": sigma_T, "P_N": P_N}

