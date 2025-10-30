# grade_fom.py
"""GRADE: Figure of Merit basada en Grover para Qonscious."""

from __future__ import annotations

import math
import random
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from qiskit import QuantumCircuit

from qonscious.foms.figure_of_merit import FigureOfMerit

if TYPE_CHECKING:
    from qonscious.adapters.backend_adapter import BackendAdapter
    from qonscious.results.result_types import FigureOfMeritResult


# -------------------------- Helpers Grover --------------------------

def _optimal_grover_rounds(N: int, M: int) -> int:
    """Número óptimo de iteraciones de Grover (R)."""
    if not (0 < M < N):
        return 0
    theta = math.asin(math.sqrt(M / N))
    R = int(math.floor((math.pi / (4 * theta)) - 0.5))
    return max(0, R)


def _generate_search_params(
    num_targets: int,
    num_qubits: int | None = None,
    search_space_size: int | None = None,
    targets_int: list[int] | None = None,
) -> tuple[list[int], list[str]]:
    """Devuelve (search_space, targets_binary) para Grover."""
    # Elegir n y N
    if num_qubits is not None:
        n = int(num_qubits)
        N = 2**n
    elif search_space_size is not None:
        N_real = int(search_space_size)
        if N_real <= 0:
            raise ValueError("search_space_size debe ser > 0")
        n = math.ceil(math.log2(N_real))
        N = 2**n
    else:
        n = max(1, math.ceil(math.log2(num_targets)))
        N = 2**n

    # Elegir targets
    max_real = search_space_size if search_space_size is not None else N
    space_real = list(range(max_real))

    if targets_int is None:
        if num_targets > len(space_real):
            raise ValueError(
                f"num_targets ({num_targets}) > tamaño del espacio real ({len(space_real)})"
            )
        targets_int = random.sample(space_real, k=num_targets)
    else:
        for t in targets_int:
            if not (0 <= t < max_real):
                raise ValueError(f"target fuera de rango real: {t} ∉ [0,{max_real-1}]")

    targets_binary = [format(t, f"0{n}b") for t in targets_int]
    search_space = list(range(N))
    return search_space, targets_binary


def _construct_oracle(marked_states: list[str], num_qubits: int) -> QuantumCircuit:
    """Oráculo multi-objetivo: aplica fase −1 a cada estado marcado."""
    qc = QuantumCircuit(num_qubits, name="Oracle")
    target_q = num_qubits - 1

    for target in marked_states:
        bits_le = list(reversed(target))
        zero_idx = [i for i, b in enumerate(bits_le) if b == "0"]

        for i in zero_idx:
            qc.x(i)

        qc.h(target_q)
        if num_qubits > 1:
            qc.mcx(list(range(num_qubits - 1)), target_q)
        else:
            qc.z(target_q)
        qc.h(target_q)

        for i in zero_idx:
            qc.x(i)

    return qc


def _construct_diffusion(num_qubits: int) -> QuantumCircuit:
    """Difusor estándar (inversión sobre la media)."""
    dq = QuantumCircuit(num_qubits, name="Diffusion")
    dq.h(range(num_qubits))
    dq.x(range(num_qubits))
    if num_qubits > 1:
        dq.h(num_qubits - 1)
        dq.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        dq.h(num_qubits - 1)
    else:
        dq.z(0)
    dq.x(range(num_qubits))
    dq.h(range(num_qubits))
    return dq


def _compute_grade_score(
    counts: dict[str, int],
    target_states: list[str],
    shots: int,
    lambd: float,
    mu: float,
) -> dict[str, Any]:
    """Score = P_T − λ·σ_T − μ·P_N, con fail-safe (score=0 si μ·P_N ≥ P_T)."""
    P = {s: c / shots for s, c in counts.items()}
    P_T = sum(P.get(s, 0.0) for s in target_states)
    P_N = 1.0 - P_T

    M = len(target_states)
    if M:
        p_list = [P.get(s, 0.0) for s in target_states]
        p_bar = P_T / M
        sigma_T = math.sqrt(sum((p - p_bar) ** 2 for p in p_list) / M)
    else:
        sigma_T = 0.0

    score = P_T - (lambd * sigma_T) - (mu * P_N)
    if mu * P_N >= P_T:
        score = 0.0

    return {
        "score": max(0.0, score),
        "P_T": P_T,
        "sigma_T": sigma_T,
        "P_N": P_N,
    }


# -------------------------- FoM principal --------------------------

class GroverFigureOfMerit(FigureOfMerit):
    """GRADE: corre Grover multi-objetivo y puntúa el resultado."""

    def __init__(
        self,
        num_targets: int,
        lambd: float = 1.0,
        mu: float = 1.0,
        *,
        default_num_qubits: int | None = None,
        default_search_space_size: int | None = None,
        default_targets_int: list[int] | None = None,
        default_shots: int = 1024,
    ):
        self.default_num_targets = num_targets
        self.default_lambd = lambd
        self.default_mu = mu
        self.default_num_qubits = default_num_qubits
        self.default_search_space_size = default_search_space_size
        self.default_targets_int = default_targets_int
        self.default_shots = default_shots

    def evaluate(self, backend_adapter: BackendAdapter, **kwargs) -> FigureOfMeritResult:
        # Parámetros efectivos
        shots = kwargs.get("shots", self.default_shots)
        lambd = kwargs.get("lambda_factor", self.default_lambd)
        mu = kwargs.get("mu_factor", self.default_mu)
        n_user = kwargs.get("num_qubits", self.default_num_qubits)
        N_user = kwargs.get("search_space_size", self.default_search_space_size)
        T_user = kwargs.get("targets_int", self.default_targets_int)

        # Espacio y targets
        M_req = kwargs.get("num_targets", self.default_num_targets)
        search_space, targets_binary = _generate_search_params(
            num_targets=M_req,
            num_qubits=n_user,
            search_space_size=N_user,
            targets_int=T_user,
        )

        M = len(targets_binary)
        n = len(targets_binary[0]) if M > 0 else 1
        N = len(search_space)
        R = _optimal_grover_rounds(N, M)

        # Circuito Grover
        qc = QuantumCircuit(n, n)
        qc.h(range(n))
        oracle_qc = _construct_oracle(targets_binary, n)
        diffusion_qc = _construct_diffusion(n)
        for _ in range(R):
            qc.compose(oracle_qc, qubits=range(n), inplace=True)
            qc.compose(diffusion_qc, qubits=range(n), inplace=True)
            qc.barrier()
        qc.measure(range(n), range(n))

        # Ejecución
        experiment_result = backend_adapter.run(qc, shots=shots)
        if experiment_result is None:
            raise RuntimeError("backend_adapter.run devolvió None.")
        counts = (
            experiment_result.get("counts", {})
            if isinstance(experiment_result, dict)
            else getattr(experiment_result, "counts", {})
        )

        # Score
        metrics = _compute_grade_score(counts, targets_binary, shots, lambd, mu)

        properties = {
            "num_qubits": n,
            "search_space_size": N,
            "targets_count": M,
            "grover_iterations": R,
            "target_states": targets_binary,
            **metrics,
            "lambda_factor": lambd,
            "mu_factor": mu,
            "shots": shots,
        }

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "figure_of_merit": self.__class__.__name__,
            "properties": properties,
            "experiment_result": experiment_result,
        }
