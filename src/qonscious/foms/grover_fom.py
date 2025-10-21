# grade_benchmark.py

import numpy as np
import random
import math
from typing import Dict, List, Optional, Any, Callable
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from qiskit import QuantumCircuit
from qonscious.foms.figure_of_merit import FigureOfMerit
if TYPE_CHECKING:
    from qonscious.adapters.backend_adapter import BackendAdapter
    from qonscious.results.result_types import ExperimentResult, FigureOfMeritResult
# --- Estructuras de Datos Estándar ---


# --- Funciones de Generación y Codificación (Algoritmo 1 y 2) ---

def _generate_search_params(num_targets: int) -> (int, int, List[str]):
    """Calcula n, N y targets binarios aleatorios (Algoritmo 1)."""
    if num_targets <= 0:
        raise ValueError("num_targets debe ser positivo.")
    
    n = math.ceil(math.log2(num_targets))
    if n == 0:
         n = 1
    N = 2**n
    
    targets_int = random.sample(range(N), num_targets)
    targets_binary = [format(t, f'0{n}b') for t in targets_int]
    
    return n, N, targets_binary

# ALGORITMO 3: Construcción del Oráculo Multi-Target
def _construct_oracle(marked_states: List[str], num_qubits: int) -> QuantumCircuit:
    """Construye un oráculo de fase para múltiples targets (Algoritmo 3)."""
    qc = QuantumCircuit(num_qubits, name="Oracle")
    target_qubit = num_qubits - 1
    
    for target in marked_states:
        zero_indices = [i for i, bit in enumerate(target) if bit == '0']
        
        # Pre-conditioning X gates (para convertir 0s en 1s para el MCX) 
        for i in zero_indices:
            qc.x(i)

        # H-MCX-H = MCZ Phase Flip 
        qc.h(target_qubit)
        
        control_qubits = list(range(num_qubits - 1))
        
        if num_qubits > 1:
             qc.mcx(control_qubits, target_qubit)
        elif num_qubits == 1:
             # Si n=1, el MCX sin control es una X (la secuencia H X H = Z)
             qc.x(target_qubit) 

        qc.h(target_qubit)

        # Post-conditioning X gates (limpiar) 
        for i in zero_indices:
            qc.x(i)
            
    return qc

# Helper: Construcción del Difusor
def _construct_diffusion(num_qubits: int) -> QuantumCircuit:
    """Construye el operador de Difusión de Grover."""
    dq = QuantumCircuit(num_qubits, name='Diffusion')
    dq.h(range(num_qubits))
    dq.x(range(num_qubits))
    
    # Inversión de fase del estado |0...0> (Implementación MCZ)
    
    if num_qubits > 1:
        # MCZ sobre todos los cúbits
        dq.h(num_qubits-1)
        dq.mcx(list(range(num_qubits - 1)), num_qubits-1)
        dq.h(num_qubits-1)
    elif num_qubits == 1:
        # En n=1, la inversión sobre |0> es una compuerta Z
        dq.z(0) 

    dq.x(range(num_qubits))
    dq.h(range(num_qubits))
    return dq

# ALGORITMO 4: Cálculo de la Puntuación
def _compute_grade_score(counts: Dict[str, int], target_states: List[str], shots: int, lambd: float, mu: float) -> Dict[str, Any]:
    """Calcula la puntuación GRADE y sus componentes (Algoritmo 4)."""
    P = {state: count / shots for state, count in counts.items()}
    
    # 1. P_T (Probabilidad Acumulada del Objetivo) , Eq. 1
    P_T = sum(P.get(s, 0.0) for s in target_states)
    
    # 2. P_N (Probabilidad de Estados No Objetivo) , Eq. 3
    P_N = 1.0 - P_T
    M = len(target_states)
    
    # 3. sigma_T (Desviación Estándar de Targets) , Eq. 2
    P_s_list = [P.get(s, 0.0) for s in target_states]

    if M > 0:
        P_bar_T = P_T / M
        sum_sq_diff = sum((P_s - P_bar_T)**2 for P_s in P_s_list)
        sigma_T = math.sqrt(sum_sq_diff / M)
    else:
        sigma_T = 0.0
        
    # 4. Score Base: Score = PT - lambda*sigma_T - mu*PN , Eq. 1
    score_raw = P_T - (lambd * sigma_T) - (mu * P_N)
    
    score = score_raw 
    
    # 5. Restricción del Cero (Fail-Safe Operacional) , p. 7-8
    if mu * P_N >= P_T: 
        score = 0.0
        
    # Retorna métricas clave
    return {
        "score": max(0.0, score), # Asegura que el score final sea no negativo
        "P_T": P_T,
        "sigma_T": sigma_T,
        "P_N": P_N,
    }

# --- CLASE PRINCIPAL FIGURE OF MERIT ---

class GroverFigureOfMerit(FigureOfMerit):
    """
    Implementa el benchmark GRADE siguiendo el protocolo FigureOfMerit.
    Los parámetros por defecto se pueden sobrescribir vía kwargs en evaluate.
    """
    def __init__(self, num_targets: int, lambd: float = 1.0, mu: float = 1.0):
        self.default_num_targets = num_targets
        self.default_lambd = lambd
        self.default_mu = mu

    def evaluate(self, backend_adapter: BackendAdapter, **kwargs) -> FigureOfMeritResult:
        """
        Ejecuta el circuito de Grover, simula la ejecución en el backend
        y calcula la puntuación GRADE.
        """
        shots = kwargs.get("shots", 1024)
        lambd = kwargs.get("lambda_factor", self.default_lambd)
        mu = kwargs.get("mu_factor", self.default_mu)
        num_targets = kwargs.get("num_targets", self.default_num_targets)
        
        # I. Generación de parámetros
        try:
            n, N, targets_binary = _generate_search_params(num_targets)
        except ValueError:
             # Devuelve resultado de fallo si la entrada es inválida
             return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "figure_of_merit": self.__class__.__name__,
                "properties": {"score": 0.0, "error": "Invalid num_targets"},
                "experiment_result": None
             }
        
        M = len(targets_binary)
        R = int(round((math.pi / 4) * math.sqrt(N / M))) if M > 0 and N > 0 else 1
        
        # II. Construcción del Circuito
        qc = QuantumCircuit(n, n) 
        qc.h(range(n)) 
        
        # Oráculo (Alg 3) y Difusor
        oracle_qc = _construct_oracle(targets_binary, n)
        diffusion_qc = _construct_diffusion(n)
        
        oracle_gate = oracle_qc.to_gate(label="GRADE_Oracle")
        diffusion_gate = diffusion_qc.to_gate(label="Grover_Diffuser")

        # Aplicar R iteraciones
        for _ in range(R):
            qc.append(oracle_gate, range(n))
            qc.append(diffusion_gate, range(n))
            qc.barrier()
            
        qc.measure(range(n), range(n))

        # III. Ejecución (Agnóstica al Backend)
        # El adaptador maneja la transpilación y la ejecución
        experiment_result: ExperimentResult = backend_adapter.run(qc, shots=shots) 
        raw_counts: Dict[str, int] = experiment_result.get("counts", {})
        
        # IV. Cálculo del Score (Alg 4)
        score_metrics = _compute_grade_score(raw_counts, targets_binary, shots, lambd, mu)
        
        # V. Formato FigureOfMeritResult
        properties_details = {
            "num_qubits": n,
            "search_space_size": N,
            "targets_count": M,
            "grover_iterations": R,
            "target_states": targets_binary,
            **score_metrics, # 'score', P_T, sigma_T, P_N
            "lambda_factor": lambd, 
            "mu_factor": mu,       
            "shots": shots         
        }
        
        evaluation_result: FigureOfMeritResult = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "figure_of_merit": self.__class__.__name__,
            "properties": properties_details, # Contiene el 'score'
            "experiment_result": experiment_result,
        }
        
        return evaluation_result

# Fin de grade_benchmark.py
