# grade_benchmark.py
"""
Este script implementa el benchmark GRADE (Grover's Algorithm Details Evaluation)
como una "Figure of Merit" (FoM) para la plataforma Qonscious.
El objetivo de GRADE es evaluar la calidad de un QPU (Quantum Processing Unit)
midiendo su rendimiento en la ejecución del algoritmo de Grover bajo diferentes
condiciones, penalizando no solo la baja probabilidad de éxito, sino también
la variabilidad en la medición de los estados objetivo.
"""

from __future__ import annotations
import numpy as np
import random
import math
from typing import TYPE_CHECKING, Dict, List, Optional, Any, Callable, Tuple
from collections import defaultdict
from datetime import datetime, timezone

# --- Importaciones de Computación Cuántica ---
from qiskit import QuantumCircuit, transpile
from qonscious.foms.figure_of_merit import FigureOfMerit

# Type checking para evitar importaciones circulares con los adapters/results
if TYPE_CHECKING:
    from qonscious.adapters.backend_adapter import BackendAdapter
    from qonscious.results.result_types import ExperimentResult, FigureOfMeritResult


# --- Funciones Auxiliares del Algoritmo de Grover ---
#   Duda, ¿ponerlas como métodos estáticos dentro de la clase?
def _optimal_grover_rounds(N: int, M: int) -> int:
    """
    Calcula el número óptimo de iteraciones de Grover (R).

    La fórmula teórica para maximizar la probabilidad de éxito es:
    R = floor( (pi / (4 * theta)) - 0.5 )
    donde theta = asin(sqrt(M / N)).

    Args:
        N (int): Tamaño total del espacio de búsqueda (2^n).
        M (int): Número de estados objetivo (targets).

    Returns:
        int: Número óptimo de rondas (R).
    """
    if M <= 0 or M >= N:
        # Casos degenerados: 
        # Si M=0, no hay nada que buscar.
        # Si M=N, todos son targets (la probabilidad inicial ya es 1).
        return 0
    
    # Ángulo theta
    theta = math.asin(math.sqrt(M / N))
    
    # Cálculo de R
    R = int(math.floor((math.pi / (4 * theta)) - 0.5))
    
    # R debe ser al menos 0 (en casos donde M/N es grande, R puede dar negativo)
    return max(0, R)


def _generate_search_params(
    num_targets: int,
    num_qubits: int | None = None,
    search_space_size: int | None = None,
    targets_int: list[int] | None = None,
) -> tuple[list[int], list[str]]:#tuple para poder devolver ambos valores
    """
    Determina los parámetros del problema de búsqueda: n, N, y los targets.

    *** Nota sobre el Paper GRADE ***
    Esta función fusiona la lógica de los Algoritmos 1 y 2 del paper:

    -   Algoritmo 1 (Fallback): Define el problema si el usuario SOLO especifica
        el número de targets (M). En este caso, n se calcula como el mínimo
        necesario para albergar M targets (n = ceil(log2(M))).
    
    -   Algoritmo 2 (User-Defined): Define el problema si el usuario especifica
        el tamaño del espacio de búsqueda, ya sea con `num_qubits` (n) o 
        `search_space_size` (N_real).

    Esta función maneja ambas lógicas para generar la configuración del benchmark.
    """
    
    import math, random
    
    # --- Parte 1: Elegimos n (qubits) y N (espacio 2^n) ---
    # Esta sección implementa la lógica fusionada de Alg. 1 y 2.

    if num_qubits is not None:
        # Lógica Algoritmo 2 (Caso A): El usuario fija n
        n = int(num_qubits)
        N = 2**n
    
    elif search_space_size is not None:
        # Lógica Algoritmo 2 (Caso B): El usuario fija un N "real"
        N_real = int(search_space_size)
        if N_real <= 0: raise ValueError("search_space_size debe ser > 0")
        
        # n debe ser suficiente para codificar N_real estados (0...N_real-1)
        n = math.ceil(math.log2(N_real))
        # N (el espacio de Hilbert) es la potencia de 2 que lo contiene
        N = 2**n  
    
    else:
        # Lógica Algoritmo 1 (Fallback): Solo se dio M (num_targets)
        # Se usa el mínimo 'n' que puede contener M targets.
        n = max(1, math.ceil(math.log2(num_targets)))
        N = 2**n

    # --- Parte 2: Elegimos los M targets ---
    
    # El espacio "real" es N si n/N fue elegido por Alg. 1 o Alg. 2A,
    # o N_real si fue elegido por Alg. 2B.
    max_real = search_space_size if search_space_size is not None else N
    space_real = list(range(max_real))

    if targets_int is None:
        # Si el usuario no provee una lista de targets, los elegimos al azar
        if num_targets > len(space_real):
            raise ValueError(f"num_targets ({num_targets}) > tamaño del espacio real ({len(space_real)})")
        targets_int = random.sample(space_real, k=num_targets)
    
    else:
        # Si el usuario provee los targets, validamos que estén en el rango
        for t in targets_int:
            if not (0 <= t < max_real):
                raise ValueError(f"target fuera de rango real: {t} ∉ [0,{max_real-1}]")

    # --- Parte 3: Codificamos targets a bitstrings de n bits ---
    
    # Los enteros se convierten a strings binarios (bitstrings) de longitud n
    targets_binary = [format(t, f'0{n}b') for t in targets_int]
    
    # El espacio de búsqueda que se ejecutará en Qiskit es siempre 0..2^n-1
    search_space = list(range(N))  

    return search_space, targets_binary


# --- ALGORITMO 3: Construcción del Oráculo Multi-Target ---
def _construct_oracle(marked_states: List[str], num_qubits: int) -> QuantumCircuit:
    """
    Construye un oráculo de fase que marca múltiples estados objetivo.
    
    Implementación del Algoritmo 3 del paper GRADE.
    
    El oráculo aplica un cambio de fase (-1) a cada estado en `marked_states`.
    Lo hace iterando sobre cada target y aplicando una compuerta 
    Multi-Controlled Z (MCZ) "customizada" para ese target.
    
    Una MCZ normal aplica fase a |11...1>. Para aplicar fase a |t_1 t_2 ... t_n>,
    primero se "flipea" (con X) los qubits donde t_i es 0, se aplica la MCZ
    a |11...1>, y se vuelve a flipear. REVISAR SI ES ASI
    """
    qc = QuantumCircuit(num_qubits, name="Oracle")
    
    # Usamos el último qubit como el objetivo para la implementación
    # H-MCX-H de la MCZ.
    target_qubit = num_qubits - 1

    for target in marked_states:
        # Qiskit ordena los bits al revés (Little-Endian, qubit 0 es LSB).
        # Revertimos el bitstring para que coincida con los índices de Qiskit.
        bits_le = list(reversed(target))
        
        # Identificamos los qubits que están en '0' en el target
        zero_indices = [i for i, bit in enumerate(bits_le) if bit == '0']

        # 1. Pre-conditioning: Aplicamos X donde el target es '0'
        # Esto transforma el estado |target> en |11...1>
        for i in zero_indices:
            qc.x(i)

        # 2. Aplicar MCZ (Multi-Controlled Z)
        # Se implementa como H en el último qubit, MCX en el último, H de nuevo.
        qc.h(target_qubit)
        
        if num_qubits > 1:
            control_qubits = list(range(num_qubits - 1))
            qc.mcx(control_qubits, target_qubit)
        else:
            # Caso n=1: MCZ es solo Z (o H-X-H)
            qc.x(target_qubit)  
            
        qc.h(target_qubit)

        # 3. Limpiar (Post-conditioning): Invertimos los X iniciales
        # Esto devuelve |11...1> al estado |target> (con la fase aplicada)
        for i in zero_indices:
            qc.x(i)

        # Duda: hay alguna otra manera de construir el oráculo más eficientemente? a lo que me refiero es que si hay muchos targets, este método puede ser costoso.

    return qc

# --- Helper: Construcción del Difusor ---
def _construct_diffusion(num_qubits: int) -> QuantumCircuit:
    """
    Construye el operador de Difusión de Grover (inversión sobre la media).
    
    Este operador es estándar y no depende de los targets.
    Se implementa como: H-X-MCZ-X-H (tener en cuenta que MCZ es la inversión de fase y hay productos tensoriales de kronecker).
    (Aplica una fase -1 solo al estado |0...0> en la base de Hadamard).
    """
    dq = QuantumCircuit(num_qubits, name='Diffusion')
    
    # 1. Aplicar H a todos los qubits
    dq.h(range(num_qubits))
    
    # 2. Aplicar X a todos los qubits (prepara para MCZ sobre |0...0>)
    dq.x(range(num_qubits))
    
    # 3. Inversión de fase del estado |0...0> (Implementación MCZ)
    if num_qubits > 1:
        # MCZ estándar (H-MCX-H) sobre el estado |1...1> (que era |0...0>)
        dq.h(num_qubits-1)
        dq.mcx(list(range(num_qubits - 1)), num_qubits-1)
        dq.h(num_qubits-1)
    elif num_qubits == 1:
        # Caso n=1: La inversión sobre |0> es una compuerta Z
        dq.z(0) 

    # 4. Limpiar X
    dq.x(range(num_qubits))
    
    # 5. Limpiar H
    dq.h(range(num_qubits))
    
    return dq

# --- ALGORITMO 4: Cálculo de la Puntuación (Score) ---
def _compute_grade_score(counts: Dict[str, int], target_states: List[str], shots: int, lambd: float, mu: float) -> Dict[str, Any]:
    """
    Calcula la puntuación GRADE y sus componentes (Algoritmo 4 del paper).
    
    La puntuación es:
    Score = P_T - (lambda * sigma_T) - (mu * P_N)
    
    Donde:
    - P_T: Probabilidad acumulada de *todos* los estados objetivo.
    - P_N: Probabilidad acumulada de *todos* los estados no-objetivo.
    - sigma_T: Desviación estándar de las probabilidades *entre* los estados objetivo.
    - lambda, mu: Pesos de penalización.
    """
    
    # P: Diccionario de probabilidades {estado: prob}
    P = {state: count / shots for state, count in counts.items()}
    
    # 1. P_T (Probabilidad Acumulada del Objetivo) , Eq. 1
    # Suma de las probabilidades de todos los bitstrings que son targets.
    P_T = sum(P.get(s, 0.0) for s in target_states)
    
    # 2. P_N (Probabilidad de Estados No Objetivo) , Eq. 3
    # 1.0 menos la probabilidad de los objetivos.
    P_N = 1.0 - P_T
    M = len(target_states)
    
    # 3. sigma_T (Desviación Estándar de Targets) , Eq. 2
    # Mide cuán "desigual" es la distribución de probabilidad *entre* los targets.
    # Un hardware ideal debería dar la misma probabilidad a todos los targets.
    # Si sigma_T es alto, el hardware está sesgado.
    
    # Lista de probabilidades solo para los estados objetivo
    P_s_list = [P.get(s, 0.0) for s in target_states]

    if M > 0:
        # Promedio de probabilidad POR target
        P_bar_T = P_T / M 
        # Suma de diferencias cuadradas
        sum_sq_diff = sum((P_s - P_bar_T)**2 for P_s in P_s_list)
        # Desviación estándar
        sigma_T = math.sqrt(sum_sq_diff / M)
    else:
        # No hay targets, no hay desviación.
        sigma_T = 0.0
        
    # 4. Score Base: Score = PT - lambda*sigma_T - mu*PN , Eq. 1
    # El score base es P_T, penalizado por la variabilidad (sigma_T)
    # y por la probabilidad de fallo (P_N).
    score_raw = P_T - (lambd * sigma_T) - (mu * P_N)
    
    score = score_raw 
    
    # 5. Restricción del Cero (Fail-Safe Operacional) , p. 7-8
    # Si la penalización por fallos (mu * P_N) es mayor o igual
    # que la probabilidad de éxito (P_T), el benchmark se considera un
    # fallo total y el score es 0.
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
    Implementa el benchmark GRADE (Grover's Algorithm Details Evaluation)
    siguiendo el protocolo FigureOfMerit de Qonscious.
    
    Esta clase orquesta todo el proceso:
    1. Configura el problema (Algoritmos 1 y 2).
    2. Calcula las rondas óptimas.
    3. Construye el oráculo (Algoritmo 3) y el difusor.
    4. Ensambla el circuito de Grover.
    5. Lo ejecuta en el backend.
    6. Calcula el score (Algoritmo 4).
    """
    
    def __init__(self, num_targets: int, lambd: float = 1.0, mu: float = 1.0):
        """
        Inicializa la Figura de Mérito.
        
        Args:
            num_targets (int): El número de targets (M) por defecto.
            lambd (float): El peso de penalización de la desviación (lambda).
            mu (float): El peso de penalización de los no-targets (mu).
        """
        self.default_num_targets = num_targets
        self.default_lambd = lambd
        self.default_mu = mu

    def evaluate(self, backend_adapter: BackendAdapter, **kwargs) -> FigureOfMeritResult:
        """
        Ejecuta el benchmark GRADE completo.
        
        Acepta `kwargs` para sobrescribir los defaults o pasar parámetros
        adicionales (num_qubits, search_space_size, targets_int, shots).
        """
        
        # --- Paso I: Configuración y Pre-cálculo ---
        
        # Obtener parámetros de ejecución (de kwargs o los defaults)
        shots = kwargs.get("shots", 1024)
        lambd = kwargs.get("lambda_factor", self.default_lambd)
        mu    = kwargs.get("mu_factor", self.default_mu)
        M_req = kwargs.get("num_targets", self.default_num_targets)

        # Parámetros opcionales para definir el espacio de búsqueda (Alg. 2)
        n_user  = kwargs.get("num_qubits")
        N_user  = kwargs.get("search_space_size")
        T_user  = kwargs.get("targets_int")

        # Generar los parámetros del problema (n, N, M, targets)
        # (Llamada a la función que fusiona Algoritmos 1 y 2)
        search_space, targets_binary = _generate_search_params(
            num_targets=M_req, num_qubits=n_user, search_space_size=N_user, targets_int=T_user
        )

        # Derivar n, N y M finales (reales)
        M = len(targets_binary)
        n = len(targets_binary[0]) if M > 0 else (n_user if n_user else 1)
        N = len(search_space) # N = 2^n

        # Calcular iteraciones óptimas (R)
        R = _optimal_grover_rounds(N, M)

        # --- Paso II: Construcción del Circuito de Grover ---
        
        # 1. Inicializar circuito con n qubits y n bits clásicos
        qc = QuantumCircuit(n, n)
        
        # 2. Poner en superposición (Hadamard a todos)
        qc.h(range(n))
        qc.barrier() # Barrera visual

        # 3. Construir los operadores
        oracle_qc    = _construct_oracle(targets_binary, n) # (Algoritmo 3)
        diffusion_qc = _construct_diffusion(n)

        # 4. Aplicar R iteraciones de (Oráculo + Difusor)
        for _ in range(R):
            qc.compose(oracle_qc,    qubits=range(n), inplace=True)
            qc.compose(diffusion_qc, qubits=range(n), inplace=True)
            qc.barrier() # Barrera visual entre iteraciones

        # 5. Medir al final
        qc.measure(range(n), range(n))

        # --- Paso III: Ejecución en el Backend ---
        
        # El backend_adapter se encarga de transpilar y ejecutar
        experiment_result: ExperimentResult = backend_adapter.run(qc, shots=shots)
        raw_counts: Dict[str, int] = experiment_result["counts"]

        # --- Paso IV: Cálculo del Score ---
        
        # (Llamada al Algoritmo 4)
        score_metrics = _compute_grade_score(
            raw_counts, targets_binary, shots, lambd, mu
        )

        # --- Paso V: Formateo del Resultado ---
        
        # Recopilar todos los detalles de la ejecución
        properties_details = {
            "num_qubits": n,
            "search_space_size": N,
            "targets_count": M,
            "grover_iterations": R,
            "target_states": targets_binary,
            **score_metrics, # Incluye 'score', 'P_T', 'sigma_T', 'P_N'
            "lambda_factor": lambd,
            "mu_factor": mu,
            "shots": shots,
        }

        # Formato estándar de Qonscious
        evaluation_result: FigureOfMeritResult = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "figure_of_merit": self.__class__.__name__, # "GroverFigureOfMerit"
            "properties": properties_details,
            "experiment_result": experiment_result, # Contiene 'counts', 'job_id', etc.
        }
        
        return evaluation_result