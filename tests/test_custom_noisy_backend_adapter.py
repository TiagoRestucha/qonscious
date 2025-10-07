from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from qiskit import QuantumCircuit

from qonscious.adapters.custom_noisy_backend_adapter import CustomNoisyBackendAdapter

if TYPE_CHECKING:
    from qonscious.results.result_types import ExperimentResult


def test_custom_noisy_backend_basic_run():
    """Test basic circuit execution with custom noisy backend."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    adapter = CustomNoisyBackendAdapter(n_qubits=5)
    result: ExperimentResult = adapter.run(qc, shots=1024)

    # Check result structure
    assert isinstance(result, dict)
    assert set(result.keys()) >= {
        "counts",
        "shots",
        "timestamps",
        "raw_results",
        "backend_properties",
    }

    # Validate counts format
    counts = result["counts"]
    assert isinstance(counts, dict)
    assert all(isinstance(k, str) and len(k) == 2 for k in counts)
    assert all(isinstance(v, int) and v >= 0 for v in counts.values())
    assert sum(counts.values()) == 1024

    # Validate backend name
    assert result["backend_properties"]["name"] == "CustomNoisyBackendAdapter"
    
    # Validate noise model metadata
    noise_info = result["backend_properties"]["noise_model"]
    assert "n_qubits" in noise_info
    assert "depol_prob_1q" in noise_info
    assert "depol_prob_2q" in noise_info
    assert "readout_error_prob" in noise_info


def test_custom_noisy_backend_with_noise():
    """Test that noise actually affects the results."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    # High noise backend
    adapter = CustomNoisyBackendAdapter(
        n_qubits=5,
        depol_prob_1q=0.05,
        depol_prob_2q=0.1,
        readout_error_prob=0.05,
    )
    result: ExperimentResult = adapter.run(qc, shots=2000)

    counts = result["counts"]
    
    # With high noise, we should see some errors (not perfect Bell state)
    # Perfect Bell state would only have '00' and '11'
    # With noise, we should see some '01' and '10'
    total_errors = counts.get("01", 0) + counts.get("10", 0)
    
    # With this level of noise, we expect at least some errors
    assert total_errors > 0, "Expected some noise-induced errors"


def test_t1_t2_properties():
    """Test T1 and T2 properties."""
    adapter = CustomNoisyBackendAdapter(
        n_qubits=3,
        t1_times={0: 40.0, 1: 50.0, 2: 60.0},
        t2_times={0: 60.0, 1: 70.0, 2: 80.0},
    )
    
    t1s = adapter.t1s
    t2s = adapter.t2s
    
    assert len(t1s) == 3
    assert len(t2s) == 3
    assert t1s[0] == 40.0
    assert t1s[1] == 50.0
    assert t1s[2] == 60.0
    assert t2s[0] == 60.0
    assert t2s[1] == 70.0
    assert t2s[2] == 80.0


def test_n_qubits_property():
    """Test n_qubits property."""
    adapter = CustomNoisyBackendAdapter(n_qubits=7)
    assert adapter.n_qubits == 7


def test_noise_model_access():
    """Test that we can access the noise model."""
    adapter = CustomNoisyBackendAdapter(n_qubits=5)
    noise_model = adapter.get_noise_model()
    
    assert noise_model is not None
    assert hasattr(noise_model, "noise_qubits")


def test_update_noise_parameters():
    """Test updating noise parameters dynamically."""
    adapter = CustomNoisyBackendAdapter(
        n_qubits=3,
        depol_prob_1q=0.001,
        depol_prob_2q=0.01,
    )
    
    # Initial values
    assert adapter.depol_prob_1q == 0.001
    assert adapter.depol_prob_2q == 0.01
    
    # Update parameters
    adapter.update_noise_parameters(
        depol_prob_1q=0.005,
        depol_prob_2q=0.02,
    )
    
    # Check updated values
    assert adapter.depol_prob_1q == 0.005
    assert adapter.depol_prob_2q == 0.02


def test_custom_errors_disabled():
    """Test backend with custom errors disabled."""
    adapter = CustomNoisyBackendAdapter(
        n_qubits=3,
        enable_custom_errors=False,
    )
    
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    
    result: ExperimentResult = adapter.run(qc, shots=1024)
    
    # Should still work, just without custom Kraus/SuperOp errors
    assert isinstance(result, dict)
    assert sum(result["counts"].values()) == 1024


def test_transpile():
    """Test circuit transpilation."""
    adapter = CustomNoisyBackendAdapter(n_qubits=5)
    
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure_all()
    
    transpiled = adapter.transpile(qc)
    
    assert isinstance(transpiled, QuantumCircuit)
    assert transpiled.num_qubits >= qc.num_qubits


def test_different_gate_times():
    """Test backend with custom gate times."""
    adapter = CustomNoisyBackendAdapter(
        n_qubits=3,
        gate_times={"single": 100, "two": 500},
    )
    
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    
    result: ExperimentResult = adapter.run(qc, shots=1024)
    
    assert isinstance(result, dict)
    assert sum(result["counts"].values()) == 1024


def test_thermal_population():
    """Test backend with different thermal population."""
    adapter = CustomNoisyBackendAdapter(
        n_qubits=3,
        thermal_population=0.05,  # Higher thermal population
    )
    
    qc = QuantumCircuit(1)
    qc.measure_all()
    
    result: ExperimentResult = adapter.run(qc, shots=5000)
    
    # With thermal population, we should see some |1⟩ states even without gates
    counts = result["counts"]
    ones_count = counts.get("1", 0)
    
    # Should see some thermal excitation
    assert ones_count > 0, "Expected some thermal excitation"


def test_print_noise_summary(capsys):
    """Test noise summary printing."""
    adapter = CustomNoisyBackendAdapter(n_qubits=3)
    
    adapter.print_noise_summary()
    
    captured = capsys.readouterr()
    assert "Custom Noisy Backend Adapter" in captured.out
    assert "Number of qubits: 3" in captured.out
    assert "Depolarizing Errors" in captured.out
    assert "Thermal Relaxation" in captured.out
    assert "Readout Errors" in captured.out


def test_bell_state_with_low_noise():
    """Test Bell state preparation with low noise."""
    adapter = CustomNoisyBackendAdapter(
        n_qubits=5,
        depol_prob_1q=0.0001,
        depol_prob_2q=0.001,
        readout_error_prob=0.001,
    )
    
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    
    result: ExperimentResult = adapter.run(qc, shots=2000)
    
    counts = result["counts"]
    
    # With low noise, most results should be '00' or '11'
    bell_states = counts.get("00", 0) + counts.get("11", 0)
    error_states = counts.get("01", 0) + counts.get("10", 0)
    
    # Bell states should dominate
    assert bell_states > error_states * 10


def test_ghz_state_with_noise():
    """Test GHZ state preparation with noise."""
    adapter = CustomNoisyBackendAdapter(
        n_qubits=5,
        depol_prob_1q=0.01,
        depol_prob_2q=0.02,
    )
    
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure_all()
    
    result: ExperimentResult = adapter.run(qc, shots=2000)
    
    counts = result["counts"]
    
    # GHZ state should have '000' and '111' as dominant outcomes
    # But with noise, we'll see other states too
    assert len(counts) > 2, "Expected noise to introduce various outcomes"
