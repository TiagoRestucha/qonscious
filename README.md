[![CI](https://github.com/lifia-unlp/qonscious/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/lifia-unlp/qonscious/actions/workflows/ci.yml)

**Qonscious** is a runtime framework designed to support the conditional execution of quantum circuits based on resource introspection. It helps you build quantum applications that are aware of backend conditions — such as entanglement, coherence, or fidelity — before execution.

# Why Qonscious?

In the NISQ era, quantum hardware is noisy, resource-limited, and variable over time. Static resource assumptions lead to unreliable results. **Qonscious** makes quantum programs introspective and adaptive.

For a deeper discussion on the motivation behind Qonscious, read [our article](https://arxiv.org/html/2508.19276v1)

# Key Features

- Figures of Merit evaluation (e.g., get CHSH score, T1, T2, ...)
- Conditional execution on compliance with figures of merit checks
- One circuit, many backends: abstract backends and hide complexity behind adaptors (currently available for SampleV2, Aer Simulator, IBM Backends, IBM Simulators, IONQ backends)
- Inversion of control: pass a callback, not only a circuit
- Rich, uniform results from all backends, including backend configuration, and any figures of merit you need as conditional context
- Built-in logging, extensibility, and fallback logic

# Use cases

These are some scenarios where you may use Qonscious:

- Run a circuit conditional on your target computer (or simulator) checking some figures of merit (e.g., number of qubits, CHSH score, etc.)
- Benchmark a computer (or simulator) in terms of a collection of figures of merit.
- Explore correlations between experiment results and figures of merit of a given computer (or simulator)
- ...

# Installation

We encourage installing Qiskit via pip to make sure you have the latest released version:

````
pip install qonscious
````

If you preffer working on the source code (or you'd like to contribute to the development of Qonscious read the [instructions for contributos](CONTRIBUTING.md))

# Examples

The [notebooks](./notebooks/) folder contains several examples of using Qonscious in different use cases. 

We suggest you start with **chsh_test_demo.ipynb**

# Documentation

Up-to-date documentation is available on [github pages](https://lifia-unlp.github.io/qonscious/)







