# Official Implementation: GS-KAN (Generalized Sprecher KAN)

This repository contains the official code and implementation for the paper **"GS-KAN: Parameter-Efficient Kolmogorov-Arnold
Networks via Sprecher-Type Shared Basis Functions"** by Oscar Eliasson.

It introduces **GS-KAN (Generalized Sprecher KAN)**, a novel neural architecture grounded in Sprecher's refinement of the Kolmogorov-Arnold theorem, designed to improve parameter efficiency and computational performance.

##  Experiments & Evaluation

The architecture is tested and benchmarked against standard KAN, WavKAN and standard MLPs across three distinct experimental domains:

1.  **Synthetic Function Regression:**
    Evaluates the model's ability to approximate complex mathematical functions with high precision.
2.  **Tabular Data Regression (California Housing):**
    Tests performance on real-world tabular datasets to measure generalization in housing price prediction.
3.  **Multi-Class Classification (Fashion MNIST):**
    Assesses the architecture's capability in image classification tasks compared to established baselines.

**Note:** Results may vary slightly due to hardware differences and GPU non-determinism, despite the use of fixed random seeds.

##  Project Overview

The study benchmarks the following architectures:
*   **GS-KAN (Ours):** Utilizing layer-wise shared basis functions for enhanced efficiency.
*   **WavKAN:** Wavelet-based KAN implementation.
*   **Standard KAN:** The efficient-kan implementation.
*   **General MLP:** Standard Multi-Layer Perceptron baseline with SiLU/ReLU activations.

##  Usage

To reproduce the results from the paper, run the following scripts:

* `function_experiment.py`: Synthetic function approximation (Table 1).
* `housing_experiment.py`: California Housing tabular regression (Table 2).
* `fashion_experiment.py`: Fashion-MNIST classification (Table 3).

*Note: All dependencies must be installed via `requirements.txt` before running.*

##  Installation

To install all dependencies, including the external `efficient-kan` library, run:

```bash
pip install -r requirements.txt