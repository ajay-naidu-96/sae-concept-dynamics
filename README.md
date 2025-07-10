# Sparse Autoencoder Concept Discarding Analysis

## Project Overview

This project investigates whether Sparse Autoencoders (SAEs) can selectively discard or retain specific concepts during training, using MNIST digit classification as a testbed. We explore the experimental settings where SAEs learn to encode, preserve, or eliminate representations through systematic analysis of neuron-level activations and concept recovery capabilities.

## Research Questions

- Can trained SAEs selectively discard certain concepts while retaining others?
- What are the training dynamics that lead to concept discarding vs retention?
- Are there predictable patterns in which concepts get discarded together?

## Methodology

### Experimental Setup
- **Dataset**: MNIST handwritten digits (0-9)
- **Pipeline**: Neural network → activation extraction → SAE training → concept probing
- **Evaluation**: Linear probes trained on SAE neuron activations for digit classification
- **Metrics**: AUC, accuracy, precision/recall for 1-vs-all digit classification

### Key Components
1. **Activation Extraction**: Extract intermediate activations from neural networks on MNIST
2. **SAE Training**: Train sparse autoencoders on extracted activations
3. **Concept Probing**: Logistic regression classifiers on SAE outputs to measure concept recovery
4. **Temporal Analysis**: Track concept recovery throughout training
5. **Cross-SAE Comparison**: Systematic analysis across different training configurations

## Repository Structure

```
├── notebooks/          # Sample notebooks for analysis & visualization
├── src/
│   ├── mnist_activation_logger.py  # Extract activations from neural networks
│   ├── sae_trainer.py              # Train SAE on extracted activations
│   ├── infer.py                    # Extract SAE outputs and compute metrics
│   ├── models/
│   │   └── oracle.py               # Neural network models for activation extraction
│   ├── metrics/
│   │   ├── metrics_calc.py         # Compute concept recovery metrics
│   │   ├── logistic_regression_classifier.py  # Probe classifiers
│   │   └── plotter.py              # Visualization tools
│   └── script_utils/
│       ├── loader.py               # Data loading utilities
│       ├── loss.py                 # Loss functions
│       ├── model_trainer.py        # Training utilities
│       ├── train_util.py           # Training helpers
│       └── utils.py                # General utilities
├── README.md
└── requirements.txt
```

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.9+
- CUDA-capable GPU (recommended)

### Setup
```bash
# Clone repository
git clone https://github.com/yourusername/sae-concept-discarding.git
cd sae-concept-discarding

# Create virtual environment
python -venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Extract Neural Network Activations
```bash
# Extract activations from trained neural networks on MNIST
python src/mnist_activation_logger.py --model_path models/trained_model.pth --output_dir data/activations/
```

### 2. Train Sparse Autoencoder
```bash
# Train SAE on extracted activations
python src/sae_trainer.py --activation_dir data/activations/ --output_dir models/sae/
```

### 3. Analyze Concept Recovery
```bash
# Extract SAE outputs and compute concept recovery metrics
python src/infer.py --sae_path models/sae/model.pth --activation_dir data/activations/ --output_dir results/
```

### 4. Visualize Results
```bash
# Generate analysis reports and visualizations
python src/metrics/plotter.py --results_dir results/
```

## Workflow

1. **Activation Extraction**: Use `mnist_activation_logger.py` to extract intermediate activations from trained neural networks processing MNIST data
2. **SAE Training**: Use `sae_trainer.py` to train sparse autoencoders on the extracted activations
3. **Concept Recovery Analysis**: Use `infer.py` to:
   - Extract SAE outputs for test data
   - Train linear probes on SAE activations
   - Compute concept recovery metrics
   - Analyze which concepts are retained vs discarded

## Key Findings

### Current Results
- [Document your main findings here as you discover them]
- **Concept Recovery Patterns**: [Summary of which concepts are typically retained/discarded]
- **Training Dynamics**: [Key observations about how concept discarding emerges during training]
- **Hyperparameter Effects**: [Which training settings influence concept discarding]

### Validation Status
- [ ] Baseline concept recovery established
- [ ] Cross-SAE comparison completed
- [ ] Training dynamics analyzed
- [ ] Statistical significance validated
- [ ‎] Generalization testing performed

## Configuration

### SAE Training Parameters
Key hyperparameters under investigation:
- **Sparsity penalty**: λ ∈ [0.001, 0.1, 1.0]
- **Hidden dimensions**: [128, 256, 512, 1024]
- **Learning rate**: [1e-4, 1e-3, 1e-2]
- **Training epochs**: [50, 100, 200]
- **Batch size**: [64, 128, 256]

### Evaluation Settings
- **Probe training**: 80/20 train/test split, stratified sampling
- **Cross-validation**: 5-fold CV for robustness
- **Metrics**: AUC (primary), accuracy, F1-score
- **Selectivity threshold**: AUC > 0.85 for "selective" neurons

## Experiments

### Completed
1. **Baseline Study**: Standard SAE training with concept probing
2. **Hyperparameter Sweep**: Systematic variation of training parameters
3. **[Add your completed experiments]**

### Planned
1. **Intervention Study**: Manual concept ablation during training
2. **Temporal Analysis**: Fine-grained tracking of concept emergence/disappearance
3. **Generalization Testing**: Evaluation on modified MNIST variants
4. **[Add your planned experiments]**

## Results


