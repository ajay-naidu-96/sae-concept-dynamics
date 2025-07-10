# Sparse Autoencoder Concept Dynamics

## Overview

This project investigates the dynamics of concept retention and discarding in Sparse Autoencoders (SAEs) using toy datasets. We explore experimental settings where SAEs learn to encode, preserve, or eliminate representations through systematic analysis of neuron-level activations and concept recovery capabilities. This is a hypothesis-driven research project focused on understanding when and why certain concepts survive SAE training while others are filtered out.

## Research Questions

- Can trained SAEs selectively discard certain concepts while retaining others?
- What are the training dynamics that lead to concept discarding vs retention?
- Are there predictable patterns in which concepts get discarded together?

## Methodology

### Experimental Setup
- **Dataset**: MNIST handwritten digits (0-9) 
- **Pipeline**: Neural network → activation extraction → SAE training → concept probing
- **Evaluation**: Linear probes trained on SAE neuron activations for concept classification
- **Metrics**: AUC, accuracy, precision/recall for concept recovery analysis

### Key Components
1. **Activation Extraction**: Extract intermediate activations from neural networks on MNIST
2. **SAE Training**: Train sparse autoencoders on extracted activations
3. **Concept Probing**: Logistic regression classifiers on SAE outputs to measure known concept recovery
4. **Cross-SAE Comparison**: Systematic analysis across different training configurations

## Repository Structure

```
├── notebooks/          # Sample notebooks for analysis & visualization
├── src/
│   ├── mnist_activation_logger.py  # Extract activations from neural networks
│   ├── sae_trainer.py              # Train SAE on extracted activations
│   ├── infer.py                    # Extract SAE outputs and compute metrics
│   ├── models/
│   │   └── oracle.py               # Sample neural network model for activation extraction
│   ├── metrics/
│   │   ├── metrics_calc.py         # experimental metrics
│   │   ├── logistic_regression_classifier.py  # Probe classifiers
│   │   └── plotter.py              # experimental
│   └── script_utils/
│       ├── loader.py               # Data Loaders
│       ├── loss.py                 # Loss functions
│       ├── model_trainer.py      
│       ├── train_util.py           # Training helpers
│       └── utils.py                
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
git clone git@github.com:ajay-naidu-96/sae-concept-dynamics.git
cd sae-concept-dynamics

# Create virtual environment
python -venv venv
source venv/bin/activate 

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

Needs a readme update, the scripts are still experimental!

### 1. Extract Neural Network Activations
```bash
# Extract activations from trained neural networks on MNIST
CUDA_VISIBLE_DEVICES=0 python3 src/mnist_activation_logger.py --log_dir data/activations/
```

### 2. Train Sparse Autoencoder
```bash
# Train SAE on extracted activations
CUDA_VISIBLE_DEVICES=0 python3 src/sae_trainer.py --log_dir data/activations/ --sae_type vanilla
```

### 3. Analyze Concept Recovery
```bash
# Extract SAE outputs and compute concept recovery metrics
CUDA_VISIBLE_DEVICES=0 python3 src/infer.py --log_dir ./logs_vanilla_sae/
```

### 4. Visualize Results


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
- **Concept Recovery Patterns**: 
- **Training Dynamics**: 
- **Hyperparameter Effects**: 

### Validation Status
- [ ] Baseline concept recovery 
- [ ] Cross-SAE comparison 
- [ ] Training dynamics 

## Configuration

### SAE Training Parameters
Key hyperparameters under investigation:
- **Sparsity penalty**: 
  - Vanilla SAE: λ ∈ 20 logarithmically spaced values between 10^-5 and 10^0
  - TopK SAE: k ∈ [0.02, 0.05, 0.1, 0.2, 0.5, 0.75, 0.95]
- **Hidden dimensions**: [256, 512]
- **Learning rate**: [1e-3]
- **Training epochs**: [30]
- **Batch size**: [1024]

### Evaluation Settings
- **Probe training**: 80/20 train/test split, stratified sampling
- **Cross-validation**: 5-fold CV for robustness
- **Metrics**: AUC score for classification
- **Selectivity threshold**: AUC > 0.85 for "selective" neurons

## Experiments

### Completed
1. **Baseline Study**: Standard SAE training with concept probing
2. **Hyperparameter Sweep**: Systematic variation of training parameters

### Planned
1. **Trained vs Random**: Probing randomly initialized sae vs trained sae
2. **Concept Abalation**: Manual concept ablation during training

## Results

[Needs to be updated]

