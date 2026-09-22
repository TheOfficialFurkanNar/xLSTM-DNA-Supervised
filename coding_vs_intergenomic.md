# Coding vs Intergenomic DNA Classification - Ablation Study

## Overview

This experiment conducts an ablation study on xLSTM components for DNA sequence classification. The task is to distinguish between coding and intergenomic DNA sequences using the Genomic Benchmarks dataset.

## Dataset

- **Source**: `katarinagresova/Genomic_Benchmarks_demo_coding_vs_intergenomic_seqs` (Hugging Face)
- **Task**: Binary classification (coding vs intergenomic sequences)
- **Samples**: 20,000 training, 5,000 test
- **Sequence length**: 100 bp (padded/truncated)
- **Encoding**: A=0, C=1, G=2, T=3, N=4

## xLSTM Components Being Tested

### 1. Exponential Gating
- Uses `exp()` instead of `sigmoid()` for input/forget gates
- Enables gradient flow through long sequences
- **Toggle**: `--no-exp-gating`

### 2. Stabilizer Gate
- Prevents numerical overflow in exponential operations
- Computes: $\text{stab} = \max(\log_{\text{forget}} + \text{stab}_{\text{prev}}, \log_{\text{input}})$
- **Toggle**: `--no-stabilizer`

### 3. Normalizer
- Tracks cumulative gate values for stable normalization
- Output: $h = o \odot (C / (n + \epsilon))$
- **Toggle**: `--no-normalizer`

### 4. Memory Mixing
- Adds hidden state to gate computations (recurrent connections)
- Without: gates only depend on input
- **Toggle**: `--no-memory-mixing` (sLSTM only)

### 5. Layer Type
- **sLSTM**: Scalar LSTM with exponential gates
- **mLSTM**: Matrix LSTM with attention-like QKV mechanism
- **alternating**: Alternating sLSTM and mLSTM layers
- **Toggle**: `--layer-type {slstm,mlstm,alternating}`

## Parameter Scaling

| Scale   | Embedding | Hidden | Layers | Approx Params |
|---------|-----------|--------|--------|---------------|
| tiny    | 32        | 64     | 1      | ~20K          |
| small   | 32        | 128    | 2      | ~50K          |
| medium  | 64        | 128    | 2      | ~100K         |
| large   | 64        | 256    | 2      | ~250K         |
| xlarge  | 128       | 256    | 2      | ~330K         |

## Running Experiments

### Basic Usage
```bash
python train_coding_vs_intergenomic.py --scale medium
```

### Component Ablation
```bash
# Disable exponential gating
python train_coding_vs_intergenomic.py --no-exp-gating --scale medium

# Disable stabilizer
python train_coding_vs_intergenomic.py --no-stabilizer --scale medium

# Disable normalizer
python train_coding_vs_intergenomic.py --no-normalizer --scale medium

# Disable memory mixing
python train_coding_vs_intergenomic.py --no-memory-mixing --scale medium
```

### Layer Type Comparison
```bash
python train_coding_vs_intergenomic.py --layer-type slstm --scale medium
python train_coding_vs_intergenomic.py --layer-type mlstm --scale medium
python train_coding_vs_intergenomic.py --layer-type alternating --scale medium
```

### Parameter Scaling
```bash
python train_coding_vs_intergenomic.py --scale tiny
python train_coding_vs_intergenomic.py --scale small
python train_coding_vs_intergenomic.py --scale medium
python train_coding_vs_intergenomic.py --scale large
python train_coding_vs_intergenomic.py --scale xlarge
```

### Custom Training Parameters
```bash
python train_coding_vs_intergenomic.py --epochs 10 --lr 5e-4 --batch-size 64
```

## Output

### Console Output
- Configuration summary (components, scale, layer type)
- Training progress with loss and accuracy per epoch
- Test evaluation at epochs 5, 10, 15, 20
- **Ablation Results Summary** with:
  - Parameter count
  - Configuration flags
  - Final test accuracy and F1 score

### Plots
- `accuracy.jpg`: Train vs test accuracy over epochs
- `f1_score.jpg`: Train vs test F1 score over epochs
- `loss.jpg`: Train vs test loss over epochs

## Experiment Design

### Phase 1: Baseline + Individual Component Ablation
Test each component separately at medium scale to identify impact:
1. Full baseline (all components enabled)
2. No exponential gating
3. No stabilizer
4. No normalizer
5. No memory mixing

### Phase 2: Layer Type Comparison
Compare sLSTM vs mLSTM vs alternating at medium scale.

### Phase 3: Parameter Scaling
Test if component importance changes with model size:
- Compare baseline vs most impactful ablation at tiny, medium, large scales

## Metrics

- **Accuracy**: Classification accuracy
- **F1 Score**: Weighted F1 score (handles class imbalance)
- **Loss**: Cross-entropy loss
- **Parameters**: Total trainable parameters

## Implementation Details

### Model Architecture
- Embedding layer (vocab_size=5 → embedding_dim)
- LSTM layers (alternating sLSTM/mLSTM by default)
- Mean pooling over sequence
- Dropout (0.3)
- Classification head (hidden_size → num_classes=2)

### Training
- Optimizer: Adam (lr=1e-3, weight_decay=1e-4)
- Scheduler: Cosine annealing
- Gradient clipping: max_norm=1.0
- Epochs: 20
- Batch size: 128

### Notes
- Plots are saved to current directory
- GPU memory constrained to 70% to prevent overheating 
- Small delay (0.01s) per batch to reduce GPU heat
