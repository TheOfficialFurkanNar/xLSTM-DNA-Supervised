# xLSTM DNA Classification - Ablation Study

The repository implements an ablation study on xLSTM architecture components for DNA sequence classification. The goal is to identify which xLSTM ingredients matter most at different parameter scales (6.7k-166k parameters).

## Overview

xLSTM combines scalar LSTMs (sLSTM) and matrix LSTMs (mLSTM) in an alternating architecture. This study systematically disables individual components to measure their impact on DNA classification performance on the Genomic Benchmarks dataset (coding vs intergenomic sequences).

## Architectural Decisions (model.py)

### Component Toggling Strategy

**Decision**: Use `None` flags instead of boolean parameters to disable components.

**Rationale**:
- Cleaner modularity - disabled components are literally absent from the computation graph
- No redundant conditional checks in the forward pass
- Easier to verify that a component is truly not contributing
- More explicit about what's being ablated

### sLSTM (Scalar LSTM)

**Components**:
1. **Exponential Gating**: Uses `exp()` instead of `sigmoid()` for input/forget gates
   - Enables gradient flow through long sequences
   - Without: Falls back to standard sigmoid gates

2. **Stabilizer Gate**: Prevents numerical overflow in exponential operations
   - Computes: Computes: $\text{stab} = \max\left(\log(\text{forget}) + \text{stab}_{\text{prev}}, \log(\text{input})\right)$
   - Without: Unstabilized exponential gates (may overflow)

3. **Normalizer**: Tracks cumulative gate values for stable normalization
   - Output: $h = o \odot (C / (n + \epsilon))$
   - Without: Uses raw cell candidate instead of normalized cell state

4. **Memory Mixing**: Adds hidden state to gate computations
   - Gates depend on both input and previous hidden state
   - Without: Gates only depend on input (no recurrent connection in gates)

**State Tuple**: `(hidden_state, cell_state, stabilizer, normalizer)`
- Stabilizer and normalizer are `None` when disabled

### mLSTM (Matrix LSTM)

**Components**:
1. **Exponential Gating**: Same as sLSTM but for scalar input/forget gates
2. **Stabilizer Gate**: Same as sLSTM
3. **Normalizer**: Same as sLSTM but with query-key dot product denominator

**Note**: mLSTM does not have memory mixing - gates are always input-only (scalar gates control matrix operations)

**State Tuple**: `(cell_state, normalizer, stabilizer)`
- Cell state is a matrix: `(batch, hidden_size, hidden_size)`
- Normalizer and stabilizer are `None` when disabled

### xLSTM Container

**Layer Stacking**:
- First layer takes embeddings (`embedding_dim`)
- Subsequent layers take hidden state (`hidden_size`)
- Alternating pattern: sLSTM → mLSTM → sLSTM → mLSTM ...

**Classification Head**:
- Mean pooling over sequence length
- Linear projection to `num_classes`

## Ablation Study Design

### Research Question

Which xLSTM components are most impactful for DNA classification at different parameter scales?

### Planned Experiments

#### Phase 1: Baseline + Individual Component Ablation

**Objective**: Identify which components have the largest impact at medium scale (~100k params)

**Experiments**:
1. Full baseline (all components enabled)
2. No exponential gating
3. No stabilizer
4. No normalizer
5. No memory mixing

**Metrics**: Test accuracy, F1 score at best epoch (based on validation loss)

#### Phase 2: Layer Type Comparison

**Objective**: Compare sLSTM vs mLSTM vs alternating architecture

**Experiments**:
1. sLSTM only
2. mLSTM only
3. Alternating (default)

**Metrics**: Test accuracy, F1 score at best epoch (based on validation loss)

#### Phase 3: Parameter Scaling

**Objective**: Test if component importance changes with model size

**Experiments**: Compare baseline vs most impactful ablation at different scales

**Scales**:
- Tiny: ~6.7k parameters (embedding=16, hidden=32, layers=1)
- Small: ~38k parameters (embedding=16, hidden=64, layers=2)
- Medium: ~42k parameters (embedding=32, hidden=64, layers=2)
- Large: ~150k parameters (embedding=32, hidden=128, layers=2)
- XLarge: ~166k parameters (embedding=64, hidden=128, layers=2)

**Metrics**: Test accuracy, F1 score at best epoch (based on validation loss)

## Ablation Decisions

### Why These Components?

The selected components represent the key innovations of xLSTM over standard LSTM:

1. **Exponential gating**: Core departure from sigmoid gates
2. **Stabilizer**: Enables safe use of exponential gates
3. **Normalizer**: Provides stable normalization for memory
4. **Memory mixing**: Adds recurrent connections to gates (sLSTM-specific)

These are the minimal set of ablations that capture the architectural differences.

### Why None Flags?

- **Explicit**: A disabled component is literally absent from the model
- **Verifiable**: No hidden computation paths
- **Modular**: Easy to add/remove components without affecting other logic
- **Performance**: No conditional checks in the forward pass for disabled components

### Why Parameter Scaling?

xLSTM was designed to scale well. Testing across 6.7k-166k parameters reveals:
- Whether components are equally important at different scales
- If small models rely more heavily on certain components
- If large models can compensate for missing components

## Training Configuration

### Hyperparameters

- **Optimizer**: AdamW (lr=5e-4, weight_decay=1e-2)
- **Scheduler**: Cosine annealing (T_max=epochs)
- **Gradient clipping**: max_norm=1.0
- **Dropout**: 0.3 (inter-layer + pooled)
- **Input dropout**: 0.1 (embedding feature dropout)
- **Label smoothing**: 0.1
- **Batch size**: 128
- **Epochs**: 30 (default, with early stopping, patience=6)
- **Sequence length**: 200 bp (sequences are ~500bp)

### Data Augmentation

- **Reverse-complement**: 50% probability during training
- **Random mutations**: 1% per-base substitution rate during training
- **Random cropping**: During training; deterministic center crop during evaluation
- **Deduplication**: Removes duplicate sequences from training set
- **Decontamination**: Removes training sequences that appear in test set

### Evaluation

- **Validation split**: 10% of training data (stratified)
- **Early stopping**: Based on validation loss with patience=6
- **Metrics**: Accuracy, F1 score (weighted), Cross-entropy loss
- **Plots**: `{tag}_acc.jpg`, `{tag}_f1.jpg`, `{tag}_loss.jpg`

### Logging

- **Format**: JSONL (one JSON object per line)
- **Filename**: `train_log_{config}.jsonl`
- **Contents**:
  - Config: All experiment parameters and component flags
  - Step: Per-batch loss, accuracy, samples processed
  - Epoch: End-of-epoch train metrics
  - Final: Best epoch validation metrics and test metrics

## Usage

### Ablation Studies

Disable specific components:

```bash
# Baseline (all components enabled)
python xLSTM-DNA-Supervised/train_coding_vs_intergenomic.py --scale small --epochs 5 --patience 3 --out-dir xLSTM-DNA-Supervised/results

# Disable exponential gating
python xLSTM-DNA-Supervised/train_coding_vs_intergenomic.py --scale small --epochs 5 --patience 3 --out-dir xLSTM-DNA-Supervised/results --no-exp-gating

# Disable stabilizer
python xLSTM-DNA-Supervised/train_coding_vs_intergenomic.py --scale small --epochs 5 --patience 3 --out-dir xLSTM-DNA-Supervised/results --no-stabilizer

# Disable normalizer
python xLSTM-DNA-Supervised/train_coding_vs_intergenomic.py --scale small --epochs 5 --patience 3 --out-dir xLSTM-DNA-Supervised/results --no-normalizer

# Disable memory mixing
python xLSTM-DNA-Supervised/train_coding_vs_intergenomic.py --scale small --epochs 5 --patience 3 --out-dir xLSTM-DNA-Supervised/results --no-memory-mixing
```

### Layer Type Comparison

Compare different layer architectures:

```bash
# sLSTM only
python xLSTM-DNA-Supervised/train_coding_vs_intergenomic.py --scale small --epochs 5 --patience 3 --out-dir xLSTM-DNA-Supervised/results --layer-type slstm

# mLSTM only
python xLSTM-DNA-Supervised/train_coding_vs_intergenomic.py --scale small --epochs 5 --patience 3 --out-dir xLSTM-DNA-Supervised/results --layer-type mlstm

# Alternating (default)
python xLSTM-DNA-Supervised/train_coding_vs_intergenomic.py --scale small --epochs 5 --patience 3 --out-dir xLSTM-DNA-Supervised/results --layer-type alternating
```

### Parameter Scaling

Test different model sizes:

```bash
# Tiny (~6.7k params)
python xLSTM-DNA-Supervised/train_coding_vs_intergenomic.py --scale tiny --epochs 5 --patience 3 --out-dir xLSTM-DNA-Supervised/results

# Small (~38k params, default)
python xLSTM-DNA-Supervised/train_coding_vs_intergenomic.py --scale small --epochs 5 --patience 3 --out-dir xLSTM-DNA-Supervised/results

# Medium (~42k params)
python xLSTM-DNA-Supervised/train_coding_vs_intergenomic.py --scale medium --epochs 5 --patience 3 --out-dir xLSTM-DNA-Supervised/results

# Large (~150k params)
python xLSTM-DNA-Supervised/train_coding_vs_intergenomic.py --scale large --epochs 5 --patience 3 --out-dir xLSTM-DNA-Supervised/results

# XLarge (~166k params)
python xLSTM-DNA-Supervised/train_coding_vs_intergenomic.py --scale xlarge --epochs 5 --patience 3 --out-dir xLSTM-DNA-Supervised/results
```

## References

- xLSTM paper: [Extended Long Short-Term Memory](https://arxiv.org/abs/2405.04517)
- Genomic Benchmarks: [Genomic Benchmarks Dataset](https://huggingface.co/datasets/katarinagresova/Genomic_Benchmarks_demo_coding_vs_intergenomic_seqs)
- BioxLSTM paper: [BioxLSTM](https://arxiv.org/abs/2411.04165v1)
- xLSTM 7B paper: [xLSTM 7B](https://arxiv.org/abs/2503.13427)