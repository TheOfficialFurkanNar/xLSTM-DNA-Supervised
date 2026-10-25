# train_coding_vs_intergenomic.py
import os
# Use short path to avoid Windows MAX_PATH limit (260 chars)
os.environ['HF_HOME'] = r'C:\hfc'
os.environ['HF_DATASETS_CACHE'] = r'C:\hfc\datasets'
os.environ['HF_HUB_CACHE'] = r'C:\hfc\hub'

import argparse
import json
import random
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from model import sLSTM, mLSTM


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def feature_mask(batch_size, dim, p, device, training):
    """Dropout mask sampled ONCE per sequence and reused at every timestep
    (variational / locked dropout). Returns None when inactive.
    Using Bernoulli Distribution for simple PMF"""
    if not training or p <= 0.0:
        return None
    keep = 1.0 - p
    return torch.bernoulli(torch.full((batch_size, dim), keep, device=device)) / keep


class xLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=2, num_classes=2,
                 dropout=0.3, input_dropout=0.1,
                 layer_type='alternating', use_exp_gating=True, use_stabilizer=True,
                 use_normalizer=True, use_memory_mixing=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout_p = dropout
        self.input_dropout_p = input_dropout
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)  # used on the pooled vector only

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            input_dim = embedding_dim if i == 0 else hidden_size

            if layer_type == 'slstm':
                self.layers.append(sLSTM(input_dim, hidden_size, use_exp_gating, use_stabilizer, use_normalizer, use_memory_mixing))
            elif layer_type == 'mlstm':
                self.layers.append(mLSTM(input_dim, hidden_size, use_exp_gating, use_stabilizer, use_normalizer))
            else:  # alternating
                if i % 2 == 0:
                    self.layers.append(sLSTM(input_dim, hidden_size, use_exp_gating, use_stabilizer, use_normalizer, use_memory_mixing))
                else:
                    self.layers.append(mLSTM(input_dim, hidden_size, use_exp_gating, use_stabilizer, use_normalizer))

        self.output_proj = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)  # (B, T, E)
        B, T, E = embedded.shape
        dev = embedded.device
        n_layers = len(self.layers)

        # One mask per sequence (not per timestep) -> doesn't scramble the recurrent signal.
        in_mask = feature_mask(B, E, self.input_dropout_p, dev, self.training)
        inter_masks = [feature_mask(B, self.hidden_size, self.dropout_p, dev, self.training)
                       for _ in range(n_layers - 1)]

        states = [None] * n_layers
        hidden_sum = 0
        for t in range(T):
            x_t = embedded[:, t, :]
            if in_mask is not None:
                x_t = x_t * in_mask
            for i, layer in enumerate(self.layers):
                if isinstance(layer, sLSTM):
                    x_t, cell, stab, norm = layer(x_t, states[i])
                    states[i] = (x_t, cell, stab, norm)   # recurrent state stays unmasked
                else:  # mLSTM
                    x_t, cell, norm, stab = layer(x_t, states[i])
                    states[i] = (cell, norm, stab)
                # mask only the feed-forward path to the next layer
                if i < n_layers - 1 and inter_masks[i] is not None:
                    x_t = x_t * inter_masks[i]
            hidden_sum = hidden_sum + x_t

        pooled = self.dropout(hidden_sum / T)  # mean pooling
        return self.output_proj(pooled)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
ENC = {'A': 0, 'C': 1, 'G': 2, 'T': 3}          # everything else -> 4 (N / pad)
COMP = np.array([3, 2, 1, 0, 4], dtype=np.uint8)  # A<->T, C<->G, N->N


def encode(seq):
    return np.fromiter((ENC.get(b, 4) for b in seq.upper()), dtype=np.uint8, count=len(seq))


class DNADataset(Dataset):
    """Pre-encodes sequences once. In train mode: random crop, random reverse
    complement, random base substitutions. In eval mode: deterministic center crop."""
    def __init__(self, sequences, labels, seq_length, train=False, rc_prob=0.5, mut_rate=0.0, seed=0):
        self.encoded = [encode(s) for s in sequences]
        self.labels = np.asarray(labels, dtype=np.int64)
        self.seq_length = seq_length
        self.train = train
        self.rc_prob = rc_prob
        self.mut_rate = mut_rate
        self.rng = np.random.default_rng(seed)  # use num_workers=0 to keep this deterministic

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        arr = self.encoded[idx]
        L = self.seq_length
        n = len(arr)

        if n >= L:
            start = int(self.rng.integers(0, n - L + 1)) if self.train else (n - L) // 2
            arr = arr[start:start + L]
        else:
            arr = np.concatenate([arr, np.full(L - n, 4, dtype=np.uint8)])

        if self.train:
            if self.rc_prob > 0 and self.rng.random() < self.rc_prob:
                arr = COMP[arr[::-1]]
            if self.mut_rate > 0:
                mask = self.rng.random(L) < self.mut_rate
                if mask.any():
                    arr = arr.copy()
                    arr[mask] = self.rng.integers(0, 4, size=int(mask.sum()), dtype=np.uint8)

        return torch.from_numpy(arr.astype(np.int64)), torch.tensor(self.labels[idx])


def dedupe_and_decontaminate(seqs, labels, test_seqs):
    """Drop duplicate train sequences and any train sequence that also appears in test."""
    test_set = set(test_seqs)
    seen, out_s, out_l = set(), [], []
    for s, l in zip(seqs, labels):
        if s in seen or s in test_set:
            continue
        seen.add(s)
        out_s.append(s)
        out_l.append(l)
    return out_s, out_l


# ----------------------------------------------------------------------------
# Train / eval
# ----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    preds, labels = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)
        pred = logits.argmax(dim=-1)
        total_loss += loss.item() * len(X)
        correct += (pred == y).sum().item()
        total += len(X)
        preds.extend(pred.cpu().numpy())
        labels.extend(y.cpu().numpy())
    return total_loss / total, correct / total, f1_score(labels, preds, average='weighted')


def train_model(model, train_loader, val_loader, device, args, log_file=None, config=None):
    # Weight decay only on matrices (not biases / gate vectors)
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        (decay if p.ndim >= 2 else no_decay).append(p)
    optimizer = optim.AdamW([
        {'params': decay, 'weight_decay': args.weight_decay},
        {'params': no_decay, 'weight_decay': 0.0},
    ], lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    train_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    eval_criterion = nn.CrossEntropyLoss()  # plain CE so train/val/test losses are comparable

    history = {k: [] for k in ['train_loss', 'train_acc', 'train_f1', 'val_loss', 'val_acc', 'val_f1']}
    best_val, best_epoch, best_state, bad_epochs = float('inf'), 0, None, 0

    log_fp = open(log_file, 'w') if log_file else None
    if log_fp and config:
        log_fp.write(json.dumps({'type': 'config', **config}) + '\n')

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for step, (X_batch, y_batch) in enumerate(pbar):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = train_criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            with torch.no_grad():
                plain_loss = eval_criterion(logits.detach(), y_batch).item()
                pred = logits.argmax(dim=-1)
            total_loss += plain_loss * len(X_batch)
            correct += (pred == y_batch).sum().item()
            total += len(X_batch)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

            step_acc = correct / total
            pbar.set_postfix({'loss': f"{plain_loss:.4f}", 'acc': f"{step_acc:.3f}"})
            if log_fp:
                log_fp.write(json.dumps({'type': 'step', 'epoch': epoch, 'step': step, 'loss': plain_loss,
                                         'accuracy': step_acc, 'samples_processed': total}) + '\n')
            if args.step_sleep > 0:
                time.sleep(args.step_sleep)  # reduce GPU heat

        scheduler.step()

        # NOTE: train metrics are measured with dropout + augmentation ON, so they are
        # pessimistic. Val loss is the signal to watch for overfitting.
        train_loss = total_loss / total
        train_acc = correct / total
        train_f1 = f1_score(all_labels, all_preds, average='weighted')
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, device, eval_criterion)

        for k, v in zip(history.keys(), [train_loss, train_acc, train_f1, val_loss, val_acc, val_f1]):
            history[k].append(v)

        if log_fp:
            log_fp.write(json.dumps({'type': 'epoch', 'epoch': epoch,
                                     'train_loss': train_loss, 'train_accuracy': train_acc, 'train_f1': train_f1,
                                     'val_loss': val_loss, 'val_accuracy': val_acc, 'val_f1': val_f1}) + '\n')

        print(f"Epoch {epoch:2d}/{args.epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} F1: {train_f1:.3f} "
              f"| Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} F1: {val_f1:.3f}")

        # Early stopping on validation loss, keep best weights
        if val_loss < best_val - 1e-4:
            best_val, best_epoch, bad_epochs = val_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch} (best epoch: {best_epoch}, val loss {best_val:.4f})")
                break

    if log_fp:
        log_fp.close()
    return history, best_epoch, best_state


def plot_metrics(history, best_epoch, tag, output_dir='.'):
    os.makedirs(output_dir, exist_ok=True)
    epochs = list(range(1, len(history['train_loss']) + 1))
    for name, title in [('acc', 'Accuracy'), ('f1', 'F1 Score'), ('loss', 'Loss')]:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, history[f'train_{name}'], 'o-', label=f'Train {title} (aug+dropout on)')
        plt.plot(epochs, history[f'val_{name}'], 's-', label=f'Val {title}')
        if best_epoch:
            plt.axvline(best_epoch, color='gray', linestyle='--', label=f'Best epoch ({best_epoch})')
        plt.xlabel('Epoch')
        plt.ylabel(title)
        plt.title(f'Train vs Val {title}')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'{tag}_{name}.jpg'), format='jpg', dpi=150, bbox_inches='tight')
        plt.close()
    print(f"Plots saved to {output_dir}/")


"""
Argument parsers for Terminal

Component toggles:
  --no-exp-gating: Disable exponential gating (default: enabled)
  --no-stabilizer: Disable stabilizer gate (default: enabled)
  --no-normalizer: Disable normalizer (default: enabled)
  --no-memory-mixing: Disable memory mixing (default: enabled)
  --layer-type: Layer architecture - slstm, mlstm, or alternating (default: alternating)
  --scale: Model size - tiny, small, medium, large, or xlarge (default: small)

Training:
  --epochs: Number of training epochs (default: 30)
  --lr: Learning rate (default: 5e-4)
  --batch-size: Batch size (default: 128)
  --seed: Random seed for model initialization, shuffling, and augmentation (default: 42)
  --data-seed: Seed for train/val split (keep fixed across ablation arms) (default: 0)
  --patience: Early-stopping patience in epochs on validation loss (default: 6)
  --step-sleep: Sleep time between steps to reduce GPU heat (default: 0.01)
  --out-dir: Output directory for logs and plots (default: .)

Regularization (keep IDENTICAL across ablation arms):
  --dropout: Inter-layer + pooled dropout (default: 0.3)
  --input-dropout: Embedding feature dropout (default: 0.1)
  --weight-decay: Weight decay for AdamW optimizer (default: 1e-2)
  --label-smoothing: Label smoothing factor (default: 0.1)

Data / augmentation:
  --seq-length: Window length for sequences (sequences are ~500bp) (default: 200)
  --val-frac: Fraction of training data for validation (default: 0.1)
  --max-train: Randomly subsample train set (0 = use all) (default: 0)
  --rc-prob: Reverse-complement augmentation probability (default: 0.5)
  --mut-rate: Per-base random substitution rate for training (default: 0.01)
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='xLSTM DNA Classification with Ablation Studies')

    # Component toggles
    parser.add_argument('--no-exp-gating', action='store_true', help='Disable exponential gating')
    parser.add_argument('--no-stabilizer', action='store_true', help='Disable stabilizer gate')
    parser.add_argument('--no-normalizer', action='store_true', help='Disable normalizer')
    parser.add_argument('--no-memory-mixing', action='store_true', help='Disable memory mixing')
    parser.add_argument('--layer-type', type=str, default='alternating', choices=['slstm', 'mlstm', 'alternating'])
    parser.add_argument('--scale', type=str, default='small', choices=['tiny', 'small', 'medium', 'large', 'xlarge'],
                        help='tiny~5k, small~15k, medium~30k, large~60k, xlarge~120k params')

    # Training
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42, help='Model init / shuffling / augmentation seed')
    parser.add_argument('--data-seed', type=int, default=0, help='Seed for the train/val split (keep fixed across ablation arms)')
    parser.add_argument('--patience', type=int, default=6, help='Early-stopping patience (epochs) on val loss')
    parser.add_argument('--step-sleep', type=float, default=0.01)
    parser.add_argument('--out-dir', type=str, default='.')

    # Regularization (keep IDENTICAL across ablation arms)
    parser.add_argument('--dropout', type=float, default=0.3, help='Inter-layer + pooled dropout')
    parser.add_argument('--input-dropout', type=float, default=0.1, help='Embedding feature dropout')
    parser.add_argument('--weight-decay', type=float, default=1e-2)
    parser.add_argument('--label-smoothing', type=float, default=0.1)

    # Data / augmentation
    parser.add_argument('--seq-length', type=int, default=200, help='Window length (sequences are ~500bp)')
    parser.add_argument('--val-frac', type=float, default=0.1)
    parser.add_argument('--max-train', type=int, default=0, help='Randomly subsample train set (0 = use all)')
    parser.add_argument('--rc-prob', type=float, default=0.5, help='Reverse-complement augmentation probability')
    parser.add_argument('--mut-rate', type=float, default=0.01, help='Per-base random substitution rate (train only)')

    args = parser.parse_args()
    set_seed(args.seed)

    use_exp_gating = not args.no_exp_gating
    use_stabilizer = not args.no_stabilizer
    use_normalizer = not args.no_normalizer
    use_memory_mixing = not args.no_memory_mixing


    """
    Parameter scale configurations
    tiny (16, 32, 1): 6,674 parameters
    small (16, 64, 2): 38,228 parameters
    medium (32, 64, 2): 42,404 parameters
    large (32, 128, 2): 149,668 parameters
    xlarge (64, 128, 2): 165,956 parameters
    """
    scale_configs = {
        'tiny': (16, 32, 1),
        'small': (16, 64, 2),
        'medium': (32, 64, 2),
        'large': (32, 128, 2),
        'xlarge': (64, 128, 2)
    }
    embedding_dim, hidden_size, num_layers = scale_configs[args.scale]
#Use GPU if available use CPU otherwise
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.7) # Set GPU memory fraction to prevent OOMs.
    print(f"Using device: {device}")
    print(f"Configuration: exp_gating={use_exp_gating}, stabilizer={use_stabilizer}, normalizer={use_normalizer}, memory_mixing={use_memory_mixing}")
    print(f"Layer type: {args.layer_type}, Scale: {args.scale} (emb={embedding_dim}, hid={hidden_size}, layers={num_layers})")

    # ---- Data: full (non-streamed) dataset so we don't depend on file ordering ----
    print("Loading dataset from Hugging Face...")
    dataset = load_dataset("katarinagresova/Genomic_Benchmarks_demo_coding_vs_intergenomic_seqs")

    raw_train_seqs, raw_train_labels = list(dataset['train']['seq']), list(dataset['train']['label'])
    test_seqs, test_labels = list(dataset['test']['seq']), list(dataset['test']['label'])
    print(f"Raw train: {len(raw_train_seqs)} {dict(Counter(raw_train_labels))} | "
          f"Raw test: {len(test_seqs)} {dict(Counter(test_labels))}")

    train_seqs, train_labels = dedupe_and_decontaminate(raw_train_seqs, raw_train_labels, test_seqs)
    print(f"After removing duplicates / train-test overlap: {len(train_seqs)} train sequences "
          f"(removed {len(raw_train_seqs) - len(train_seqs)})")

    if args.max_train and args.max_train < len(train_seqs):
        idx = np.random.default_rng(args.data_seed).choice(len(train_seqs), args.max_train, replace=False)
        train_seqs = [train_seqs[i] for i in idx]
        train_labels = [train_labels[i] for i in idx]

    # Stratified train/val split; val is used for early stopping, test is touched ONCE at the end
    tr_idx, va_idx = train_test_split(np.arange(len(train_seqs)), test_size=args.val_frac,
                                      stratify=train_labels, random_state=args.data_seed)
    val_seqs = [train_seqs[i] for i in va_idx]
    val_labels = [train_labels[i] for i in va_idx]
    train_seqs = [train_seqs[i] for i in tr_idx]
    train_labels = [train_labels[i] for i in tr_idx]
    print(f"Train: {len(train_seqs)} | Val: {len(val_seqs)} | Test: {len(test_seqs)}")

    train_dataset = DNADataset(train_seqs, train_labels, args.seq_length, train=True,
                               rc_prob=args.rc_prob, mut_rate=args.mut_rate, seed=args.seed)
    val_dataset = DNADataset(val_seqs, val_labels, args.seq_length, train=False)
    test_dataset = DNADataset(test_seqs, test_labels, args.seq_length, train=False)

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # ---- Model ----
    vocab_size, num_classes = 5, 2
    model = xLSTM(vocab_size, embedding_dim, hidden_size, num_layers, num_classes,
                  dropout=args.dropout, input_dropout=args.input_dropout,
                  layer_type=args.layer_type, use_exp_gating=use_exp_gating, use_stabilizer=use_stabilizer,
                  use_normalizer=use_normalizer, use_memory_mixing=use_memory_mixing).to(device)
    param_count = count_parameters(model)
    print(f"Model parameters: {param_count:,}")

    tag = (f"{args.scale}_{args.layer_type}_exp{use_exp_gating}_stab{use_stabilizer}"
           f"_norm{use_normalizer}_mix{use_memory_mixing}_seed{args.seed}")
    os.makedirs(args.out_dir, exist_ok=True)
    log_filename = os.path.join(args.out_dir, f"train_log_{tag}.jsonl")

    disabled_components = [name for name, on in [('exponential_gating', use_exp_gating), ('stabilizer', use_stabilizer),
                                                 ('normalizer', use_normalizer), ('memory_mixing', use_memory_mixing)] if not on]
    config = {
        'scale': args.scale,
        'layer_type': args.layer_type,
        'seed': args.seed,
        'data_seed': args.data_seed,
        'components': {'exponential_gating': use_exp_gating, 'stabilizer': use_stabilizer,
                       'normalizer': use_normalizer, 'memory_mixing': use_memory_mixing},
        'disabled_components': disabled_components if disabled_components else 'none',
        'model_config': {'embedding_dim': embedding_dim, 'hidden_size': hidden_size, 'num_layers': num_layers,
                         'vocab_size': vocab_size, 'num_classes': num_classes, 'total_parameters': param_count},
        'regularization': {'dropout': args.dropout, 'input_dropout': args.input_dropout,
                           'weight_decay': args.weight_decay, 'label_smoothing': args.label_smoothing,
                           'rc_prob': args.rc_prob, 'mut_rate': args.mut_rate, 'patience': args.patience},
        'training_config': {'epochs': args.epochs, 'lr': args.lr, 'batch_size': args.batch_size,
                            'seq_length': args.seq_length},
        'dataset': {'train_samples': len(train_seqs), 'val_samples': len(val_seqs), 'test_samples': len(test_seqs)}
    }

    # ---- Train (early stopping on val loss) ----
    history, best_epoch, best_state = train_model(model, train_loader, val_loader, device, args,
                                                  log_file=log_filename, config=config)
    print(f"Training log saved to {log_filename}")

    # ---- Single test evaluation using the best-val checkpoint ----
    model.load_state_dict(best_state)
    test_loss, test_acc, test_f1 = evaluate(model, test_loader, device, nn.CrossEntropyLoss())

    best_i = best_epoch - 1
    with open(log_filename, 'a') as f:
        f.write(json.dumps({'type': 'final', 'best_epoch': best_epoch,
                            'val_loss': history['val_loss'][best_i], 'val_accuracy': history['val_acc'][best_i],
                            'val_f1': history['val_f1'][best_i],
                            'test_loss': test_loss, 'test_accuracy': test_acc, 'test_f1': test_f1}) + '\n')

    print("\n" + "=" * 80)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 80)
    print(f"Parameters: {param_count:,}")
    print(f"Scale: {args.scale} | Layer Type: {args.layer_type} | Seed: {args.seed}")
    print(f"Exp Gating: {use_exp_gating} | Stabilizer: {use_stabilizer} | Normalizer: {use_normalizer} | Memory Mixing: {use_memory_mixing}")
    print(f"Best epoch (by val loss): {best_epoch}")
    print(f"Val  Acc: {history['val_acc'][best_i]:.4f} | Val  F1: {history['val_f1'][best_i]:.4f} | Val  Loss: {history['val_loss'][best_i]:.4f}")
    print(f"Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f} | Test Loss: {test_loss:.4f}")
    print("=" * 80)

    plot_metrics(history, best_epoch, tag, args.out_dir)