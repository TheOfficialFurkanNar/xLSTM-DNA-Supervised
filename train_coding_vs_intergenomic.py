# train_coding_vs_intergenomic.py
import os
os.environ['HF_HOME'] = os.path.join(os.getcwd(), 'hf_home')
os.environ['HF_DATASETS_CACHE'] = os.path.join(os.getcwd(), 'hf_home', 'datasets')
os.environ['HF_HUB_CACHE'] = os.path.join(os.getcwd(), 'hf_home', 'hub')

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from model import sLSTM, mLSTM
import time
import argparse
import json

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


class xLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=2, num_classes=2, dropout=0.3,
                 layer_type='alternating', use_exp_gating=True, use_stabilizer=True, 
                 use_normalizer=True, use_memory_mixing=True):
        super(xLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Stack of LSTM layers based on layer_type
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
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        embedded = self.dropout(embedded)
        
        # Process through LSTM layers
        batch_size, seq_len, _ = embedded.shape
        states = [None] * len(self.layers)
        
        # Aggregate sequence representation (mean pooling)
        hidden_states = []
        
        for t in range(seq_len):
            x_t = embedded[:, t, :]  # (batch_size, embedding_dim)
            x_t = self.dropout(x_t)
            for i, layer in enumerate(self.layers):
                if isinstance(layer, sLSTM):
                    x_t, cell, stab, norm = layer(x_t, states[i])
                    states[i] = (x_t, cell, stab, norm)
                else:  # mLSTM
                    x_t, cell, norm, stab = layer(x_t, states[i])
                    states[i] = (cell, norm, stab)
            hidden_states.append(x_t)
        
        # Mean pooling over sequence
        pooled = torch.stack(hidden_states, dim=1).mean(dim=1)  # (batch_size, hidden_size)
        pooled = self.dropout(pooled)
        
        logits = self.output_proj(pooled)  # (batch_size, num_classes)
        return logits


class DNADataset(Dataset):
    def __init__(self, sequences, labels, seq_length=200):
        self.sequences = sequences
        self.labels = labels
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        
        # Encode DNA sequence (A=0, C=1, G=2, T=3, N=4)
        encoding = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        encoded = [encoding.get(base.upper(), 4) for base in seq]
        
        # Pad or truncate to seq_length
        if len(encoded) > self.seq_length:
            encoded = encoded[:self.seq_length]
        else:
            encoded = encoded + [4] * (self.seq_length - len(encoded))
        
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def train_model(model, train_loader, test_loader, device, epochs=20, lr=5e-3, checkpoint_epochs=[5, 10, 15, 20], log_file=None, config=None):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    # Track metrics at checkpoint epochs
    metrics = {
        'train_acc': {},
        'test_acc': {},
        'train_f1': {},
        'test_f1': {},
        'train_loss': {},
        'test_loss': {}
    }
    
    # Open log file if specified
    log_fp = None
    if log_file:
        log_fp = open(log_file, 'w')
        # Write configuration as first line
        if config:
            log_fp.write(json.dumps({'type': 'config', **config}) + '\n')

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for step, (X_batch, y_batch) in enumerate(pbar):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * len(X_batch)
            correct += (logits.argmax(dim=-1) == y_batch).sum().item()
            total += len(X_batch)
            all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            
            step_acc = correct / total
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{step_acc:.3f}"})
            
            # Log step progress
            if log_fp:
                log_entry = {
                    'type': 'step',
                    'epoch': epoch,
                    'step': step,
                    'loss': loss.item(),
                    'accuracy': step_acc,
                    'samples_processed': total
                }
                log_fp.write(json.dumps(log_entry) + '\n')
            
            # Small delay to reduce GPU heat
            time.sleep(0.01)

        scheduler.step()

        train_acc = correct / total
        avg_loss = total_loss / total
        train_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        # Log epoch summary
        if log_fp:
            log_entry = {
                'type': 'epoch',
                'epoch': epoch,
                'train_loss': avg_loss,
                'train_accuracy': train_acc,
                'train_f1': train_f1
            }
            log_fp.write(json.dumps(log_entry) + '\n')

        # Evaluate at checkpoint epochs
        if epoch in checkpoint_epochs:
            model.eval()
            test_correct = 0
            test_total = 0
            test_loss = 0.0
            test_preds = []
            test_labels = []
            
            with torch.no_grad():
                for X_batch, y_batch in tqdm(test_loader, desc=f"Test Epoch {epoch}"):
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    logits = model(X_batch)
                    loss = criterion(logits, y_batch)
                    test_loss += loss.item() * len(X_batch)
                    test_correct += (logits.argmax(dim=-1) == y_batch).sum().item()
                    test_total += len(X_batch)
                    test_preds.extend(logits.argmax(dim=-1).cpu().numpy())
                    test_labels.extend(y_batch.cpu().numpy())
            
            test_acc = test_correct / test_total
            test_avg_loss = test_loss / test_total
            test_f1 = f1_score(test_labels, test_preds, average='weighted')
            
            # Store metrics
            metrics['train_acc'][epoch] = train_acc
            metrics['test_acc'][epoch] = test_acc
            metrics['train_f1'][epoch] = train_f1
            metrics['test_f1'][epoch] = test_f1
            metrics['train_loss'][epoch] = avg_loss
            metrics['test_loss'][epoch] = test_avg_loss
            
            # Log test evaluation
            if log_fp:
                log_entry = {
                    'type': 'evaluation',
                    'epoch': epoch,
                    'test_loss': test_avg_loss,
                    'test_accuracy': test_acc,
                    'test_f1': test_f1
                }
                log_fp.write(json.dumps(log_entry) + '\n')
            
            print(f"Epoch {epoch:2d} | Train Acc: {train_acc:.3f} | Test Acc: {test_acc:.3f} | Train F1: {train_f1:.3f} | Test F1: {test_f1:.3f} | Train Loss: {avg_loss:.4f} | Test Loss: {test_avg_loss:.4f}")
            model.train()
        else:
            print(f"Epoch {epoch:2d}/{epochs} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.3f} | Train F1: {train_f1:.3f}")
    
    # Close log file
    if log_fp:
        log_fp.close()

    return metrics


def plot_metrics(metrics, checkpoint_epochs, output_dir='.'):
    """Plot accuracy, F1, and loss metrics and save as JPG files"""
    epochs = sorted(metrics['train_acc'].keys())
    
    # Accuracy plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [metrics['train_acc'][e] for e in epochs], 'o-', label='Train Accuracy')
    plt.plot(epochs, [metrics['test_acc'][e] for e in epochs], 's-', label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train vs Test Accuracy')
    plt.legend()
    plt.grid(True)
    plt.xticks(epochs)
    plt.savefig(f'{output_dir}/accuracy.jpg', format='jpg', dpi=150, bbox_inches='tight')
    plt.close()
    
    # F1 score plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [metrics['train_f1'][e] for e in epochs], 'o-', label='Train F1')
    plt.plot(epochs, [metrics['test_f1'][e] for e in epochs], 's-', label='Test F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Train vs Test F1 Score')
    plt.legend()
    plt.grid(True)
    plt.xticks(epochs)
    plt.savefig(f'{output_dir}/f1_score.jpg', format='jpg', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [metrics['train_loss'][e] for e in epochs], 'o-', label='Train Loss')
    plt.plot(epochs, [metrics['test_loss'][e] for e in epochs], 's-', label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train vs Test Loss')
    plt.legend()
    plt.grid(True)
    plt.xticks(epochs)
    plt.savefig(f'{output_dir}/loss.jpg', format='jpg', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to {output_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='xLSTM DNA Classification with Ablation Studies')
    
    # Component toggles
    parser.add_argument('--no-exp-gating', action='store_true', help='Disable exponential gating')
    parser.add_argument('--no-stabilizer', action='store_true', help='Disable stabilizer gate')
    parser.add_argument('--no-normalizer', action='store_true', help='Disable normalizer')
    parser.add_argument('--no-memory-mixing', action='store_true', help='Disable memory mixing')
    
    # Layer type
    parser.add_argument('--layer-type', type=str, default='alternating', choices=['slstm', 'mlstm', 'alternating'],
                        help='Type of LSTM layers to use')
    
    # Parameter scaling
    parser.add_argument('--scale', type=str, default='medium', choices=['tiny', 'small', 'medium', 'large', 'xlarge'],
                        help='Model scale for parameter count (tiny~20k, small~50k, medium~100k, large~250k, xlarge~330k)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    
    args = parser.parse_args()
    
    # Set component flags
    use_exp_gating = not args.no_exp_gating
    use_stabilizer = not args.no_stabilizer
    use_normalizer = not args.no_normalizer
    use_memory_mixing = not args.no_memory_mixing
    
    # Set parameter scale
    scale_configs = {
        'tiny': (32, 64, 1),
        'small': (32, 128, 2),
        'medium': (64, 128, 2),
        'large': (64, 256, 2),
        'xlarge': (128, 256, 2)
    }
    embedding_dim, hidden_size, num_layers = scale_configs[args.scale]
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Configuration: exp_gating={use_exp_gating}, stabilizer={use_stabilizer}, normalizer={use_normalizer}, memory_mixing={use_memory_mixing}")
    print(f"Layer type: {args.layer_type}, Scale: {args.scale} (emb={embedding_dim}, hid={hidden_size}, layers={num_layers})")
    
    # Load dataset
    print("Loading dataset from Hugging Face...")
    dataset = load_dataset("katarinagresova/Genomic_Benchmarks_demo_coding_vs_intergenomic_seqs", streaming=True)
    
    # Use official train/test split - take first N samples for demo
    train_samples = 20000
    test_samples = 5000
    
    train_seqs = []
    train_labels = []
    for i, item in enumerate(dataset['train']):
        if i >= train_samples:
            break
        train_seqs.append(item['seq'])
        train_labels.append(item['label'])
    
    test_seqs = []
    test_labels = []
    for i, item in enumerate(dataset['test']):
        if i >= test_samples:
            break
        test_seqs.append(item['seq'])
        test_labels.append(item['label'])
    
    print(f"Train samples: {len(train_seqs)}, Test samples: {len(test_seqs)}")
    
    # Check for data overlap
    overlap = set(train_seqs) & set(test_seqs)
    print(f"Data overlap between train and test: {len(overlap)} sequences")
    
    # Create datasets and dataloaders
    seq_length = 100
    batch_size = 128
    
    train_dataset = DNADataset(train_seqs, train_labels, seq_length=seq_length)
    test_dataset = DNADataset(test_seqs, test_labels, seq_length=seq_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create xLSTM model
    vocab_size = 5
    num_classes = 2
    dropout = 0.3
    
    model = xLSTM(vocab_size, embedding_dim, hidden_size, num_layers, num_classes, dropout,
                 args.layer_type, use_exp_gating, use_stabilizer, use_normalizer, use_memory_mixing).to(device)
    param_count = count_parameters(model)
    print(f"Model parameters: {param_count:,}")
    
    # Generate log file name and config
    log_filename = f"train_log_{args.scale}_{args.layer_type}_exp{use_exp_gating}_stab{use_stabilizer}_norm{use_normalizer}_mix{use_memory_mixing}.jsonl"
    
    # Explicitly list disabled components
    disabled_components = []
    if not use_exp_gating:
        disabled_components.append('exponential_gating')
    if not use_stabilizer:
        disabled_components.append('stabilizer')
    if not use_normalizer:
        disabled_components.append('normalizer')
    if not use_memory_mixing:
        disabled_components.append('memory_mixing')
    
    config = {
        'scale': args.scale,
        'layer_type': args.layer_type,
        'components': {
            'exponential_gating': use_exp_gating,
            'stabilizer': use_stabilizer,
            'normalizer': use_normalizer,
            'memory_mixing': use_memory_mixing
        },
        'disabled_components': disabled_components if disabled_components else 'none',
        'model_config': {
            'embedding_dim': embedding_dim,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'vocab_size': vocab_size,
            'num_classes': num_classes,
            'dropout': dropout,
            'total_parameters': param_count
        },
        'training_config': {
            'epochs': args.epochs,
            'lr': args.lr,
            'batch_size': batch_size,
            'seq_length': seq_length
        },
        'dataset': {
            'train_samples': len(train_seqs),
            'test_samples': len(test_seqs)
        }
    }
    
    # Train model
    checkpoint_epochs = [5, 10, 15, 20]
    metrics = train_model(model, train_loader, test_loader, device, epochs=args.epochs, lr=args.lr, checkpoint_epochs=checkpoint_epochs, log_file=log_filename, config=config)
    print(f"Training log saved to {log_filename}")
    
    # Print ablation results summary
    final_test_acc = metrics['test_acc'][max(metrics['test_acc'].keys())]
    final_test_f1 = metrics['test_f1'][max(metrics['test_f1'].keys())]
    print("\n" + "="*80)
    print("ABLATION RESULTS SUMMARY")
    print("="*80)
    print(f"Parameters: {param_count:,}")
    print(f"Scale: {args.scale} | Layer Type: {args.layer_type}")
    print(f"Exp Gating: {use_exp_gating} | Stabilizer: {use_stabilizer} | Normalizer: {use_normalizer} | Memory Mixing: {use_memory_mixing}")
    print(f"Final Test Accuracy: {final_test_acc:.4f}")
    print(f"Final Test F1: {final_test_f1:.4f}")
    print("="*80)
    
    # Plot metrics
    plot_metrics(metrics, checkpoint_epochs)