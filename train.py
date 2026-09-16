# train.py
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
from safetensors.torch import save_file
from model import sLSTM, mLSTM
import time

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


class xLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=2, num_classes=2, dropout=0.3):
        super(xLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Stack of LSTM layers (alternating sLSTM and mLSTM)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i % 2 == 0:
                self.layers.append(sLSTM(embedding_dim if i == 0 else hidden_size, hidden_size))
            else:
                self.layers.append(mLSTM(embedding_dim if i == 0 else hidden_size, hidden_size))
        
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


def train_model(model, train_loader, test_loader, device, epochs=20, lr=5e-3, checkpoint_epochs=[5, 10, 15, 20]):
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

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for X_batch, y_batch in pbar:
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
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{correct/total:.3f}"})
            
            # Small delay to reduce GPU heat
            time.sleep(0.01)

        scheduler.step()

        train_acc = correct / total
        avg_loss = total_loss / total
        train_f1 = f1_score(all_labels, all_preds, average='weighted')

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
            
            print(f"Epoch {epoch:2d} | Train Acc: {train_acc:.3f} | Test Acc: {test_acc:.3f} | Train F1: {train_f1:.3f} | Test F1: {test_f1:.3f} | Train Loss: {avg_loss:.4f} | Test Loss: {test_avg_loss:.4f}")
            model.train()
        else:
            print(f"Epoch {epoch:2d}/{epochs} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.3f} | Train F1: {train_f1:.3f}")

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
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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
    
    # Create xLSTM model with smaller size to prevent overfitting
    vocab_size = 5  # A, C, G, T, N
    embedding_dim = 64
    hidden_size = 128
    num_layers = 2
    num_classes = 2
    dropout = 0.3
    
    model = xLSTM(vocab_size, embedding_dim, hidden_size, num_layers, num_classes, dropout).to(device)
    param_count = count_parameters(model)
    print(f"Model parameters: {param_count:,}")
    
    # Train model
    checkpoint_epochs = [5, 10, 15, 20]
    metrics = train_model(model, train_loader, test_loader, device, epochs=20, lr=1e-3, checkpoint_epochs=checkpoint_epochs)
    
    # Save model weights in safetensors format
    print("Saving model weights in safetensors format...")
    state_dict = model.state_dict()
    metadata = {
        'vocab_size': str(vocab_size),
        'embedding_dim': str(embedding_dim),
        'hidden_size': str(hidden_size),
        'num_layers': str(num_layers),
        'num_classes': str(num_classes),
        'total_parameters': str(param_count),
        'seq_length': str(seq_length),
        'train_samples': str(len(train_seqs)),
        'test_samples': str(len(test_seqs))
    }
    save_file(state_dict, "model_weights.safetensors", metadata=metadata)
    print("Model weights and metadata saved to model_weights.safetensors")
    
    # Plot metrics
    plot_metrics(metrics, checkpoint_epochs)