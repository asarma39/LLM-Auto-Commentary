from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data_loading.cricket_dataset import create_cricket_datasets, CricketSignalsDataset
from src.models.seq_cnn import SimpleCricketCNN


def get_device() -> torch.device:
    """Return the appropriate device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dataloaders(
    train_ds,
    val_ds,
    test_ds,
    batch_size: int = 8,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders for train, validation, and test datasets."""
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Train the model for one epoch and return (average loss, accuracy)."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    avg_loss = total_loss / max(1, total_samples)
    accuracy = correct / max(1, total)
    return avg_loss, accuracy


@torch.no_grad()
def evaluate_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Evaluate the model and return accuracy."""
    model.eval()
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return correct / max(1, total)


@torch.no_grad()
def compute_confusion_matrix(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> np.ndarray:
    """Compute confusion matrix and return as numpy array."""
    model.eval()
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1)
        
        for true_label, pred_label in zip(y.cpu().numpy(), preds.cpu().numpy()):
            confusion_matrix[true_label, pred_label] += 1

    return confusion_matrix


def train_seq_classifier(
    data_root: str,
    max_len: int = 400,
    batch_size: int = 8,
    lr: float = 1e-3,
    num_epochs: int = 50,
    patience: int = 8,
    model_dir: str = "models/checkpoints",
    return_history: bool = False,
) -> Tuple[Dict, List[Dict]]:
    """Train a SimpleCricketCNN model on ViSig cricket umpire signals."""
    device = get_device()
    print(f"Using device: {device}")

    train_ds, val_ds, test_ds = create_cricket_datasets(
        root=data_root,
        max_len=max_len,
    )

    base_ds: CricketSignalsDataset = train_ds.dataset  
    input_dim = base_ds.feature_dim
    num_classes = base_ds.num_classes

    print(f"Input dim: {input_dim}, num_classes: {num_classes}")
    print(f"Dataset sizes -> train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    train_loader, val_loader, test_loader = make_dataloaders(
        train_ds, val_ds, test_ds, batch_size=batch_size
    )

    model = SimpleCricketCNN(
        input_dim=input_dim,
        num_classes=num_classes,
        num_channels=128,
        num_layers=2,
        kernel_size=5,
        dropout=0.3,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    best_state = None
    epochs_no_improve = 0
    history = []  # Track training history

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc = evaluate_accuracy(model, val_loader, device)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
        })

        improved = val_acc > best_val_acc + 1e-4
        if improved:
            best_val_acc = val_acc
            best_state = {
                "model_state": model.state_dict(),
                "input_dim": input_dim,
                "num_classes": num_classes,
                "epoch": epoch,
                "val_acc": val_acc,
            }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(
            f"Epoch {epoch:03d}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f}, val_acc={val_acc:.3f}, "
            f"best_val_acc={best_val_acc:.3f}, no_improve={epochs_no_improve}"
        )

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    if best_state is None:
        print("Warning: no improvement recorded; using final model state.")
        best_state = {
            "model_state": model.state_dict(),
            "input_dim": input_dim,
            "num_classes": num_classes,
            "epoch": epoch,
            "val_acc": best_val_acc,
        }

    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    ckpt_file = model_path / "visig_simple_cnn.pt"
    torch.save(best_state, ckpt_file)
    print(f"Saved best model to {ckpt_file} (val_acc={best_state['val_acc']:.3f})")

    model.load_state_dict(best_state["model_state"])
    test_acc = evaluate_accuracy(model, test_loader, device)
    print(f"\nTest accuracy: {test_acc:.3f}")

    print("\nConfusion Matrix (rows=true, cols=predicted):")
    confusion_matrix = compute_confusion_matrix(model, test_loader, device, num_classes)
    
    idx_to_label = base_ds.idx_to_label
    label_names = [idx_to_label[i] for i in range(num_classes)]
    
    max_label_len = max(len(name) for name in label_names) if label_names else 8
    header = f"{'True\\Pred':<{max_label_len + 2}}"
    for name in label_names:
        header += f"{name[:8]:>10}"  
    print(header)
    print("-" * len(header))
    
    for i, true_label in enumerate(label_names):
        row = f"{true_label:<{max_label_len + 2}}"
        for j in range(num_classes):
            count = confusion_matrix[i, j]
            row += f"{count:>10}"
        print(row)
    
    print("\nPer-class accuracy:")
    for i, label_name in enumerate(label_names):
        class_correct = confusion_matrix[i, i]
        class_total = confusion_matrix[i, :].sum()
        class_acc = class_correct / max(1, class_total)
        print(f"  {label_name}: {class_acc:.3f} ({class_correct}/{class_total})")
    
    results = {
        "test_acc": test_acc,
        "best_val_acc": best_val_acc,
        "confusion_matrix": confusion_matrix,
        "label_names": label_names,
        "checkpoint_path": str(ckpt_file),
        "num_classes": num_classes,
    }
    
    if return_history:
        return results, history
    return results, []


def test_model(
    checkpoint_path: str,
    data_root: str,
    max_len: int = 400,
    batch_size: int = 8,
) -> Dict:
    """
    Load a saved model and evaluate it on the test set.
    
    Args:
        checkpoint_path: Path to the saved model checkpoint
        data_root: Root directory containing .mat files
        max_len: Maximum sequence length (should match training)
        batch_size: Batch size for evaluation
    
    Returns:
        Dictionary with test results including accuracy, confusion matrix, etc.
    """
    device = get_device()
    print(f"Using device: {device}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Validation accuracy: {checkpoint['val_acc']:.3f}")
    
    input_dim = checkpoint['input_dim']
    num_classes = checkpoint['num_classes']
    
    train_ds, val_ds, test_ds = create_cricket_datasets(
        root=data_root,
        max_len=max_len,
    )
    
    base_ds: CricketSignalsDataset = test_ds.dataset
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    model = SimpleCricketCNN(
        input_dim=input_dim,
        num_classes=num_classes,
        num_channels=128,
        num_layers=2,
        kernel_size=5,
        dropout=0.3,
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    test_acc = evaluate_accuracy(model, test_loader, device)
    confusion_matrix = compute_confusion_matrix(model, test_loader, device, num_classes)
    
    idx_to_label = base_ds.idx_to_label
    label_names = [idx_to_label[i] for i in range(num_classes)]
    
    print(f"\nTest accuracy: {test_acc:.3f}")
    print(f"\nConfusion Matrix (rows=true, cols=predicted):")
    
    max_label_len = max(len(name) for name in label_names) if label_names else 8
    header = f"{'True\\Pred':<{max_label_len + 2}}"
    for name in label_names:
        header += f"{name[:8]:>10}"
    print(header)
    print("-" * len(header))
    
    for i, true_label in enumerate(label_names):
        row = f"{true_label:<{max_label_len + 2}}"
        for j in range(num_classes):
            count = confusion_matrix[i, j]
            row += f"{count:>10}"
        print(row)
    
    print("\nPer-class accuracy:")
    for i, label_name in enumerate(label_names):
        class_correct = confusion_matrix[i, i]
        class_total = confusion_matrix[i, :].sum()
        class_acc = class_correct / max(1, class_total)
        print(f"  {label_name}: {class_acc:.3f} ({class_correct}/{class_total})")
    
    return {
        "test_acc": test_acc,
        "confusion_matrix": confusion_matrix,
        "label_names": label_names,
        "num_classes": num_classes,
    }


if __name__ == "__main__":
    data_root = os.getenv("VISIG_ROOT")
    if not data_root:
        raise SystemExit(
            "Please set VISIG_ROOT environment variable to the directory "
            "containing the ViSig .mat files."
        )

    train_seq_classifier(
        data_root=data_root,
        max_len=400,
        batch_size=8,
        lr=1e-3,
        num_epochs=40,
        patience=8,
        return_history=False,
    )

