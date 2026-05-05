"""
GNN Trainer Module
===================
Training loop for the GraphSAGE fraud detection model.

Handles the full training pipeline:
    1. Build HeteroData from events + graph (via gnn_data.py)
    2. Create heterogeneous GraphSAGE model (via gnn_model.py)
    3. Train with class-weighted CrossEntropyLoss
    4. Early stopping on validation AUC (patience=20)
    5. Save best model checkpoint to data/models/gnn_fraud.pt

Class imbalance handling:
    Like XGBoost's scale_pos_weight, we use weighted CrossEntropyLoss
    where the fraud class gets a weight of (num_legit / num_fraud) ≈ 65.
    This tells the network: "missing a fraud case is 65× worse than
    a false positive" — the same principle as SMOTE but applied at the
    loss function level (more natural for neural networks).

Usage:
    from src.ml.gnn_trainer import train_gnn

    metrics = train_gnn(
        events_path="data/synthetic/events.csv",
        graph_path="data/graphs/account_graph.pkl",
        output_dir="data/models/",
        epochs=200,
    )
"""

import logging
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_gnn(
    events_path: str,
    graph_path: str,
    output_dir: str,
    hidden_channels: int = 64,
    epochs: int = 200,
    lr: float = 0.01,
    patience: int = 20,
) -> dict:
    """
    Full GNN training pipeline with early stopping.

    Args:
        events_path:     Path to events.csv
        graph_path:      Path to account_graph.pkl
        output_dir:      Directory to save gnn_fraud.pt
        hidden_channels: GNN hidden layer dimensionality
        epochs:          Maximum training epochs
        lr:              Learning rate for Adam optimizer
        patience:        Early stopping patience (epochs without improvement)

    Returns:
        dict with keys: gnn_auc, gnn_precision, gnn_recall, gnn_f1,
                        gnn_epochs_trained
    """
    import torch
    
    # Fix OpenMP deadlock when mixing XGBoost and PyTorch in the same process
    torch.set_num_threads(1)
    
    from sklearn.metrics import (
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    from src.ml.gnn_data import build_hetero_data
    from src.ml.gnn_model import create_fraud_gnn

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── 1. Build HeteroData ──────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("GNN STEP 1: Building HeteroData from events + graph...")
    logger.info("=" * 60)
    data = build_hetero_data(events_path, graph_path)

    # ── 2. Create model ──────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("GNN STEP 2: Creating heterogeneous GraphSAGE model...")
    logger.info("=" * 60)
    model = create_fraud_gnn(data.metadata(), hidden_channels=hidden_channels)

    # ── 3. Setup training ────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("GNN STEP 3: Configuring optimizer and loss...")
    logger.info("=" * 60)

    # Class-weighted loss — same principle as XGBoost's scale_pos_weight
    y = data["account"].y
    num_legit = int((y == 0).sum())
    num_fraud = int((y == 1).sum())

    if num_fraud == 0:
        logger.warning("No fraud accounts found — GNN training skipped.")
        return {"gnn_auc": 0.0, "gnn_f1": 0.0, "gnn_epochs_trained": 0}

    class_weight = torch.tensor(
        [1.0, float(num_legit) / float(num_fraud)],
        dtype=torch.float32,
    )
    logger.info(f"Class weights: legit={class_weight[0]:.1f}, fraud={class_weight[1]:.1f}")

    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weight)

    # ── Dummy forward pass to initialize LazyModule parameters ──
    with torch.no_grad():
        model(data.x_dict, data.edge_index_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_mask = data["account"].train_mask
    val_mask = data["account"].val_mask

    # ── 4. Training loop with early stopping ─────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"GNN STEP 4: Training for up to {epochs} epochs (patience={patience})...")
    logger.info("=" * 60)

    best_val_auc = 0.0
    best_state = None
    epochs_without_improvement = 0
    actual_epochs = 0

    for epoch in range(1, epochs + 1):
        # ── Train step ────────────────────────────────────────────────────
        model.train()
        optimizer.zero_grad()

        # Forward pass: feed all node features + all edges
        out = model(data.x_dict, data.edge_index_dict)

        # Loss only on training account nodes
        pred_train = out["account"][train_mask]
        true_train = y[train_mask]

        loss = loss_fn(pred_train, true_train)
        loss.backward()
        optimizer.step()

        # ── Validation step ───────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            out_val = model(data.x_dict, data.edge_index_dict)
            pred_val = out_val["account"][val_mask]
            true_val = y[val_mask]

            # Convert logits to probabilities for AUC
            probs_val = torch.softmax(pred_val, dim=1)[:, 1].numpy()
            labels_val = true_val.numpy()

            # Skip AUC if only one class present in val set
            if len(np.unique(labels_val)) < 2:
                val_auc = 0.0
            else:
                val_auc = roc_auc_score(labels_val, probs_val)

        actual_epochs = epoch

        # Log every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"  Epoch {epoch:3d}/{epochs} — "
                f"loss: {loss.item():.4f}, val_auc: {val_auc:.4f}"
            )

        # ── Early stopping check ──────────────────────────────────────────
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            logger.info(
                f"  Early stopping at epoch {epoch} — "
                f"no improvement for {patience} epochs. "
                f"Best val AUC: {best_val_auc:.4f}"
            )
            break

    # ── 5. Load best model and compute final metrics ─────────────────────────
    logger.info("=" * 60)
    logger.info("GNN STEP 5: Evaluating best model on validation set...")
    logger.info("=" * 60)

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        out_final = model(data.x_dict, data.edge_index_dict)
        pred_final = out_final["account"][val_mask]

        probs_final = torch.softmax(pred_final, dim=1)[:, 1].numpy()
        preds_binary = (probs_final >= 0.5).astype(int)
        labels_final = y[val_mask].numpy()

    if len(np.unique(labels_final)) < 2:
        gnn_auc = 0.0
        gnn_precision = 0.0
        gnn_recall = 0.0
        gnn_f1 = 0.0
    else:
        gnn_auc = roc_auc_score(labels_final, probs_final)
        gnn_precision = precision_score(labels_final, preds_binary, zero_division=0)
        gnn_recall = recall_score(labels_final, preds_binary, zero_division=0)
        gnn_f1 = f1_score(labels_final, preds_binary, zero_division=0)

    print("\n" + "─" * 50)
    print("  GraphSAGE GNN Validation Results")
    print("─" * 50)
    print(f"  AUC       : {gnn_auc:.4f}")
    print(f"  Precision : {gnn_precision:.4f}")
    print(f"  Recall    : {gnn_recall:.4f}")
    print(f"  F1        : {gnn_f1:.4f}")
    print(f"  Epochs    : {actual_epochs} (best at {actual_epochs - epochs_without_improvement})")

    # ── 6. Save model ────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("GNN STEP 6: Saving model artifacts...")
    logger.info("=" * 60)

    model_path = output_path / "gnn_fraud.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "metadata": data.metadata(),
        "hidden_channels": hidden_channels,
        "metrics": {
            "auc": gnn_auc,
            "precision": gnn_precision,
            "recall": gnn_recall,
            "f1": gnn_f1,
        },
    }, str(model_path))
    logger.info(f"GNN model saved → {model_path}")

    # Also save the HeteroData metadata separately so the scorer can
    # reconstruct the model architecture without loading the full dataset
    metadata_path = output_path / "gnn_metadata.pkl"
    import pickle
    with open(metadata_path, "wb") as f:
        pickle.dump({
            "metadata": data.metadata(),
            "hidden_channels": hidden_channels,
            "account_feature_mean": data["account"].x.mean(dim=0).numpy(),
            "account_feature_std": data["account"].x.std(dim=0).numpy(),
        }, f)
    logger.info(f"GNN metadata saved → {metadata_path}")

    metrics = {
        "gnn_auc": round(gnn_auc, 4),
        "gnn_precision": round(gnn_precision, 4),
        "gnn_recall": round(gnn_recall, 4),
        "gnn_f1": round(gnn_f1, 4),
        "gnn_epochs_trained": actual_epochs,
    }

    return metrics


# ─── Quick verification ──────────────────────────────────────────────────────
if __name__ == "__main__":
    metrics = train_gnn(
        events_path="data/synthetic/events.csv",
        graph_path="data/graphs/account_graph.pkl",
        output_dir="data/models/",
        epochs=100,
    )
    print("\n" + "=" * 50)
    print("  GNN Training Complete — Summary")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k:<25} {v}")
    print("=" * 50)
