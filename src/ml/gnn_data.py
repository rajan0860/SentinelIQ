"""
GNN Data Bridge Module
======================
Transforms the SentinelIQ NetworkX graph and tabular events into a
PyTorch Geometric (PyG) HeteroData structure for GraphSAGE training.

The graph has 3 node types:
  - Account (has tabular features)
  - Device (no features — receives learned embeddings)
  - IP (no features — receives learned embeddings)

And 4 edge types (forward + reverse for bidirectional message passing):
  - (account, uses_device, device)     / (device, rev_uses_device, account)
  - (account, uses_ip, ip)             / (ip, rev_uses_ip, account)
"""

import logging
import pickle

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

logger = logging.getLogger(__name__)


def build_hetero_data(events_path: str, graph_path: str) -> HeteroData:
    """
    Constructs a PyG HeteroData object from the raw events and NetworkX graph.

    Args:
        events_path: Path to events.csv (contains tabular features)
        graph_path:  Path to account_graph.pkl (NetworkX graph)

    Returns:
        A torch_geometric.data.HeteroData object ready for GNN training.
    """
    logger.info("Loading NetworkX graph and tabular events...")
    
    # Load NetworkX graph
    with open(graph_path, "rb") as f:
        nx_graph = pickle.load(f)

    # Load and aggregate tabular events per account
    events_df = pd.read_csv(events_path)
    
    # We only use a subset of features for the GNN to learn from
    # We exclude graph-based features since the GNN will learn them naturally
    tabular_features = [
        "transaction_amount",
        "account_age_days",
        "device_change_count",
        "ip_country_mismatch",
        "velocity_1hr",
        "avg_txn_amount_30d",
        "failed_login_count_24hr"
    ]
    
    # Aggregate features by account
    account_stats = events_df.groupby("account_id")[tabular_features].mean()
    account_labels = events_df.groupby("account_id")["is_fraud"].max()

    # Create mapping from string IDs to integer indices for PyG
    account_mapping = {n: i for i, n in enumerate(
        [n for n, d in nx_graph.nodes(data=True) if d.get("node_type") == "account"]
    )}
    device_mapping = {n: i for i, n in enumerate(
        [n for n, d in nx_graph.nodes(data=True) if d.get("node_type") == "device"]
    )}
    ip_mapping = {n: i for i, n in enumerate(
        [n for n, d in nx_graph.nodes(data=True) if d.get("node_type") == "ip"]
    )}

    num_accounts = len(account_mapping)
    num_devices = len(device_mapping)
    num_ips = len(ip_mapping)

    logger.info(f"Graph stats: {num_accounts} accounts, {num_devices} devices, {num_ips} IPs.")

    # ── 1. Create Node Feature Tensors ───────────────────────────────────────
    
    # Account features
    x_account = torch.zeros((num_accounts, len(tabular_features)), dtype=torch.float)
    y_account = torch.zeros(num_accounts, dtype=torch.long)

    # Standardise features globally to help GNN convergence
    eps = 1e-8
    mean = account_stats.mean()
    std = account_stats.std() + eps
    account_stats_norm = (account_stats - mean) / std

    for acc_id, idx in account_mapping.items():
        # Graph nodes have an 'ACC:' prefix, but events dataframe has 'ACC-XXXX'
        raw_acc_id = acc_id.replace("ACC:", "")
        
        if raw_acc_id in account_stats_norm.index:
            features = account_stats_norm.loc[raw_acc_id].values
            label = int(account_labels.loc[raw_acc_id])
            x_account[idx] = torch.tensor(features, dtype=torch.float)
            y_account[idx] = torch.tensor(label, dtype=torch.long)

    data = HeteroData()
    data["account"].x = x_account
    data["account"].y = y_account

    # Device and IP nodes have no inherent features. We initialize them with
    # a constant 1.0. PyG's SAGEConv will automatically project this into a
    # learned embedding space during the first layer.
    data["device"].x = torch.ones((num_devices, 1), dtype=torch.float)
    data["ip"].x = torch.ones((num_ips, 1), dtype=torch.float)

    # ── 2. Create Edge Index Tensors ─────────────────────────────────────────

    acc_device_src, acc_device_dst = [], []
    acc_ip_src, acc_ip_dst = [], []

    for u, v in nx_graph.edges():
        u_type = nx_graph.nodes[u].get("node_type")
        v_type = nx_graph.nodes[v].get("node_type")

        # Handle Account <-> Device
        if u_type == "account" and v_type == "device":
            acc_device_src.append(account_mapping[u])
            acc_device_dst.append(device_mapping[v])
        elif u_type == "device" and v_type == "account":
            acc_device_src.append(account_mapping[v])
            acc_device_dst.append(device_mapping[u])
            
        # Handle Account <-> IP
        elif u_type == "account" and v_type == "ip":
            acc_ip_src.append(account_mapping[u])
            acc_ip_dst.append(ip_mapping[v])
        elif u_type == "ip" and v_type == "account":
            acc_ip_src.append(account_mapping[v])
            acc_ip_dst.append(ip_mapping[u])

    # Convert to PyG COO format [2, num_edges]
    edge_index_acc_dev = torch.tensor([acc_device_src, acc_device_dst], dtype=torch.long)
    edge_index_acc_ip = torch.tensor([acc_ip_src, acc_ip_dst], dtype=torch.long)

    # Add forward edges
    data["account", "uses_device", "device"].edge_index = edge_index_acc_dev
    data["account", "uses_ip", "ip"].edge_index = edge_index_acc_ip

    # Add reverse edges (required for PyG's to_hetero bidirectional passing)
    data["device", "rev_uses_device", "account"].edge_index = edge_index_acc_dev.flip([0])
    data["ip", "rev_uses_ip", "account"].edge_index = edge_index_acc_ip.flip([0])

    # ── 3. Create Train/Val Masks ────────────────────────────────────────────
    
    # 80/20 train/val split
    torch.manual_seed(42)
    perm = torch.randperm(num_accounts)
    split_idx = int(num_accounts * 0.8)
    
    train_mask = torch.zeros(num_accounts, dtype=torch.bool)
    val_mask = torch.zeros(num_accounts, dtype=torch.bool)
    
    train_mask[perm[:split_idx]] = True
    val_mask[perm[split_idx:]] = True

    data["account"].train_mask = train_mask
    data["account"].val_mask = val_mask

    return data
