"""
GraphSAGE GNN Model
====================
Defines a 2-layer GraphSAGE architecture for heterogeneous fraud detection
on the Account-Device-IP relationship graph.

Architecture:
    Input → SAGEConv(in, 64) → ReLU → Dropout(0.3)
          → SAGEConv(64, 64)  → ReLU → Dropout(0.3)
          → Linear(64, 2)     → output (legit vs fraud logits)

Why 2 layers?
    Each SAGEConv layer aggregates information from one hop of neighbours.
    Two layers = 2-hop neighbourhood aggregation:
        Layer 1: Account ← its direct devices + IPs
        Layer 2: Account ← neighbours' neighbours (other accounts sharing devices)
    This is exactly the depth needed to see Account → Device → Other Account
    relationships — the core structural signal for fraud ring detection.

Why GraphSAGE (not GCN or GAT)?
    GraphSAGE is inductive — it learns a neighbour-aggregation *function*,
    not fixed node embeddings. This means it can score brand-new accounts
    that weren't in the training graph, which is critical for fraud detection
    where fraudsters constantly create new accounts.

The model is defined as a standard homogeneous GNN, then converted to
handle the heterogeneous graph structure using PyG's to_hetero() utility.
This automatically creates separate weight matrices for each edge type.

Usage:
    from src.ml.gnn_model import create_fraud_gnn

    model = create_fraud_gnn(data.metadata(), hidden_channels=64)
"""

import logging

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class FraudGNN(torch.nn.Module):
    """
    2-layer GraphSAGE for node-level fraud classification.

    This class defines the *homogeneous* version of the model. It is
    converted to a heterogeneous model by create_fraud_gnn() using
    PyG's to_hetero() — which automatically duplicates the convolution
    layers for each edge type in the graph.

    Architecture detail:
        conv1:  SAGEConv(in_features → hidden_channels)
                Aggregates 1-hop neighbour features using mean pooling,
                then concatenates with the node's own features and
                projects through a linear layer.

        conv2:  SAGEConv(hidden_channels → hidden_channels)
                Same operation on the already-aggregated representations,
                extending the receptive field to 2 hops.

        linear: Linear(hidden_channels → 2)
                Maps the learned 64-dim embedding to 2 class logits
                (legit, fraud) for CrossEntropyLoss.
    """

    def __init__(self, hidden_channels: int = 64):
        super().__init__()
        # (-1, -1) tells SAGEConv to lazily infer input dimensions
        # on the first forward pass. Required for heterogeneous graphs
        # where different node types have different feature dimensions.
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, 2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the 2-layer GraphSAGE + classification head.

        Args:
            x:          Node feature tensor
            edge_index: Edge index tensor (COO format, shape [2, num_edges])

        Returns:
            Logits tensor of shape [num_nodes, 2] (legit, fraud)
        """
        # Layer 1: aggregate 1-hop neighbours
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)

        # Layer 2: aggregate 2-hop neighbours (from already-aggregated features)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)

        # Classification head
        x = self.linear(x)
        return x


def create_fraud_gnn(
    metadata: tuple,
    hidden_channels: int = 64,
) -> torch.nn.Module:
    """
    Create a heterogeneous GraphSAGE model for fraud detection.

    Uses PyG's to_hetero() to automatically convert the homogeneous
    FraudGNN into a model that handles our 3 node types and 4 edge types:
        - account → device, account → ip (forward)
        - device → account, ip → account (reverse)

    to_hetero() creates separate weight matrices for each edge type,
    then aggregates incoming messages with 'sum'. This means an account
    node receives:
        sum(messages_from_devices, messages_from_ips)
    at each layer.

    Args:
        metadata:        Output of HeteroData.metadata() — describes the
                         graph's node types and edge types.
        hidden_channels: Dimensionality of the hidden GNN layers.

    Returns:
        A heterogeneous torch.nn.Module ready for training.
    """
    logger.info(f"Creating FraudGNN (hidden_channels={hidden_channels})...")

    model = FraudGNN(hidden_channels=hidden_channels)
    model = to_hetero(model, metadata, aggr="sum")

    logger.info("Heterogeneous FraudGNN created (parameters will be initialized on first forward pass).")

    return model
