"""
Graph Feature Extractor
=======================
Loads the serialised NetworkX account-device-IP graph (built in Phase 2)
and extracts four structural features per account node.

These graph-derived features expose patterns that pure tabular ML cannot see:
  - Accounts that sit at the hub of many device/IP sharing relationships
  - Accounts embedded in large connected fraud clusters
  - Devices shared across multiple accounts (mule rings)
  - IP addresses shared across multiple accounts (credential stuffing / synthetic identity)

Usage:
    extractor = GraphFeatureExtractor("data/graphs/account_graph.pkl")
    features  = extractor.extract("ACC-00412")
    # {"degree_centrality": 0.12, "component_size": 4,
    #  "shared_device_count": 2, "ip_reuse_count": 1}
"""

import pickle
import logging
from pathlib import Path

import networkx as nx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class GraphFeatureExtractor:
    """
    Extracts graph-structural features for a given account from the
    pre-built NetworkX relationship graph.

    Performance note:
        degree_centrality is pre-computed once at init time (O(V + E)).
        Subsequent calls to extract() are O(k) where k is the account's
        local neighbourhood size — fast even for large graphs.
    """

    # Safe default returned when an account ID is not present in the graph
    _ZERO_FEATURES: dict = {
        "degree_centrality": 0.0,
        "component_size": 0,
        "shared_device_count": 0,
        "ip_reuse_count": 0,
    }

    def __init__(self, graph_path: str):
        """
        Load the pickled NetworkX graph and pre-compute centrality.

        Args:
            graph_path: Path to the serialised graph file produced by
                        src/ingestion/graph_builder.py
        """
        path = Path(graph_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Graph file not found at: {path}\n"
                "Run src/ingestion/graph_builder.py first to generate it."
            )

        logger.info(f"Loading graph from {path} ...")
        with open(path, "rb") as f:
            self.graph: nx.Graph = pickle.load(f)

        logger.info(
            f"Graph loaded — {self.graph.number_of_nodes():,} nodes, "
            f"{self.graph.number_of_edges():,} edges."
        )

        # Pre-compute degree centrality for all nodes ONCE.
        # Calling nx.degree_centrality() inside extract() would recompute
        # the entire graph on every call — far too slow at inference time.
        logger.info("Pre-computing degree centrality ...")
        self._centrality: dict = nx.degree_centrality(self.graph)
        logger.info("GraphFeatureExtractor ready.")

    def extract(self, account_id: str) -> dict:
        """
        Compute the four structural features for a single account.

        Args:
            account_id: The raw account identifier (e.g. "ACC-00412").
                        The method prepends the graph node prefix internally.

        Returns:
            dict with keys:
                degree_centrality  (float)  0.0 – 1.0
                component_size     (int)    total nodes in this account's cluster
                shared_device_count(int)    devices shared with ≥ 1 other account
                ip_reuse_count     (int)    IPs shared with ≥ 1 other account
        """
        # Graph nodes are stored with a type prefix to avoid collisions
        # between account IDs, device IDs, and IP addresses that may overlap.
        acc_node = f"ACC:{account_id}"

        # Graceful fallback: if the account never appeared in the ingested
        # events (e.g. brand-new account), return safe zero defaults.
        if acc_node not in self.graph:
            return dict(self._ZERO_FEATURES)

        # ── Feature 1: Degree Centrality ─────────────────────────────────────
        # Retrieved from pre-computed dict — O(1)
        degree_centrality = self._centrality.get(acc_node, 0.0)

        # ── Feature 2: Connected Component Size ──────────────────────────────
        # nx.node_connected_component returns the full set of nodes reachable
        # from acc_node. A massive component size indicates the account is
        # embedded in a large fraud cluster (all sharing devices/IPs).
        component_nodes = nx.node_connected_component(self.graph, acc_node)
        component_size = len(component_nodes)

        # ── Features 3 & 4: Shared Device / IP Counts ────────────────────────
        # Walk the account's direct neighbours. For each DEV: or IP: neighbour,
        # count how many OTHER accounts (ACC: nodes) also connect to that node.
        # Subtract 1 to exclude the account itself from the count.
        shared_device_count = 0
        ip_reuse_count = 0

        for neighbour in self.graph.neighbors(acc_node):
            if neighbour.startswith("DEV:"):
                # Count how many accounts share this specific device
                co_users = [
                    n for n in self.graph.neighbors(neighbour)
                    if n.startswith("ACC:") and n != acc_node
                ]
                shared_device_count += len(co_users)

            elif neighbour.startswith("IP:"):
                # Count how many accounts share this specific IP
                co_users = [
                    n for n in self.graph.neighbors(neighbour)
                    if n.startswith("ACC:") and n != acc_node
                ]
                ip_reuse_count += len(co_users)

        return {
            "degree_centrality": round(degree_centrality, 6),
            "component_size": component_size,
            "shared_device_count": shared_device_count,
            "ip_reuse_count": ip_reuse_count,
        }


# ─── Quick verification ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    import sys

    # Allow running directly from the project root or from this file's location
    sys.path.append(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )

    GRAPH_PATH = "data/graphs/account_graph.pkl"
    extractor = GraphFeatureExtractor(GRAPH_PATH)

    # Sample a legitimate account and a synthetic-ring account
    test_accounts = ["ACC-00001", "ACC-90001"]

    print("\n" + "=" * 55)
    print("  Graph Feature Extraction — Verification")
    print("=" * 55)
    for acct in test_accounts:
        feats = extractor.extract(acct)
        print(f"\nAccount : {acct}")
        for k, v in feats.items():
            print(f"  {k:<25} {v}")
    print("\n" + "=" * 55)
    print("Step 3.1 complete ✓")
