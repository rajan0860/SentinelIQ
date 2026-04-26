"""
train_model.py — CLI wrapper for the Phase 3 training pipeline.

Thin entry point that parses command-line arguments and delegates to
src.ml.train.train(). Keeps the training script clean and gives operators
a simple, memorable command to retrain the model.

Usage:
    python scripts/train_model.py

    # With explicit paths:
    python scripts/train_model.py \
        --data  data/synthetic/events.csv \
        --graph data/graphs/account_graph.pkl \
        --output data/models/

    # Override individual paths:
    python scripts/train_model.py --data data/raw/live_events.csv
"""

import argparse
import os
import sys

# ── ensure project root is on PYTHONPATH ─────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.ml.train import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="train_model.py",
        description="Train the SentinelIQ ensemble fraud detector "
                    "(XGBoost + Isolation Forest).",
    )
    parser.add_argument(
        "--data",
        default="data/synthetic/events.csv",
        help="Path to events CSV (default: data/synthetic/events.csv)",
    )
    parser.add_argument(
        "--graph",
        default="data/graphs/account_graph.pkl",
        help="Path to account graph pickle (default: data/graphs/account_graph.pkl)",
    )
    parser.add_argument(
        "--output",
        default="data/models/",
        help="Directory to save model artifacts (default: data/models/)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"\n{'='*60}")
    print("  SentinelIQ — Model Training")
    print(f"{'='*60}")
    print(f"  Data   : {args.data}")
    print(f"  Graph  : {args.graph}")
    print(f"  Output : {args.output}")
    print(f"{'='*60}\n")

    metrics = train(
        data_path=args.data,
        graph_path=args.graph,
        output_dir=args.output,
    )

    print(f"\n{'='*60}")
    print("  Training complete — artifacts saved to:", args.output)
    print(f"  XGBoost AUC  : {metrics['xgb_auc']}")
    print(f"  F1 Score     : {metrics['xgb_f1']}")
    print(f"  ISO Rate     : {metrics['iso_detection_rate']}")
    print(f"{'='*60}\n")
