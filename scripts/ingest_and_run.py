"""
ingest_and_run.py — Manual trigger for the full SentinelIQ ingestion pipeline.

Runs the complete flow:
    1. Load events from CSV
    2. Build / refresh the account-device-IP graph
    3. Score every event with the ensemble ML model (XGBoost + Isolation Forest + SHAP)
    4. Save scored events to data/processed/scored_events.json (feeds the API)
    5. Optionally embed historical cases into ChromaDB (--embed-cases flag)
    6. Run the LangGraph agent on flagged events and add reports to the review queue

Usage:
    # Standard run (no re-embedding)
    python scripts/ingest_and_run.py

    # First run — embed historical cases into ChromaDB
    python scripts/ingest_and_run.py --embed-cases

    # Custom paths
    python scripts/ingest_and_run.py \\
        --events   data/synthetic/events.csv \\
        --graph    data/graphs/account_graph.pkl \\
        --output   data/processed/scored_events.json \\
        --max-cases 10

Prerequisites:
    python scripts/generate_data.py   (creates events.csv + historical_cases.json)
    python scripts/train_model.py     (creates model artifacts in data/models/)
"""

import argparse
import os
import sys
import time

# ── Ensure project root is on PYTHONPATH ─────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.ingestion.pipeline import IngestionPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ingest_and_run.py",
        description="SentinelIQ — Ingest events, score with ML, investigate with agent.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--events",
        default="data/synthetic/events.csv",
        help="Path to the events CSV file.",
    )
    parser.add_argument(
        "--graph",
        default="data/graphs/account_graph.pkl",
        help="Path to save / load the account-device-IP graph.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/scored_events.json",
        help="Path to write scored events JSON (consumed by the API).",
    )
    parser.add_argument(
        "--historical-cases",
        default="data/synthetic/historical_cases.json",
        help="Path to historical cases JSON for RAG seeding.",
    )
    parser.add_argument(
        "--embed-cases",
        action="store_true",
        default=False,
        help="Embed historical cases into ChromaDB before running the agent. "
             "Required on first run; safe to re-run (upsert is idempotent).",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=20,
        help="Maximum number of flagged events to investigate with the LangGraph agent per run.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print("  SentinelIQ — Ingestion & Agent Run")
    print(f"{'='*60}")
    print(f"  Events CSV      : {args.events}")
    print(f"  Graph output    : {args.graph}")
    print(f"  Scored output   : {args.output}")
    print(f"  Embed cases     : {args.embed_cases}")
    print(f"  Max agent cases : {args.max_cases}")
    print(f"{'='*60}\n")

    start = time.time()

    pipeline = IngestionPipeline()
    summary = pipeline.run(
        events_path=args.events,
        graph_output_path=args.graph,
        scored_output_path=args.output,
        historical_cases_path=args.historical_cases,
        embed_cases=args.embed_cases,
        max_agent_cases=args.max_cases,
    )

    elapsed = time.time() - start

    print(f"\n{'='*60}")
    print("  Pipeline Complete — Summary")
    print(f"{'='*60}")
    print(f"  Events loaded        : {summary['events_loaded']:,}")
    print(f"  Events scored        : {summary['events_scored']:,}")
    print(f"  Events flagged       : {summary['events_flagged']:,}")
    print(f"  Cases investigated   : {summary['cases_investigated']}")
    print(f"  Cases added to queue : {summary['cases_added_to_queue']}")
    print(f"  Elapsed time         : {elapsed:.1f}s")

    if summary["errors"]:
        print(f"\n  ⚠️  Errors ({len(summary['errors'])}):")
        for err in summary["errors"][:5]:   # show first 5 to avoid wall of text
            print(f"    - {err}")
        if len(summary["errors"]) > 5:
            print(f"    ... and {len(summary['errors']) - 5} more.")
    else:
        print("\n  ✅ No errors.")

    print(f"{'='*60}\n")

    if summary["cases_added_to_queue"] > 0:
        print(
            f"  {summary['cases_added_to_queue']} case(s) added to the review queue.\n"
            "  Open the dashboard to review them:\n"
            "    streamlit run src/dashboard/app.py\n"
        )

    if summary["errors"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
