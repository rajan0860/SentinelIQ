"""
Ingestion Pipeline Module
=========================
Unified orchestrator that runs the full SentinelIQ processing flow:

    EventLoader → GraphBuilder → GraphFeatureExtractor
        → FraudScorer → InvestigationAgent → ReviewQueueManager

On first run (or when --embed-cases is passed), it also bootstraps the
ChromaDB vector store by embedding historical case documents.

Usage (as module):
    from src.ingestion.pipeline import IngestionPipeline

    pipeline = IngestionPipeline()
    summary = pipeline.run(
        events_path="data/synthetic/events.csv",
        graph_output_path="data/graphs/account_graph.pkl",
        scored_output_path="data/processed/scored_events.json",
        embed_cases=True,
    )

Usage (via scripts/ingest_and_run.py):
    python scripts/ingest_and_run.py
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.ingestion.event_loader import EventLoader
from src.ingestion.graph_builder import GraphBuilder
from src.ml.graph_features import GraphFeatureExtractor
from src.ml.scorer import FraudScorer
from src.agent.graph import InvestigationAgent
from src.review.queue import ReviewQueueManager
from src.rag.vector_store import upsert_historical_cases

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Default paths — all relative to project root
_DEFAULT_EVENTS_PATH        = "data/synthetic/events.csv"
_DEFAULT_GRAPH_PATH         = "data/graphs/account_graph.pkl"
_DEFAULT_SCORED_OUTPUT_PATH = "data/processed/scored_events.json"
_DEFAULT_HISTORICAL_CASES   = "data/synthetic/historical_cases.json"
_DEFAULT_XGB_PATH           = os.getenv("XGB_MODEL_PATH", "data/models/xgboost_fraud.json")
_DEFAULT_ISO_PATH           = os.getenv("ISO_MODEL_PATH", "data/models/isolation_forest.pkl")
_DEFAULT_FEATURE_NAMES_PATH = "data/models/feature_names.pkl"

_RISK_THRESHOLD_HIGH     = float(os.getenv("RISK_THRESHOLD_HIGH", "0.75"))
_RISK_THRESHOLD_CRITICAL = float(os.getenv("RISK_THRESHOLD_CRITICAL", "0.90"))


class IngestionPipeline:
    """
    Orchestrates the full ingestion, scoring, and investigation flow.

    Designed to be called either on a schedule (AGENT_RUN_INTERVAL) or
    triggered manually via the POST /ingest API endpoint.
    """

    def __init__(self):
        self.loader        = EventLoader()
        self.queue_manager = ReviewQueueManager()

        # Lazy-initialised — only loaded when run() is called so the class
        # can be imported without requiring model artifacts to exist yet.
        self._scorer: Optional[FraudScorer]          = None
        self._agent:  Optional[InvestigationAgent]   = None
        self._extractor: Optional[GraphFeatureExtractor] = None

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_scorer(self) -> FraudScorer:
        if self._scorer is None:
            logger.info("Loading FraudScorer (XGBoost + Isolation Forest + SHAP)...")
            self._scorer = FraudScorer(
                xgb_path=_DEFAULT_XGB_PATH,
                iso_path=_DEFAULT_ISO_PATH,
                feature_names_path=_DEFAULT_FEATURE_NAMES_PATH,
            )
        return self._scorer

    def _load_agent(self) -> InvestigationAgent:
        if self._agent is None:
            logger.info("Compiling LangGraph InvestigationAgent...")
            self._agent = InvestigationAgent()
        return self._agent

    def _load_extractor(self, graph_path: str) -> GraphFeatureExtractor:
        if self._extractor is None:
            logger.info("Loading GraphFeatureExtractor...")
            self._extractor = GraphFeatureExtractor(graph_path)
        return self._extractor

    def _build_feature_dict(self, row: pd.Series, graph_features: dict) -> dict:
        """
        Merge a raw event row with its graph-derived features into the
        flat dict that FraudScorer.score_event() expects.
        """
        return {
            "transaction_amount":      float(row.get("transaction_amount", 0)),
            "account_age_days":        float(row.get("account_age_days", 0)),
            "device_change_count":     float(row.get("device_change_count", 0)),
            "ip_country_mismatch":     float(row.get("ip_country_mismatch", 0)),
            "velocity_1hr":            float(row.get("velocity_1hr", 0)),
            "avg_txn_amount_30d":      float(row.get("avg_txn_amount_30d", 0)),
            "failed_login_count_24hr": float(row.get("failed_login_count_24hr", 0)),
            # Graph features
            "degree_centrality":       graph_features.get("degree_centrality", 0.0),
            "component_size":          float(graph_features.get("component_size", 0)),
            "shared_device_count":     float(graph_features.get("shared_device_count", 0)),
            "ip_reuse_count":          float(graph_features.get("ip_reuse_count", 0)),
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        events_path: str            = _DEFAULT_EVENTS_PATH,
        graph_output_path: str      = _DEFAULT_GRAPH_PATH,
        scored_output_path: str     = _DEFAULT_SCORED_OUTPUT_PATH,
        historical_cases_path: str  = _DEFAULT_HISTORICAL_CASES,
        embed_cases: bool           = False,
        max_agent_cases: int        = 20,
    ) -> dict:
        """
        Execute the full ingestion pipeline.

        Args:
            events_path:           Path to events CSV.
            graph_output_path:     Where to save / load the NetworkX graph.
            scored_output_path:    Where to write scored events JSON for the API.
            historical_cases_path: Path to historical_cases.json for RAG seeding.
            embed_cases:           If True, upsert historical cases into ChromaDB
                                   before running the agent. Set True on first run.
            max_agent_cases:       Cap on how many flagged events the LangGraph
                                   agent investigates per run (LLM calls are slow).

        Returns:
            Summary dict with counts for monitoring / API response.
        """
        logger.info("=" * 60)
        logger.info("  SentinelIQ Ingestion Pipeline — START")
        logger.info("=" * 60)

        summary = {
            "events_loaded":    0,
            "events_scored":    0,
            "events_flagged":   0,
            "cases_investigated": 0,
            "cases_added_to_queue": 0,
            "errors":           [],
        }

        # ── Step 1: Load events ───────────────────────────────────────────────
        logger.info("STEP 1: Loading events...")
        try:
            df = self.loader.load_events(events_path)
            summary["events_loaded"] = len(df)
        except Exception as e:
            msg = f"Failed to load events: {e}"
            logger.error(msg)
            summary["errors"].append(msg)
            return summary

        # ── Step 2: Build / refresh the account graph ─────────────────────────
        logger.info("STEP 2: Building account-device-IP graph...")
        try:
            builder = GraphBuilder()
            builder.build_from_dataframe(df)
            builder.save_graph(graph_output_path)
        except Exception as e:
            msg = f"Graph build failed: {e}"
            logger.error(msg)
            summary["errors"].append(msg)
            return summary

        # ── Step 3: Load graph feature extractor ──────────────────────────────
        logger.info("STEP 3: Loading graph feature extractor...")
        try:
            extractor = self._load_extractor(graph_output_path)
        except Exception as e:
            msg = f"GraphFeatureExtractor init failed: {e}"
            logger.error(msg)
            summary["errors"].append(msg)
            return summary

        # ── Step 4: Load scorer ───────────────────────────────────────────────
        logger.info("STEP 4: Loading FraudScorer...")
        try:
            scorer = self._load_scorer()
        except Exception as e:
            msg = f"FraudScorer init failed: {e}"
            logger.error(msg)
            summary["errors"].append(msg)
            return summary

        # ── Step 5: Score every event ─────────────────────────────────────────
        logger.info(f"STEP 5: Scoring {len(df):,} events...")
        scored_events = []
        flagged_events = []

        # Build account-level graph feature cache to avoid redundant lookups
        account_graph_cache: dict = {}
        for account_id in df["account_id"].unique():
            account_graph_cache[account_id] = extractor.extract(str(account_id))

        for _, row in df.iterrows():
            try:
                graph_feats = account_graph_cache.get(row["account_id"], {})
                feature_dict = self._build_feature_dict(row, graph_feats)
                score_result = scorer.score_event(feature_dict)

                scored_event = {
                    "event_id":           str(row.get("event_id", "")),
                    "account_id":         str(row.get("account_id", "")),
                    "timestamp":          str(row.get("timestamp", "")),
                    "transaction_amount": float(row.get("transaction_amount", 0)),
                    "risk_score":         score_result["risk_score"],
                    "risk_level":         score_result["risk_level"],
                    "xgb_prob":           score_result["xgb_prob"],
                    "iso_anom":           score_result["iso_anom"],
                    "flags":              score_result["flags"],
                    "explanation":        score_result.get("explanation", ""),
                }
                scored_events.append(scored_event)

                if score_result["risk_score"] >= _RISK_THRESHOLD_HIGH:
                    flagged_events.append({
                        "event_id":    scored_event["event_id"],
                        "event_data":  feature_dict,
                        "ml_score":    score_result,
                    })

            except Exception as e:
                logger.warning(f"Scoring failed for event {row.get('event_id', '?')}: {e}")
                summary["errors"].append(str(e))

        summary["events_scored"]  = len(scored_events)
        summary["events_flagged"] = len(flagged_events)
        logger.info(
            f"Scoring complete — {len(scored_events):,} scored, "
            f"{len(flagged_events):,} flagged (≥ {_RISK_THRESHOLD_HIGH})"
        )

        # ── Step 6: Persist raw events and scored events for the API ─────────────
        logger.info("STEP 6: Saving events to disk...")
        try:
            # Save raw events to data/raw/ for audit trail
            raw_path = Path("data/raw") / f"events_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(raw_path, index=False)
            logger.info(f"Raw events saved → {raw_path}")

            # Save scored events for the API
            out_path = Path(scored_output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            # Sort by risk score descending so the API returns highest-risk first
            scored_events.sort(key=lambda x: x["risk_score"], reverse=True)
            with open(out_path, "w") as f:
                json.dump(scored_events, f, indent=2, default=str)
            logger.info(f"Scored events saved → {out_path}")
        except Exception as e:
            msg = f"Failed to save events: {e}"
            logger.error(msg)
            summary["errors"].append(msg)

        # ── Step 7: Embed historical cases (first-run bootstrap) ──────────────
        if embed_cases:
            logger.info("STEP 7: Embedding historical cases into ChromaDB...")
            try:
                upsert_historical_cases(historical_cases_path)
                logger.info("Historical cases embedded successfully.")
            except Exception as e:
                msg = f"Historical case embedding failed: {e}"
                logger.error(msg)
                summary["errors"].append(msg)
        else:
            logger.info("STEP 7: Skipping historical case embedding (embed_cases=False).")

        # ── Step 8: Run LangGraph agent on flagged events ─────────────────────
        agent_candidates = flagged_events[:max_agent_cases]
        logger.info(
            f"STEP 8: Running LangGraph agent on "
            f"{len(agent_candidates)} / {len(flagged_events)} flagged events "
            f"(cap: {max_agent_cases})..."
        )

        if agent_candidates:
            try:
                agent = self._load_agent()
            except Exception as e:
                msg = f"Agent init failed: {e}"
                logger.error(msg)
                summary["errors"].append(msg)
                agent = None

            if agent:
                for item in agent_candidates:
                    try:
                        report = agent.investigate(
                            event_id=item["event_id"],
                            event_data=item["event_data"],
                            ml_score=item["ml_score"],
                        )
                        if report:
                            self.queue_manager.add_case(report)
                            summary["cases_investigated"]    += 1
                            summary["cases_added_to_queue"]  += 1
                    except Exception as e:
                        logger.warning(
                            f"Agent investigation failed for {item['event_id']}: {e}"
                        )
                        summary["errors"].append(str(e))

        # ── Done ──────────────────────────────────────────────────────────────
        logger.info("=" * 60)
        logger.info("  SentinelIQ Ingestion Pipeline — COMPLETE")
        logger.info(f"  Events loaded      : {summary['events_loaded']:,}")
        logger.info(f"  Events scored      : {summary['events_scored']:,}")
        logger.info(f"  Events flagged     : {summary['events_flagged']:,}")
        logger.info(f"  Cases investigated : {summary['cases_investigated']}")
        logger.info(f"  Cases in queue     : {summary['cases_added_to_queue']}")
        if summary["errors"]:
            logger.warning(f"  Errors             : {len(summary['errors'])}")
        logger.info("=" * 60)

        return summary


# ─── Quick verification ───────────────────────────────────────────────────────
if __name__ == "__main__":
    pipeline = IngestionPipeline()
    result = pipeline.run(embed_cases=True, max_agent_cases=3)
    print("\nPipeline Summary:")
    for k, v in result.items():
        print(f"  {k:<28} {v}")
