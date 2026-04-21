"""
Event Loader Module
===================
Responsible for loading raw transaction data from CSV or database sources,
validating the schema, and performing basic cleaning before it is passed
to the Graph Builder or ML models.
"""

import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from typing import Union

class EventLoader:
    def __init__(self):
        # We define explicitly what data we expect. If the upstream data 
        # changes and breaks this schema, we want to fail loud and early.
        self.required_columns = [
            "event_id", 
            "account_id", 
            "timestamp", 
            "transaction_amount",
            "device_id", 
            "ip_address", 
            "is_fraud"
            # We don't necessarily abort if we are missing ML features 
            # like velocity, but we absolutely MUST have the 3 IDs above 
            # to build our NetworkX graph.
        ]

    def load_events(self, csv_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load, validate, and clean event data.
        """
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Event data file not found at: {path}")

        logger.info(f"Loading events from {path}...")
        try:
            df = pd.read_csv(path)
        except Exception as e:
            raise ValueError(f"Failed to read CSV at {path}: {str(e)}")

        self._validate_schema(df)
        df = self._clean_data(df)
        
        logger.info(f"Successfully loaded {len(df):,} clean events.")
        return df

    def _validate_schema(self, df: pd.DataFrame):
        """Ensure all required columns are present."""
        missing_columns = [col for col in self.required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"CRITICAL ERROR: Data is missing required columns: {missing_columns}")

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform basic type casting and null removal."""
        # Ensure our timestamps are actually datetime objects, not strings
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # If an event doesn't have an account_id, device_id, or IP, we can't 
        # map it in our graph. We drop these malformed rows.
        initial_count = len(df)
        df = df.dropna(subset=["account_id", "device_id", "ip_address"])
        dropped = initial_count - len(df)
        
        if dropped > 0:
            logger.warning(f"Dropped {dropped} malformed rows missing critical IDs.")
            
        return df

if __name__ == "__main__":
    # Test the loader locally
    loader = EventLoader()
    df = loader.load_events("data/synthetic/events.csv")
    print(df.info())
