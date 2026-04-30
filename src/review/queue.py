"""
Queue Module
============
Manages the lifecycle of cases waiting for human review.
Uses a simple JSON file for persistence during prototyping.
"""

import json
import os
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# Default path relative to project root
DEFAULT_QUEUE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "review_queue.json"
)

class ReviewQueueManager:
    """
    Manages the review queue of cases.
    """
    def __init__(self, queue_path: str = DEFAULT_QUEUE_PATH):
        self.queue_path = queue_path
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        """Creates the JSON file if it doesn't exist."""
        os.makedirs(os.path.dirname(self.queue_path), exist_ok=True)
        if not os.path.exists(self.queue_path):
            with open(self.queue_path, 'w') as f:
                json.dump([], f)
                
    def _read_queue(self) -> List[Dict]:
        """Reads the queue from disk."""
        try:
            with open(self.queue_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read queue file: {e}")
            return []

    def _write_queue(self, queue_data: List[Dict]):
        """Writes the queue to disk."""
        try:
            with open(self.queue_path, 'w') as f:
                json.dump(queue_data, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to write queue file: {e}")

    def add_case(self, case_report: dict):
        """
        Adds a new case report to the pending review queue.
        """
        queue = self._read_queue()
        
        # Check if case is already in the queue to avoid duplicates
        if not any(c.get("case_id") == case_report.get("case_id") for c in queue):
            queue.append(case_report)
            self._write_queue(queue)
            logger.info(f"Added case {case_report.get('case_id')} to review queue.")
        else:
            logger.warning(f"Case {case_report.get('case_id')} is already in the queue.")

    def get_pending_cases(self) -> List[Dict]:
        """
        Returns all cases currently waiting in the queue.
        """
        return self._read_queue()

    def get_case(self, case_id: str) -> Optional[Dict]:
        """
        Retrieves a specific case by ID.
        """
        queue = self._read_queue()
        for case in queue:
            if case.get("case_id") == case_id:
                return case
        return None

    def remove_case(self, case_id: str):
        """
        Removes a case from the queue after it has been reviewed.
        """
        queue = self._read_queue()
        filtered_queue = [c for c in queue if c.get("case_id") != case_id]
        
        if len(queue) != len(filtered_queue):
            self._write_queue(filtered_queue)
            logger.info(f"Removed case {case_id} from review queue.")
        else:
            logger.warning(f"Case {case_id} not found in queue to remove.")
