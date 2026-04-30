"""
Schemas Module
==============
Defines the Pydantic models for the human review process.
"""

from pydantic import BaseModel, Field
from typing import Literal, Optional

class ReviewDecision(BaseModel):
    """
    Represents the decision made by a human reviewer on a flagged case.
    """
    case_id: str = Field(description="The unique identifier for the investigated case.")
    decision: Literal["approve", "escalate", "dismiss"] = Field(
        description="The outcome of the review. 'approve' means confirmed fraud, 'dismiss' means false positive."
    )
    reviewer_notes: Optional[str] = Field(
        default="", 
        description="Any qualitative notes or reasoning provided by the human reviewer."
    )
    reviewer_id: Optional[str] = Field(
        default="system", 
        description="The ID or username of the person who reviewed the case."
    )
