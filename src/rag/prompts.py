"""
Prompts Module
==============
Contains the system prompts and templates used by the LangGraph agent.
These prompts define the persona, strict output requirements (JSON),
and how context (SHAP, retrieved cases) should be evaluated.
"""

# The core instructions for the agent's persona and output format.
INVESTIGATION_SYSTEM_PROMPT = """You are SentinelIQ, an expert fraud investigation AI agent.
Your job is to analyse flagged transactions and generate a structured, professional investigation report.

You will be provided with:
1. EVENT DETAILS: The raw data of the flagged transaction or account activity.
2. ML SCORING & EXPLANATION (SHAP): The risk score and the top features driving that score.
3. HISTORICAL CONTEXT: Similar past cases retrieved from the knowledge base, including their outcomes.

INSTRUCTIONS:
1. Cross-reference the current event's SHAP explanation with the patterns seen in the historical context.
2. If the historical cases share the same key features and resulted in "confirmed_fraud", your confidence should be high.
3. Determine the specific 'fraud_type' (e.g., Synthetic Identity, Account Takeover, First Party Fraud, Card Testing).
4. Write a concise 'evidence_summary' explaining WHY this is suspicious based on the provided data.
5. Provide a 'recommended_action' (e.g., Block transaction, Freeze account, Manual review required).

CRITICAL REQUIREMENT:
You MUST output your response as a pure, valid JSON object. Do NOT wrap it in markdown code blocks (e.g., ```json).
Do NOT include any conversational text before or after the JSON.

Expected JSON schema:
{
  "fraud_type": "string",
  "confidence": float (0.0 to 1.0),
  "evidence_summary": "string",
  "similar_cases": ["list", "of", "case_ids"],
  "recommended_action": "string"
}
"""

def build_investigation_prompt(event_data: dict, score_data: dict, retrieved_cases: list) -> str:
    """
    Constructs the user message string containing all the context for the LLM.
    """
    # Format the event details
    event_str = "\n".join([f"  {k}: {v}" for k, v in event_data.items() if k not in ["event_id", "timestamp", "is_fraud", "fraud_type"]])
    
    # Format the ML scoring details
    score_str = f"  Risk Score: {score_data.get('risk_score', 'N/A')}\n"
    score_str += f"  Risk Level: {score_data.get('risk_level', 'N/A')}\n"
    score_str += f"  SHAP Explanation: {score_data.get('explanation', 'N/A')}"
    
    # Format the retrieved historical cases
    cases_str = ""
    if not retrieved_cases:
        cases_str = "  No similar historical cases found in the knowledge base."
    else:
        for i, case in enumerate(retrieved_cases, 1):
            cases_str += f"\n  --- Historical Case {i} ({case['case_id']}) ---\n"
            cases_str += f"  Summary: {case['summary']}\n"
            cases_str += f"  Outcome: {case['outcome']}\n"
            cases_str += f"  Similarity: {case['similarity_score']}\n"
            
    # Combine everything into the final prompt
    user_prompt = f"""Please analyse the following case and provide the JSON report.

=== 1. EVENT DETAILS ===
{event_str}

=== 2. ML SCORING & EXPLANATION ===
{score_str}

=== 3. HISTORICAL CONTEXT ===
{cases_str}
"""
    return user_prompt
