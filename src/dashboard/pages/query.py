"""
Query Page
==========
Natural Language interface for interacting with the fraud knowledge base.
"""

import streamlit as st
import requests

st.set_page_config(page_title="Knowledge Base | SentinelIQ", layout="wide")

API_URL = "http://localhost:8000"

st.title("🧠 Knowledge Base Query")
st.markdown(
    "Ask SentinelIQ questions about historical fraud patterns and past investigations. "
    "Answers are grounded in the RAG knowledge base — the system will say so if it doesn't know."
)

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar: example prompts + clear button
with st.sidebar:
    st.subheader("Example Queries")
    examples = [
        "What are the common indicators of synthetic identity fraud?",
        "How do account takeover attacks typically present?",
        "What actions were taken on confirmed fraud cases?",
        "Show me cases involving IP country mismatches.",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex[:20]}"):
            st.session_state._prefill = ex

    st.divider()
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle prefilled prompt from sidebar example buttons
prefill = st.session_state.pop("_prefill", None)

# Chat input
prompt = st.chat_input("E.g., What are the common indicators of synthetic identity fraud?")
if prefill and not prompt:
    prompt = prefill

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🤔 Searching knowledge base and generating response (may take 10–30 s)..."):
            try:
                response = requests.post(
                    f"{API_URL}/query/",
                    json={"query": prompt},
                    timeout=60,
                )
                response.raise_for_status()
                data = response.json()

                answer = data.get("answer", "No answer generated.")
                st.markdown(answer)

                retrieved = data.get("retrieved_cases", [])
                if retrieved:
                    with st.expander(f"📚 {len(retrieved)} source case(s) retrieved"):
                        for c in retrieved:
                            st.markdown(
                                f"**{c.get('case_id')}** — {c.get('fraud_type', 'Unknown')} "
                                f"| Outcome: `{c.get('outcome')}` "
                                f"| Similarity: `{c.get('similarity_score', 0):.2f}`"
                            )
                            st.caption(c.get("summary", ""))
                            st.divider()

                st.session_state.messages.append({"role": "assistant", "content": answer})

            except requests.exceptions.ConnectionError:
                st.error("🚨 Could not connect to the FastAPI backend. Is it running on port 8000?")
            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out. The LLM may be slow — try again in a moment.")
            except Exception as e:
                st.error(f"Failed to query knowledge base: {e}")
