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
st.markdown("Ask SentinelIQ questions about historical fraud patterns and past investigations.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("E.g., What are the common indicators of synthetic identity fraud?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    # Fetch response from backend
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            try:
                response = requests.post(f"{API_URL}/query/", json={"query": prompt})
                response.raise_for_status()
                data = response.json()
                
                answer = data.get("answer", "No answer generated.")
                st.markdown(answer)
                
                # Show references in expander
                retrieved = data.get("retrieved_cases", [])
                if retrieved:
                    with st.expander("References"):
                        for c in retrieved:
                            st.write(f"**Case {c.get('case_id')}** ({c.get('outcome')})")
                            st.write(f"Similarity: {c.get('similarity_score')}")
                            st.caption(c.get('summary'))
                            st.divider()
                            
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"Failed to query knowledge base: {e}")
