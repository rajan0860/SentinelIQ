"""
Review Queue Page
=================
The primary workspace for human analysts to review LangGraph investigation reports.
"""

import streamlit as st
import requests

st.set_page_config(page_title="Review Queue | SentinelIQ", layout="wide")

API_URL = "http://localhost:8000"

st.title("⚖️ Case Review Queue")
st.markdown("Review agent investigations and submit decisions to train the system.")

def fetch_cases():
    try:
        res = requests.get(f"{API_URL}/cases/")
        res.raise_for_status()
        return res.json()
    except Exception as e:
        st.error(f"Failed to fetch cases: {e}")
        return []

cases = fetch_cases()

if not cases:
    st.success("🎉 The queue is empty. All cases have been reviewed!")
else:
    st.info(f"{len(cases)} cases pending review.")
    
    for case in cases:
        with st.expander(f"Case: {case.get('case_id')} | {case.get('fraud_type')} (Confidence: {case.get('confidence')})"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Agent Evidence Summary")
                st.info(case.get("evidence_summary"))
                
                st.subheader("Similar Historical Cases")
                st.write(case.get("similar_cases", []))
                
            with col2:
                st.subheader("Action Required")
                st.warning(case.get("recommended_action"))
                
                st.divider()
                st.write("**Submit Review Decision**")
                
                notes = st.text_area("Reviewer Notes", key=f"notes_{case['case_id']}")
                
                c1, c2, c3 = st.columns(3)
                
                def submit(decision_val, cid):
                    payload = {
                        "case_id": cid,
                        "decision": decision_val,
                        "reviewer_notes": st.session_state[f"notes_{cid}"],
                        "reviewer_id": "human_analyst"
                    }
                    try:
                        r = requests.post(f"{API_URL}/cases/{cid}/review", json=payload)
                        r.raise_for_status()
                        st.success(f"Decision logged for {cid}!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error submitting decision: {e}")

                if c1.button("✅ Approve", key=f"app_{case['case_id']}", help="Confirm as Fraud"):
                    submit("approve", case['case_id'])
                if c2.button("⚠️ Escalate", key=f"esc_{case['case_id']}", help="Needs deeper investigation"):
                    submit("escalate", case['case_id'])
                if c3.button("❌ Dismiss", key=f"dis_{case['case_id']}", help="Mark as False Positive"):
                    submit("dismiss", case['case_id'])
