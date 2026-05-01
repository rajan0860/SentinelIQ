"""
Graph View Page
===============
Visualises the Account-Device-IP relationship network using PyVis.
"""

import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network  # type: ignore
import pickle
import os
from pathlib import Path

st.set_page_config(page_title="Graph View | SentinelIQ", layout="wide")

st.title("🕸️ Relationship Graph Visualisation")
st.markdown("Interactive network showing links between Accounts, Devices, and IP Addresses.")

# Resolve path relative to this file so it works from any working directory
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
GRAPH_PATH = str(_PROJECT_ROOT / "data" / "graphs" / "account_graph.pkl")

@st.cache_data
def load_graph():
    try:
        with open(GRAPH_PATH, "rb") as f:
            G = pickle.load(f)
            return G
    except Exception as e:
        st.error(f"Failed to load graph from {GRAPH_PATH}: {e}")
        return None

G = load_graph()

if G:
    # Pyvis struggles with massive graphs in the browser. 
    # We will only render a subset (e.g., the largest connected component) or a sample.
    st.info(f"Full Graph Size: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    
    with st.spinner("Generating interactive network..."):
        # Take a subgraph of the first 200 nodes for performance
        subgraph_nodes = list(G.nodes())[:200]
        sub_g = G.subgraph(subgraph_nodes)
        
        # Initialize pyvis network
        net = Network(height="600px", width="100%", bgcolor="#0f1115", font_color="white")
        
        # Add nodes with colors based on type
        for node, data in sub_g.nodes(data=True):
            node_type = data.get("type", "unknown")
            color = "#8b949e" # default
            if node_type == "account": color = "#58a6ff"
            elif node_type == "device": color = "#d29922"
            elif node_type == "ip": color = "#f85149"
            
            net.add_node(node, label=str(node), title=f"Type: {node_type}", color=color)
            
        for source, target in sub_g.edges():
            net.add_edge(source, target, color="#30363d")
            
        # Generate HTML
        path = '/tmp/graph.html'
        net.save_graph(path)
        
        # Read HTML and render in Streamlit
        with open(path, 'r', encoding='utf-8') as f:
            html_string = f.read()
            
        components.html(html_string, height=620)
else:
    st.warning("Graph data not found. Ensure the ingestion pipeline has run.")
