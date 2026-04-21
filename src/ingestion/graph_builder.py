"""
Graph Builder Module
====================
Constructs an undirected NetworkX graph from transactional data.
Nodes represent Accounts, Devices, and IP addresses. 
Edges represent the relationship "Account used Device" or "Account used IP".
"""

import networkx as nx
import pandas as pd
import pickle
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class GraphBuilder:
    def __init__(self):
        # We use an undirected graph because sharing is symmetric.
        # Account A <--> Device <--> Account B
        self.graph = nx.Graph()

    def build_from_dataframe(self, df: pd.DataFrame) -> nx.Graph:
        """
        Parses the events DataFrame to populate the relationship graph.
        """
        logger.info(f"Building NetworkX graph from {len(df):,} events...")
        
        # Performance optimization:
        # If an account uses the same device 50 times, that is still just ONE
        # structural relationship in the network. By dropping duplicates first,
        # we radically speed up the graph building process.
        unique_device_links = df[['account_id', 'device_id']].drop_duplicates()
        unique_ip_links = df[['account_id', 'ip_address']].drop_duplicates()

        logger.info(f"Adding {len(unique_device_links):,} unique Account-Device edges...")
        for _, row in unique_device_links.iterrows():
            # We prefix the nodes so an account '123' and a device '123' 
            # don't accidentally merge into the same node in the graph.
            acc_node = f"ACC:{row['account_id']}"
            dev_node = f"DEV:{row['device_id']}"
            
            # networkx automatically creates nodes if they don't exist when adding an edge
            self.graph.add_node(acc_node, node_type='account')
            self.graph.add_node(dev_node, node_type='device')
            self.graph.add_edge(acc_node, dev_node, edge_type='uses_device')

        logger.info(f"Adding {len(unique_ip_links):,} unique Account-IP edges...")
        for _, row in unique_ip_links.iterrows():
            acc_node = f"ACC:{row['account_id']}"
            ip_node = f"IP:{row['ip_address']}"
            
            self.graph.add_node(acc_node, node_type='account') # Safe; updates existing
            self.graph.add_node(ip_node, node_type='ip')
            self.graph.add_edge(acc_node, ip_node, edge_type='uses_ip')

        num_nodes = self.graph.number_of_nodes()
        num_edges = self.graph.number_of_edges()
        logger.info(f"Graph complete! Total Nodes: {num_nodes:,} | Total Edges: {num_edges:,}")
        
        return self.graph

    def save_graph(self, output_path: str):
        """Serialize the graph to disk for downstream feature extraction."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving graph to {path}...")
        with open(path, 'wb') as f:
            pickle.dump(self.graph, f)
        logger.info("Save complete.")

if __name__ == "__main__":
    import os
    import sys
    # Add project root to PYTHONPATH so we can resolve 'from src...'
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    
    from src.ingestion.event_loader import EventLoader
    
    # 1. Load the data using our Step 1 class
    loader = EventLoader()
    df = loader.load_events("data/synthetic/events.csv")
    
    # 2. Build the graph using our Step 2 class
    builder = GraphBuilder()
    G = builder.build_from_dataframe(df)
    
    # 3. Save it
    builder.save_graph("data/graphs/account_graph.pkl")
