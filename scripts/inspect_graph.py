import pickle
import networkx as nx
from pathlib import Path

def inspect():
    graph_path = Path("data/graphs/account_graph.pkl")
    if not graph_path.exists():
        print(f"Error: {graph_path} not found.")
        return
        
    print(f"Loading {graph_path}...")
    with open(graph_path, "rb") as f:
        G = pickle.load(f)
        
    print("-" * 50)
    print("Graph Overview")
    print("-" * 50)
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    
    # Let's count the types of nodes
    node_types = {}
    for node, data in G.nodes(data=True):
        ntype = data.get('node_type', 'unknown')
        node_types[ntype] = node_types.get(ntype, 0) + 1
        
    for ntype, count in node_types.items():
        print(f"{ntype.capitalize()} nodes: {count}")
        
    print("\n" + "-" * 50)
    print("Finding a Synthetic Identity Ring...")
    print("-" * 50)
    
    # A fraud ring is characterized by 1 device used by MULTIPLE accounts.
    # Let's find device nodes with a degree > 1 (connected to > 1 thing, mostly accounts)
    fraud_rings = []
    
    for node, data in G.nodes(data=True):
        if data.get('node_type') == 'device':
            # Count how many accounts connect to this device
            neighbors = list(G.neighbors(node))
            accounts_connected = [n for n in neighbors if G.nodes[n].get('node_type') == 'account']
            
            if len(accounts_connected) >= 3:
                fraud_rings.append((node, accounts_connected))
                
    if fraud_rings:
        # Let's just look at the first two rings we found
        for i, (device_node, accounts) in enumerate(fraud_rings[:2]):
            print(f"\nRing #{i+1} Structure:")
            print(f"  🏢 Device Hub: {device_node}")
            print("  ├── connected to:")
            for acc in accounts:
                print(f"  │   👤 {acc}")
            print(f"  └── Total mathematically shared: {len(accounts)} distinct accounts")
    else:
        print("No fraud rings found! (Wait, that shouldn't happen based on Phase 1...)")

if __name__ == "__main__":
    inspect()
