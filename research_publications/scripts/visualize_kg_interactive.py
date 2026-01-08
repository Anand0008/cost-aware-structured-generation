"""
============================================================================
KNOWLEDGE GRAPH VISUALIZATION (Interactive)
============================================================================
Purpose: Generate an interactive HTML visualization of the Knowledge Graph.
To avoid browser crashes, we filter to:
1. "Core" concepts (High centrality)
2. Specific Subject clusters (e.g., Aerodynamics)

Outputs:
    - evaluation/plots/kg_interactive.html

Needs: pyvis, networkx
============================================================================
"""

import csv
import json
import networkx as nx
from pyvis.network import Network
from pathlib import Path

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent
KG_DIR = PROJECT_ROOT / "research_publications" / "knowledge_graph"
OUTPUT_DIR = PROJECT_ROOT / "research_publications" / "evaluation" / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Visualization Settings
TARGET_SUBJECT = "Aerodynamics"  # Focus on this cluster
MAX_NODES = 200  # Limit to avoid laggy HTML
MIN_DEGREE = 2   # Remove noise

def main():
    print("Loading Graph for Visualization...")
    G = nx.MultiDiGraph()
    
    # 1. Load Nodes
    with open(KG_DIR / "nodes.csv", 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_id = row['node_id']
            # Determine color/shape by type
            node_type = row['node_type']
            color = "#97c2fc" # default blue
            size = 10
            
            if node_type == "Question":
                color = "#ffb3ba" # red
                size = 15
            elif node_type == "Concept":
                color = "#bae1ff" # blue
                size = 20
            elif node_type == "Subject":
                color = "#ffdfba" # orange
                size = 30
            elif node_type == "Formula":
                color = "#baffc9" # green
                size = 15
            elif node_type == "CommonMistake":
                color = "#ffbfd3" # pink
                size = 15
                
            G.add_node(node_id, 
                      group=node_type, 
                      color=color, 
                      size=size,
                      title=f"{node_type}: {node_id}")

    # 2. Load Edges
    with open(KG_DIR / "edges.csv", 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            G.add_edge(row['source'], row['target'], 
                      title=row['edge_type'],
                      label=row['edge_type']) # Show label on hover

    print(f"Full Graph: {G.number_of_nodes()} nodes")
    
    # 3. Filter for Visualization
    # Strategy: Find the "Subject" node for Aerodynamics and get its neighbors (2 hops)
    print(f"Filtering for Subject: {TARGET_SUBJECT}...")
    
    try:
        if TARGET_SUBJECT not in G:
            print(f"WARNING: Subject '{TARGET_SUBJECT}' not found. Using Center of Graph.")
            center = nx.pagerank(G).popitem()[0] # Max rank node
        else:
            center = TARGET_SUBJECT
            
        # Get ego graph (radius 2) - Use undirected to get incoming edges (Topics -> Subject)
        print("Extracting neighborhood (undirected)...")
        subgraph = nx.ego_graph(G.to_undirected(), center, radius=2)
        
        # Further prune low-degree nodes if too big
        if subgraph.number_of_nodes() > 150:
            print(f"Subgraph too big ({subgraph.number_of_nodes()}). Keeping Top 150 by degree...")
            # Sort by degree and take top 150
            top_nodes = sorted(subgraph.nodes(), key=lambda n: subgraph.degree(n), reverse=True)[:150]
            subgraph = subgraph.subgraph(top_nodes)
            
        print(f"Visualizing {subgraph.number_of_nodes()} nodes...")
        
        # 4. Generate Pyvis Network with CDN resources
        net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white", select_menu=True, cdn_resources='remote')
        net.from_nx(subgraph)
        
        # Physics options for stability (ForceAtlas2Based is good for clusters)
        net.force_atlas_2based()
        
        out_file = OUTPUT_DIR / "kg_interactive.html"
        # Fix Unicode Error: Manually write with UTF-8
        html = net.generate_html()
        with open(out_file, 'w', encoding='utf-8') as f:
            f.write(html)
            
        print(f"Saved interactive graph to: {out_file}")
        
    except Exception as e:
        print(f"Error visualizing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
