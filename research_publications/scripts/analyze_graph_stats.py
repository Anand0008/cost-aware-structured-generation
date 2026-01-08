"""
Knowledge Graph Statistics Analyzer
====================================
Analyzes the generated Knowledge Graph structure for:
- Centrality (most important nodes)
- Density / Connectivity
- Component analysis (islands vs connected web)

Usage:
    python research_publications/scripts/analyze_graph_stats.py
"""

import json
import csv
from pathlib import Path
from collections import defaultdict, Counter

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
KG_DIR = PROJECT_ROOT / "research_publications" / "knowledge_graph"
NODES_CSV = KG_DIR / "nodes.csv"
EDGES_CSV = KG_DIR / "edges.csv"
OUTPUT_JSON = KG_DIR / "graph_stats.json"
OUTPUT_MD = KG_DIR / "graph_analysis_report.md"

def load_graph():
    """Load nodes and edges from CSV files."""
    nodes = {}
    edges = []
    
    # Load nodes
    if NODES_CSV.exists():
        with open(NODES_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                node_id = row.get("id") or row.get("node_id") or row.get("ID")
                if node_id:
                    nodes[node_id] = row
    
    # Load edges
    if EDGES_CSV.exists():
        with open(EDGES_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                edges.append(row)
    
    return nodes, edges

def build_adjacency(nodes, edges):
    """Build adjacency list from edges."""
    adj = defaultdict(set)
    in_degree = Counter()
    out_degree = Counter()
    
    for edge in edges:
        source = edge.get("source") or edge.get("from") or edge.get("Source")
        target = edge.get("target") or edge.get("to") or edge.get("Target")
        
        if source and target:
            adj[source].add(target)
            adj[target].add(source)  # Undirected for connectivity
            out_degree[source] += 1
            in_degree[target] += 1
    
    return adj, in_degree, out_degree

def find_connected_components(nodes, adj):
    """Find connected components using BFS."""
    visited = set()
    components = []
    
    all_nodes = set(nodes.keys())
    # Also add nodes from adjacency that might not be in nodes.csv
    for n in adj:
        all_nodes.add(n)
        for neighbor in adj[n]:
            all_nodes.add(neighbor)
    
    for node in all_nodes:
        if node not in visited:
            component = []
            queue = [node]
            while queue:
                current = queue.pop(0)
                if current not in visited:
                    visited.add(current)
                    component.append(current)
                    for neighbor in adj.get(current, []):
                        if neighbor not in visited:
                            queue.append(neighbor)
            components.append(component)
    
    return components

def calculate_centrality(adj, in_degree, out_degree, top_n=20):
    """Calculate degree centrality and find top nodes."""
    
    # Total degree = in + out
    total_degree = Counter()
    for node in set(in_degree.keys()) | set(out_degree.keys()):
        total_degree[node] = in_degree[node] + out_degree[node]
    
    # Get top nodes
    top_nodes = total_degree.most_common(top_n)
    
    return {
        "top_by_total_degree": [{"node": n, "degree": d} for n, d in top_nodes],
        "top_by_in_degree": [{"node": n, "in_degree": d} for n, d in in_degree.most_common(top_n)],
        "top_by_out_degree": [{"node": n, "out_degree": d} for n, d in out_degree.most_common(top_n)]
    }

def analyze_edge_types(edges):
    """Count edge types (relationship types)."""
    edge_types = Counter()
    for edge in edges:
        rel_type = edge.get("type") or edge.get("relation") or edge.get("Type") or "unknown"
        edge_types[rel_type] += 1
    return dict(edge_types)

def analyze_node_types(nodes):
    """Count node types."""
    node_types = Counter()
    for node_id, node_data in nodes.items():
        node_type = node_data.get("type") or node_data.get("Type") or node_data.get("node_type") or "unknown"
        node_types[node_type] += 1
    return dict(node_types)

def calculate_stats(nodes, edges, adj, components):
    """Calculate graph statistics."""
    num_nodes = len(set(nodes.keys()) | set(adj.keys()))
    num_edges = len(edges)
    
    # Density = E / (N * (N-1))
    max_edges = num_nodes * (num_nodes - 1) if num_nodes > 1 else 1
    density = num_edges / max_edges if max_edges > 0 else 0
    
    # Component stats
    component_sizes = sorted([len(c) for c in components], reverse=True)
    largest_component = component_sizes[0] if component_sizes else 0
    num_islands = sum(1 for s in component_sizes if s == 1)
    
    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "density": round(density, 6),
        "num_components": len(components),
        "largest_component_size": largest_component,
        "largest_component_pct": round(largest_component / num_nodes * 100, 2) if num_nodes > 0 else 0,
        "isolated_nodes": num_islands,
        "avg_component_size": round(sum(component_sizes) / len(components), 2) if components else 0,
        "is_connected": len(components) == 1
    }

def generate_report(stats, centrality, node_types, edge_types):
    """Generate markdown report."""
    report = f"""# Knowledge Graph Analysis Report

## Graph Overview

| Metric | Value |
|--------|-------|
| **Total Nodes** | {stats['num_nodes']:,} |
| **Total Edges** | {stats['num_edges']:,} |
| **Graph Density** | {stats['density']:.6f} |
| **Connected Components** | {stats['num_components']:,} |
| **Largest Component** | {stats['largest_component_size']:,} ({stats['largest_component_pct']}%) |
| **Isolated Nodes** | {stats['isolated_nodes']:,} |

## Connectivity Assessment

"""
    if stats['largest_component_pct'] > 95:
        report += "✅ **Highly Connected:** The graph is essentially one connected web (>95% in largest component).\n\n"
    elif stats['largest_component_pct'] > 80:
        report += "⚠️ **Mostly Connected:** Most nodes are connected (>80%), but some islands exist.\n\n"
    else:
        report += f"❌ **Fragmented:** Only {stats['largest_component_pct']}% of nodes are in the main component.\n\n"

    report += "## Node Types\n\n| Type | Count |\n|------|-------|\n"
    for ntype, count in sorted(node_types.items(), key=lambda x: -x[1]):
        report += f"| {ntype} | {count:,} |\n"

    report += "\n## Edge Types (Relationships)\n\n| Relationship | Count |\n|--------------|-------|\n"
    for etype, count in sorted(edge_types.items(), key=lambda x: -x[1]):
        report += f"| {etype} | {count:,} |\n"

    report += "\n## Top 10 Most Central Nodes (by Degree)\n\n| Rank | Node | Degree |\n|------|------|--------|\n"
    for i, item in enumerate(centrality["top_by_total_degree"][:10], 1):
        report += f"| {i} | {item['node'][:50]}... | {item['degree']} |\n"

    return report

def main():
    print("Loading Knowledge Graph...")
    nodes, edges = load_graph()
    
    if not nodes and not edges:
        print(f"Error: No graph data found.")
        print(f"  Expected nodes at: {NODES_CSV}")
        print(f"  Expected edges at: {EDGES_CSV}")
        return
    
    print(f"Loaded {len(nodes)} nodes and {len(edges)} edges.")
    
    print("Building adjacency list...")
    adj, in_degree, out_degree = build_adjacency(nodes, edges)
    
    print("Finding connected components...")
    components = find_connected_components(nodes, adj)
    
    print("Calculating statistics...")
    stats = calculate_stats(nodes, edges, adj, components)
    centrality = calculate_centrality(adj, in_degree, out_degree)
    node_types = analyze_node_types(nodes)
    edge_types = analyze_edge_types(edges)
    
    # Compile results
    results = {
        "statistics": stats,
        "centrality": centrality,
        "node_types": node_types,
        "edge_types": edge_types
    }
    
    # Save JSON
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved stats to: {OUTPUT_JSON}")
    
    # Save Report
    report = generate_report(stats, centrality, node_types, edge_types)
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Saved report to: {OUTPUT_MD}")
    
    # Print summary
    print("\n" + "="*50)
    print("KNOWLEDGE GRAPH SUMMARY")
    print("="*50)
    print(f"Nodes:              {stats['num_nodes']:,}")
    print(f"Edges:              {stats['num_edges']:,}")
    print(f"Components:         {stats['num_components']:,}")
    print(f"Largest Component:  {stats['largest_component_pct']}%")
    print(f"Is Connected Web:   {'Yes' if stats['largest_component_pct'] > 95 else 'No'}")
    print("="*50)

if __name__ == "__main__":
    main()
