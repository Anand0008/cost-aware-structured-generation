"""
KG DEMO DATA GENERATOR - FILTERED TO LAST 5 YEARS
"""

import json
import re
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent.parent
SYLLABUS_PATH = PROJECT_ROOT / "data" / "gate_ae_syllabus_structured.json"
KG_NODES_PATH = PROJECT_ROOT / "research_publications" / "knowledge_graph" / "nodes.csv"
KG_EDGES_PATH = PROJECT_ROOT / "research_publications" / "knowledge_graph" / "edges.csv"
OUTPUT_PATH = PROJECT_ROOT / "research_publications" / "demo" / "graph_data.json"

# Filter to last 5 years
MIN_YEAR = 2020
MAX_YEAR = 2026


def normalize(text: str) -> str:
    return re.sub(r'[^a-z0-9]', '', text.lower())


def extract_syllabus_items(syllabus: dict) -> list:
    items = []
    
    def recurse(obj, section="", topic="", is_core=True):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in ("note",):
                    continue
                if key.startswith("section_"):
                    parts = key.split("_")[2:]
                    section_name = " ".join(parts).title()
                    items.append({'name': section_name, 'type': 'Section', 'parent': None, 'isCore': True})
                    recurse(value, section=section_name, is_core=True)
                elif key == "core_topics":
                    recurse(value, section=section, is_core=True)
                elif key == "special_topics":
                    recurse(value, section=section, is_core=False)
                elif isinstance(value, (list, dict)):
                    topic_name = key.replace("_", " ").title()
                    items.append({'name': topic_name, 'type': 'Topic', 'parent': section, 'isCore': is_core})
                    recurse(value, section=section, topic=topic_name, is_core=is_core)
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, str):
                    items.append({'name': item.strip(), 'type': 'Concept', 'parent': topic if topic else section, 'isCore': is_core})
                else:
                    recurse(item, section=section, topic=topic, is_core=is_core)
    
    recurse(syllabus)
    return items


def get_question_year(question_id: str) -> int:
    """Extract year from GATE_YYYY_AE_QNN format"""
    match = re.search(r'GATE_(\d{4})_', question_id)
    return int(match.group(1)) if match else 0


def build_demo_data():
    print("Loading syllabus...")
    with open(SYLLABUS_PATH, 'r', encoding='utf-8') as f:
        syllabus = json.load(f)
    
    syllabus_items = extract_syllabus_items(syllabus)
    print(f"Extracted {len(syllabus_items)} syllabus items")
    
    print("Loading KG data...")
    import csv
    kg_nodes = {}
    with open(KG_NODES_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            attrs = {}
            if row.get('attributes_json'):
                try:
                    attrs = json.loads(row['attributes_json'])
                except:
                    pass
            kg_nodes[row['node_id']] = {
                'type': row['node_type'],
                'description': attrs.get('description', ''),
                'importance': attrs.get('importance', 'secondary')
            }
    
    kg_edges = []
    with open(KG_EDGES_PATH, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            kg_edges.append({'source': row['source'], 'target': row['target'], 'type': row['edge_type']})
    
    print(f"Loaded {len(kg_nodes)} nodes, {len(kg_edges)} edges")
    
    # Build index
    norm_to_id = {}
    for node_id in kg_nodes:
        norm = normalize(node_id)
        if norm not in norm_to_id:
            norm_to_id[norm] = node_id
    
    # Match syllabus
    demo_nodes = []
    node_ids = set()
    syllabus_node_ids = set()
    
    # Track hierarchy for metadata
    # item['parent'] is useful, but we also want explicit 'subject' and 'topic' on nodes if possible
    # We'll traverse up parents to find them later or just use the structure we have.
    
    for item in syllabus_items:
        name = item['name']
        norm = normalize(name)
        kg_match = norm_to_id.get(norm)
        if not kg_match:
            for kg_norm, kg_id in norm_to_id.items():
                if len(norm) > 5 and (norm in kg_norm or kg_norm in norm):
                    kg_match = kg_id
                    break
        
        node_id = kg_match if kg_match else name
        if node_id not in node_ids:
            color = '#FFD700' if item['isCore'] else '#FFA500'
            node_data = {
                'id': node_id, 
                'name': name, # Display name
                'kg_id': node_id, # Real ID
                'type': item['type'],
                'isSyllabus': True, 
                'isCore': item['isCore'],
                'parent': item['parent'], 
                'color': color
            }
            demo_nodes.append(node_data)
            node_ids.add(node_id)
            syllabus_node_ids.add(node_id)
    
    print(f"Matched: {len([n for n in demo_nodes if n['id'] != n['name']])}/{len(syllabus_items)} syllabus items")
    
    # Hierarchy edges
    syllabus_name_to_id = {n['name']: n['id'] for n in demo_nodes if n.get('isSyllabus')}
    hierarchy_edges = []
    for node in demo_nodes:
        parent_name = node.get('parent')
        if parent_name and parent_name in syllabus_name_to_id:
            parent_id = syllabus_name_to_id[parent_name]
            if parent_id in node_ids:
                hierarchy_edges.append({'source': parent_id, 'target': node['id'], 'type': 'CONTAINS'})
    
    print(f"Hierarchy edges: {len(hierarchy_edges)}")
    
    # Include ALL Questions (2007-2025)
    all_questions = [nid for nid, info in kg_nodes.items() if info['type'] == 'Question']
    print(f"Total Questions: {len(all_questions)}")
    
    for node_id in all_questions:
        if node_id not in node_ids:
            year = get_question_year(node_id)
            info = kg_nodes.get(node_id, {})
            demo_nodes.append({
                'id': node_id, 
                'name': node_id, 
                'type': 'Question',
                'year': year,
                'description': info.get('description', ''),
                'isSyllabus': False, 
                'isCore': False, 
                'color': '#ff6b6b'
            })
            node_ids.add(node_id)
    
    # Find Concepts/Formulas connected to our nodes
    connected_by_type = {'Formula': set(), 'Concept': set()}
    for edge in kg_edges:
        src, tgt = edge['source'], edge['target']
        if src in node_ids and tgt in kg_nodes and tgt not in node_ids:
            ntype = kg_nodes[tgt]['type']
            if ntype in connected_by_type:
                connected_by_type[ntype].add(tgt)
        if tgt in node_ids and src in kg_nodes and src not in node_ids:
            ntype = kg_nodes[src]['type']
            if ntype in connected_by_type:
                connected_by_type[ntype].add(src)
    
    print(f"Connected: {len(connected_by_type['Formula'])} F, {len(connected_by_type['Concept'])} C")
    
    # Include ALL connected Concepts/Formulas (no limit - Questions need their full context)
    nodes_to_add = list(connected_by_type['Formula']) + list(connected_by_type['Concept'])
    
    for node_id in nodes_to_add:
        if node_id not in node_ids:
            info = kg_nodes.get(node_id, {})
            ntype = info.get('type', 'Concept')
            color = {'Concept': '#4dabf7', 'Formula': '#69db7c'}.get(ntype, '#868e96')
            demo_nodes.append({
                'id': node_id, 
                'name': node_id[:60], 
                'type': ntype,
                'description': info.get('description', ''),
                'importance': info.get('importance', ''),
                'isSyllabus': False, 
                'isCore': False, 
                'color': color
            })
            node_ids.add(node_id)
    
    print(f"Total nodes: {len(demo_nodes)}")
    
    # Include all edges between demo nodes
    demo_edges = [e for e in kg_edges if e['source'] in node_ids and e['target'] in node_ids]
    demo_edges.extend(hierarchy_edges)
    print(f"Total edges: {len(demo_edges)}")
    
    # Verify connectivity
    connected = set()
    for e in demo_edges:
        connected.add(e['source'])
        connected.add(e['target'])
    print(f"Connected nodes: {len(connected)} / {len(node_ids)}")
    
    output = {
        'nodes': demo_nodes,
        'links': demo_edges,
        'syllabusCount': len(syllabus_node_ids),
        'sections': [
            {'name': 'Engineering Mathematics', 'id': 'Engineering Mathematics'},
            {'name': 'Flight Mechanics', 'id': 'Flight Mechanics'},
            {'name': 'Space Dynamics', 'id': 'Space Dynamics'},
            {'name': 'Aerodynamics', 'id': 'Aerodynamics'},
            {'name': 'Structures', 'id': 'Structures'},
            {'name': 'Propulsion', 'id': 'Propulsion'}
        ]
    }
    
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    build_demo_data()
