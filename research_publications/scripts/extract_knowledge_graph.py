"""
============================================================================
KNOWLEDGE GRAPH EXTRACTION
============================================================================
Purpose: Extract a real Knowledge Graph from the synthesized pipeline outputs
No API calls needed - purely parses existing JSON files

Outputs:
    - nodes.csv: All entities (concepts, topics, subjects, questions, etc.)
    - edges.csv: All relationships (REQUIRES, ENABLES, TESTS, etc.)
    - kg_statistics.json: Summary statistics

Author: Anand Wankhade
============================================================================
"""

import os
import json
import csv
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple
from collections import defaultdict
import re

# Configuration
# research_publications/scripts/ -> research_publications/ -> qbt/
PROJECT_ROOT = Path(__file__).parent.parent.parent
VOTING_ENGINE_DIR = PROJECT_ROOT / "debug_outputs" / "voting_engine"
OUTPUT_DIR = PROJECT_ROOT / "research_publications" / "knowledge_graph"


class KnowledgeGraphExtractor:
    """Extract Knowledge Graph from synthesized pipeline outputs"""
    
    def __init__(self, input_dir: Path, output_dir: Path):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Node storage: {node_type: {node_id: node_attributes}}
        self.nodes = {
            'Question': {},
            'Concept': {},
            'Topic': {},
            'Subject': {},
            'Formula': {},
            'Book': {},
            'Video': {},
            'CommonMistake': {},
            'Mnemonic': {},
            'DifficultyLevel': {}
        }
        
        # Edge storage: [(source_id, edge_type, target_id, attributes)]
        self.edges = []
        
        # Statistics
        self.stats = defaultdict(int)
    
    def extract_all(self) -> Dict:
        """Main extraction method"""
        print("=" * 60)
        print("KNOWLEDGE GRAPH EXTRACTION")
        print("=" * 60)
        
        # Find all final JSON files
        final_jsons = list(self.input_dir.glob("*_03_final_json.json"))
        print(f"\nFound {len(final_jsons)} synthesized outputs")
        
        for i, json_path in enumerate(final_jsons):
            if (i + 1) % 100 == 0:
                print(f"  Processing {i + 1}/{len(final_jsons)}...")
            
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self._extract_from_question(data)
            except Exception as e:
                print(f"  Error processing {json_path.name}: {e}")
                self.stats['errors'] += 1
        
        # Save outputs
        self._save_nodes()
        self._save_edges()
        self._save_statistics()
        
        print("\n" + "=" * 60)
        print("EXTRACTION COMPLETE")
        print("=" * 60)
        print(f"\nNodes:")
        for node_type, nodes in self.nodes.items():
            if nodes:
                print(f"  {node_type}: {len(nodes)}")
        print(f"\nEdges: {len(self.edges)}")
        print(f"\nOutput saved to: {self.output_dir}")
        
        return self.stats
    
    def _extract_from_question(self, data: Dict):
        """Extract all nodes and edges from a single question"""
        question_id = data.get('question_id', 'unknown')
        
        # 1. Add Question node
        self.nodes['Question'][question_id] = {
            'year': data.get('year'),
            'marks': data.get('marks'),
            'type': data.get('question_type'),
            'subject': data.get('subject', 'Aerospace Engineering')
        }
        self.stats['questions'] += 1
        
        # 2. Extract from tier_1_core_research
        tier1 = data.get('tier_1_core_research', {})
        if tier1:
            self._extract_hierarchical_tags(question_id, tier1)
            self._extract_prerequisites(question_id, tier1)
            self._extract_formulas(question_id, tier1)
            self._extract_textbook_refs(question_id, tier1)
            self._extract_video_refs(question_id, tier1)
            self._extract_difficulty(question_id, tier1)
        
        # 3. Extract from tier_2_student_learning
        tier2 = data.get('tier_2_student_learning', {})
        if tier2:
            self._extract_common_mistakes(question_id, tier2)
            self._extract_mnemonics(question_id, tier2)
    
    def _extract_hierarchical_tags(self, question_id: str, tier1: Dict):
        """Extract subject → topic → concept hierarchy"""
        tags = tier1.get('hierarchical_tags', {})
        
        # Subject
        subject_data = tags.get('subject', {})
        if isinstance(subject_data, dict):
            subject_name = subject_data.get('name', 'Unknown')
        else:
            subject_name = str(subject_data) if subject_data else 'Unknown'
        
        if subject_name and subject_name != 'Unknown':
            self.nodes['Subject'][subject_name] = {'confidence': subject_data.get('confidence', 0.0) if isinstance(subject_data, dict) else 0.0}
        
        # Topic
        topic_data = tags.get('topic', {})
        if isinstance(topic_data, dict):
            topic_name = topic_data.get('name', '')
            syllabus_ref = topic_data.get('syllabus_ref', '')
        else:
            topic_name = str(topic_data) if topic_data else ''
            syllabus_ref = ''
        
        if topic_name:
            self.nodes['Topic'][topic_name] = {'syllabus_ref': syllabus_ref}
            # Topic -> Subject edge
            if subject_name and subject_name != 'Unknown':
                self.edges.append((topic_name, 'PART_OF', subject_name, {}))
        
        # Concepts
        concepts = tags.get('concepts', [])
        if isinstance(concepts, list):
            for concept in concepts:
                if isinstance(concept, dict):
                    concept_name = concept.get('name', '')
                    importance = concept.get('importance', 'secondary')
                    if concept_name:
                        self.nodes['Concept'][concept_name] = {'importance': importance}
                        # Question -> Concept edge
                        self.edges.append((question_id, 'TESTS', concept_name, {'importance': importance}))
                        # Concept -> Topic edge
                        if topic_name:
                            self.edges.append((concept_name, 'BELONGS_TO', topic_name, {}))
                        self.stats['concepts'] += 1
    
    def _extract_prerequisites(self, question_id: str, tier1: Dict):
        """Extract prerequisite relationships - THE GOLD!"""
        prereqs = tier1.get('prerequisites', {})
        
        # Essential prerequisites
        essential = prereqs.get('essential', [])
        if isinstance(essential, list):
            for prereq in essential:
                if isinstance(prereq, str) and prereq:
                    prereq_clean = prereq.strip()
                    self.nodes['Concept'][prereq_clean] = self.nodes['Concept'].get(prereq_clean, {'importance': 'prerequisite'})
                    self.edges.append((question_id, 'REQUIRES', prereq_clean, {'type': 'essential'}))
                    self.stats['prerequisite_edges'] += 1
        
        # Helpful prerequisites
        helpful = prereqs.get('helpful', [])
        if isinstance(helpful, list):
            for prereq in helpful:
                if isinstance(prereq, str) and prereq:
                    prereq_clean = prereq.strip()
                    self.nodes['Concept'][prereq_clean] = self.nodes['Concept'].get(prereq_clean, {'importance': 'helpful'})
                    self.edges.append((question_id, 'BENEFITS_FROM', prereq_clean, {'type': 'helpful'}))
        
        # Dependency tree - EXTREMELY VALUABLE
        dep_tree = prereqs.get('dependency_tree', {})
        if isinstance(dep_tree, dict):
            for concept, relations in dep_tree.items():
                if isinstance(relations, list):
                    for rel in relations:
                        if isinstance(rel, str):
                            if rel.startswith('requires:'):
                                prereq = rel.replace('requires:', '').strip()
                                if prereq:
                                    self.nodes['Concept'][prereq] = self.nodes['Concept'].get(prereq, {'importance': 'foundational'})
                                    self.edges.append((concept, 'REQUIRES', prereq, {'source': 'dependency_tree'}))
                                    self.stats['requires_edges'] += 1
                            elif rel.startswith('enables:'):
                                enabled = rel.replace('enables:', '').strip()
                                if enabled:
                                    self.nodes['Concept'][enabled] = self.nodes['Concept'].get(enabled, {'importance': 'advanced'})
                                    self.edges.append((concept, 'ENABLES', enabled, {'source': 'dependency_tree'}))
                                    self.stats['enables_edges'] += 1
    
    def _extract_formulas(self, question_id: str, tier1: Dict):
        """Extract formula relationships"""
        # From formulas_used
        formulas_used = tier1.get('explanation', {}).get('formulas_used', [])
        if isinstance(formulas_used, list):
            for formula in formulas_used:
                if formula and isinstance(formula, str):
                    formula_id = self._clean_formula_id(formula)
                    if formula_id:
                        self.nodes['Formula'][formula_id] = {'latex': formula}
                        self.edges.append((question_id, 'USES_FORMULA', formula_id, {}))
                        self.stats['formulas'] += 1
        
        # From formulas_principles
        principles = tier1.get('formulas_principles', [])
        if isinstance(principles, list):
            for principle in principles:
                if isinstance(principle, dict):
                    formula_name = principle.get('name', '')
                    formula_latex = principle.get('formula', '')
                    if formula_name:
                        self.nodes['Formula'][formula_name] = {
                            'latex': formula_latex,
                            'type': principle.get('type', 'formula')
                        }
                        self.edges.append((question_id, 'USES_FORMULA', formula_name, {}))
    
    def _extract_textbook_refs(self, question_id: str, tier1: Dict):
        """Extract textbook references"""
        refs = tier1.get('textbook_references', [])
        if isinstance(refs, list):
            for ref in refs:
                if isinstance(ref, dict):
                    book_name = ref.get('book', '')
                    author = ref.get('author', '')
                    if book_name:
                        book_id = f"{book_name} ({author})" if author else book_name
                        self.nodes['Book'][book_id] = {
                            'author': author,
                            'chapter': ref.get('chapter_number', ''),
                            'section': ref.get('section', '')
                        }
                        self.edges.append((question_id, 'REFERENCES', book_id, {
                            'chapter': ref.get('chapter_title', ''),
                            'pages': ref.get('page_range', ''),
                            'relevance': ref.get('relevance_score', 0.0)
                        }))
                        self.stats['book_refs'] += 1
    
    def _extract_video_refs(self, question_id: str, tier1: Dict):
        """Extract video references"""
        refs = tier1.get('video_references', [])
        if isinstance(refs, list):
            for ref in refs:
                if isinstance(ref, dict):
                    video_url = ref.get('video_url', '')
                    professor = ref.get('professor', 'Unknown')
                    if video_url:
                        video_id = f"{professor}: {ref.get('topic_covered', 'Unknown')}"[:100]
                        self.nodes['Video'][video_id] = {
                            'url': video_url,
                            'professor': professor,
                            'timestamp': f"{ref.get('timestamp_start', '')} - {ref.get('timestamp_end', '')}"
                        }
                        self.edges.append((question_id, 'HAS_VIDEO', video_id, {
                            'relevance': ref.get('relevance_score', 0.0)
                        }))
                        self.stats['video_refs'] += 1
    
    def _extract_difficulty(self, question_id: str, tier1: Dict):
        """Extract difficulty classification"""
        diff = tier1.get('difficulty_analysis', {})
        if isinstance(diff, dict):
            overall = diff.get('overall', 'Unknown')
            score = diff.get('score', 0)
            if overall:
                self.nodes['DifficultyLevel'][overall] = {'typical_score': score}
                self.edges.append((question_id, 'HAS_DIFFICULTY', overall, {'score': score}))
    
    def _extract_common_mistakes(self, question_id: str, tier2: Dict):
        """Extract common mistakes"""
        mistakes = tier2.get('common_mistakes', [])
        if isinstance(mistakes, list):
            for mistake in mistakes:
                if isinstance(mistake, dict):
                    mistake_text = mistake.get('mistake', '')
                    if mistake_text:
                        mistake_id = self._truncate_id(mistake_text)
                        self.nodes['CommonMistake'][mistake_id] = {
                            'full_text': mistake_text,
                            'type': mistake.get('type', ''),
                            'severity': mistake.get('severity', ''),
                            'why': mistake.get('why_students_make_it', '')
                        }
                        self.edges.append((question_id, 'HAS_MISTAKE', mistake_id, {}))
                        self.stats['mistakes'] += 1
    
    def _extract_mnemonics(self, question_id: str, tier2: Dict):
        """Extract mnemonics"""
        mnemonics = tier2.get('mnemonics_memory_aids', [])
        if isinstance(mnemonics, list):
            for mnemonic in mnemonics:
                if isinstance(mnemonic, dict):
                    mnemonic_text = mnemonic.get('mnemonic', '')
                    if mnemonic_text:
                        mnemonic_id = self._truncate_id(mnemonic_text)
                        self.nodes['Mnemonic'][mnemonic_id] = {
                            'full_text': mnemonic_text,
                            'concept': mnemonic.get('concept', ''),
                            'effectiveness': mnemonic.get('effectiveness', '')
                        }
                        self.edges.append((question_id, 'HAS_MNEMONIC', mnemonic_id, {}))
                        self.stats['mnemonics'] += 1
    
    def _clean_formula_id(self, formula: str) -> str:
        """Create a clean ID for a formula"""
        if not formula:
            return ''
        # Remove LaTeX delimiters and take first 50 chars
        cleaned = formula.replace('$', '').replace('\\', '').strip()
        return cleaned[:50] if cleaned else ''
    
    def _truncate_id(self, text: str, max_len: int = 80) -> str:
        """Truncate text for use as ID"""
        if not text:
            return ''
        return text[:max_len].strip()
    
    def _save_nodes(self):
        """Save nodes to CSV"""
        nodes_path = self.output_dir / "nodes.csv"
        
        with open(nodes_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['node_id', 'node_type', 'attributes_json'])
            
            for node_type, nodes in self.nodes.items():
                for node_id, attrs in nodes.items():
                    writer.writerow([node_id, node_type, json.dumps(attrs, ensure_ascii=False)])
        
        print(f"\nSaved nodes to: {nodes_path}")
    
    def _save_edges(self):
        """Save edges to CSV"""
        edges_path = self.output_dir / "edges.csv"
        
        with open(edges_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['source', 'edge_type', 'target', 'attributes_json'])
            
            for source, edge_type, target, attrs in self.edges:
                writer.writerow([source, edge_type, target, json.dumps(attrs, ensure_ascii=False)])
        
        print(f"Saved edges to: {edges_path}")
    
    def _save_statistics(self):
        """Save statistics to JSON"""
        stats_path = self.output_dir / "kg_statistics.json"
        
        summary = {
            'total_nodes': sum(len(nodes) for nodes in self.nodes.values()),
            'total_edges': len(self.edges),
            'nodes_by_type': {k: len(v) for k, v in self.nodes.items()},
            'edges_by_type': defaultdict(int),
            'extraction_stats': dict(self.stats)
        }
        
        for _, edge_type, _, _ in self.edges:
            summary['edges_by_type'][edge_type] += 1
        
        summary['edges_by_type'] = dict(summary['edges_by_type'])
        
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Saved statistics to: {stats_path}")


def main():
    """Run Knowledge Graph extraction"""
    extractor = KnowledgeGraphExtractor(
        input_dir=VOTING_ENGINE_DIR,
        output_dir=OUTPUT_DIR
    )
    
    stats = extractor.extract_all()
    
    print("\n" + "=" * 60)
    print("KNOWLEDGE GRAPH READY FOR:")
    print("=" * 60)
    print("1. Neo4j import (via CSV)")
    print("2. NetworkX analysis")
    print("3. Graph visualization (Gephi)")
    print("4. Paper figures")


if __name__ == "__main__":
    main()
