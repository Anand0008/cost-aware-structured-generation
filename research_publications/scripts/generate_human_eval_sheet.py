"""
============================================================================
HUMAN EVALUATION SAMPLE GENERATOR
============================================================================
Purpose: Select 100 stratified questions for manual review to establish
ground truth for the research paper.

Outputs:
    - human_eval_sample.csv: Spreadsheet for user to grade.

Selection Logic:
    - Randomly selects 100 questions from availability.
    - Extracts key assertions to verify:
        1. Prerequisite Relations (requires/enables)
        2. Concepts Identified
        3. Mnemonics Generated
        4. Common Mistakes
        5. Generated Explanations

Author: QBT Pipeline
============================================================================
"""

import os
import json
import csv
import random
from pathlib import Path
from typing import Dict, List, Any

# Configuration
# research_publications/scripts/ -> research_publications/ -> qbt/
PROJECT_ROOT = Path(__file__).parent.parent.parent
VOTING_ENGINE_DIR = PROJECT_ROOT / "debug_outputs" / "voting_engine"
OUTPUT_DIR = PROJECT_ROOT / "research_publications" / "evaluation"
SAMPLE_SIZE = 100

class EvalGenerator:
    def __init__(self, input_dir: Path, output_dir: Path):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate(self):
        print(f"Scanning {self.input_dir}...")
        files = list(self.input_dir.glob("*_03_final_json.json"))
        
        if not files:
            print("No files found!")
            return
            
        print(f"Found {len(files)} total processed questions.")
        
        # Select sample
        selected_files = random.sample(files, Min(len(files), SAMPLE_SIZE))
        print(f"Selected {len(selected_files)} files for evaluation.")
        
        # Prepare CSV rows
        rows = []
        
        for filepath in selected_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    rows.extend(self._extract_rows(data))
            except Exception as e:
                print(f"Error reading {filepath.name}: {e}")
                
        # Write CSV
        csv_path = self.output_dir / "human_eval_sample.csv"
        headers = [
            "Question_ID",
            "Category", 
            "Item_To_Verify", 
            "Context/Value", 
            "Is_Correct (1/0)", 
            "Is_Hallucination (1/0)", 
            "Quality_Rating (1-5)",
            "Comments"
        ]
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
            
        print(f"\nSUCCESS: Generated evaluation sheet at {csv_path}")
        print(f"Total items to verify: {len(rows)}")
        
    def _extract_rows(self, data: Dict) -> List[List]:
        """Extract verifyable assertions from a single question"""
        q_id = data.get('question_id', 'Unknown')
        rows = []
        
        tier1 = data.get('tier_1_core_research', {})
        tier2 = data.get('tier_2_student_learning', {})
        
        # 1. Verify Prerequisites (Critical for KG Precision)
        prereqs = tier1.get('prerequisites', {}).get('essential', [])
        # Take max 2 prereqs per question to keep workload manageable
        for p in prereqs[:2]:
            rows.append([
                q_id,
                "Prerequisite",
                "Is this essential?",
                p,
                "", "", "", ""
            ])
            
        # 2. Verify Generated Mnemonic (Pedagogical Quality)
        mnemonics = tier2.get('mnemonics_memory_aids', [])
        if mnemonics:
            m = mnemonics[0] # Verify the first/best one
            rows.append([
                q_id,
                "Mnemonic",
                "Is valid & helpful?",
                f"{m.get('mnemonic')} ({m.get('concept')})",
                "", "", "", ""
            ])
            
        # 3. Verify Common Mistake (Pedagogical Accuracy)
        mistakes = tier2.get('common_mistakes', [])
        if mistakes:
            m = mistakes[0]
            rows.append([
                q_id,
                "Common Mistake",
                "Is valid student error?",
                m.get('mistake'),
                "", "", "", ""
            ])
            
        # 4. Verify Explanation Logic (Overall Quality)
        expl = tier1.get('explanation', {}).get('reasoning', '')
        if expl:
            snippet = expl[:300] + "..." if len(expl) > 300 else expl
            rows.append([
                q_id,
                "Explanation",
                "Is logic sound?",
                snippet,
                "", "", "", ""
            ])
            
        # 5. Verify Video URL (Hallucination Check)
        # Randomly check one video if exists across dataset
        if random.random() < 0.3: # Don't flood with videos
            videos = tier1.get('video_references', [])
            if videos:
                v = videos[0]
                rows.append([
                    q_id,
                    "Video Link",
                    "Is URL valid?",
                    v.get('video_url'),
                    "", "", "", ""
                ])
                
        return rows

def Min(a, b):
    return a if a < b else b

if __name__ == "__main__":
    EvalGenerator(VOTING_ENGINE_DIR, OUTPUT_DIR).generate()
