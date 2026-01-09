"""
Script to fill Human Evaluation CSV using GPT-4o.
Reads research_publications/evaluation/human_eval_sample.csv
Outputs research_publications/evaluation/human_eval_gpt4o_filled.csv

Usage:
    python research_publications/scripts/fill_human_eval_gpt4o.py [--api-key KEY]
"""

import csv
import json
import os
import sys
import argparse
import time
import base64
from pathlib import Path
from typing import Dict, List, Optional

# Add script directory to path to import from llm_judge_evaluation
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

try:
    from llm_judge_evaluation import collect_all_questions, find_question_image
except ImportError:
    # Fallback if running from root
    sys.path.append(str(current_dir / "research_publications" / "scripts"))
    try:
        from research_publications.scripts.llm_judge_evaluation import collect_all_questions, find_question_image
    except ImportError:
        # Last resort: try importing as if in same dir
        sys.path.append(str(current_dir))
        from llm_judge_evaluation import collect_all_questions, find_question_image

try:
    from openai import OpenAI
except ImportError:
    print("Error: 'openai' package not found. Please install it: pip install openai")
    sys.exit(1)

# Configuration
INPUT_CSV = Path("research_publications/evaluation/human_eval_sample.csv")
OUTPUT_CSV = Path("research_publications/evaluation/human_eval_gpt4o_filled.csv")

def encode_image(image_path: Path) -> str:
    """Encode image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_gpt4o_evaluation(client: OpenAI, question_data: Dict, row: Dict, image_path: Optional[Path] = None) -> Dict:
    """Ask GPT-4o to evaluate the specific item."""
    
    question_text = question_data["consensus"].get("question_text", "")
    correct_answer = question_data["consensus"].get("tier_1_core_research", {}).get("answer_validation", {}).get("correct_answer", "")
    
    category = row["Category"]
    item_to_verify = row["Item_To_Verify"]
    context_value = row["Context/Value"]
    
    prompt_text = f"""You are an expert Aerospace Engineering educator and content evaluator.
    
CONTEXT:
Question: {question_text}
Correct Answer: {correct_answer}

ITEM TO EVALUATE:
Category: {category}
Item Check: {item_to_verify}
Content to Review: "{context_value}"

TASK:
Evaluate the "Content to Review" based on the "Item Check".
1. Is it Correct? (1 for Yes, 0 for No). For "Common Mistake", is it a VALID mistake students make?
2. Is it a Hallucination? (1 for Yes, 0 for No). Does it invent facts/math?
3. Quality Rating (1-5). 5 = Excellent/Essential, 1 = Useless/Wrong.
4. Comments. Brief explanation (<30 words).

OUTPUT JSON ONLY:
{{
  "is_correct": 1,
  "is_hallucination": 0,
  "quality_rating": 5,
  "comments": "Explanation..."
}}
"""

    messages = [
        {"role": "system", "content": "You are a helpful AI judge. Output JSON only."}
    ]

    user_content = []
    
    # Add image if available
    if image_path:
        base64_image = encode_image(image_path)
        user_content.append({
            "type": "text",
            "text": "[IMAGE PROVIDED FOR CONTEXT]"
        })
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_image}"
            }
        })
    
    # Add text prompt
    user_content.append({
        "type": "text",
        "text": prompt_text
    })
    
    messages.append({"role": "user", "content": user_content})

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        print(f"Error calling GPT-4o: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", help="OpenAI API Key")
    args = parser.parse_args()
    
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found. Pass --api-key or set env var.")
        return

    client = OpenAI(api_key=api_key)
    
    # 1. Load Questions
    print("Loading question data...")
    all_questions = collect_all_questions()
    questions_map = {q["question_id"]: q for q in all_questions}
    print(f"Loaded {len(questions_map)} questions.")
    
    # 2. Read CSV
    if not INPUT_CSV.exists():
        print(f"Error: {INPUT_CSV} not found.")
        return
        
    rows = []
    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    
    print(f"Loaded {len(rows)} rows from CSV.")
    
    # 3. Process Rows
    filled_count = 0
    processed_rows = []
    
    print("Starting evaluation...")
    for i, row in enumerate(rows):
        qid = row["Question_ID"]
        
        # Skip if already filled (checking Is_Correct)
        if row.get("Is_Correct (1/0)") and row.get("Is_Correct (1/0)").strip():
            processed_rows.append(row)
            continue
            
        if qid not in questions_map:
            print(f"Warning: Question {qid} not found in data. Skipping.")
            processed_rows.append(row)
            continue
            
        # Find image
        q_data = questions_map[qid]
        image_path = None
        media_type = q_data["consensus"].get("tier_0_classification", {}).get("media_type", "text_only")
        if media_type == "image_based":
            # Pass year from q_data (convert to int just in case)
            image_path = find_question_image(qid, int(q_data["year"]))
            
        print(f"[{i+1}/{len(rows)}] Evaluating {qid} - {row['Category']} {'[IMAGE]' if image_path else ''}...")
        
        result = get_gpt4o_evaluation(client, q_data, row, image_path)
        
        if result:
            row["Is_Correct (1/0)"] = result.get("is_correct", "")
            row["Is_Hallucination (1/0)"] = result.get("is_hallucination", "")
            row["Quality_Rating (1-5)"] = result.get("quality_rating", "")
            row["Comments"] = result.get("comments", "")
            filled_count += 1
        
        processed_rows.append(row)
        
        # Incremental save every 5 rows
        if filled_count > 0 and filled_count % 5 == 0:
            with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(processed_rows + rows[i+1:])
            print(f"Saved progress to {OUTPUT_CSV}")

    # Final save
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(processed_rows)
    
    print(f"\nDone! Filled {filled_count} rows.")
    print(f"Saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
