"""
LLM-as-Judge Evaluation Script v2: Pairwise Comparison
=======================================================

Uses Gemini 2.5 Flash with PAIRWISE comparisons to avoid skimming.

Key improvements:
- Pairwise: FLSS vs each model (not all-at-once)
- 5 fields per API call (focused evaluation)
- 75-word max justifications
- Full question context included
- Statistical significance tests
- Win rate + confidence intervals

Usage:
    python llm_judge_evaluation.py [--sample N] [--output-dir DIR]
"""

import json
import argparse
import os
import time
import statistics
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Optional, Any
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import random
import math

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INDIVIDUAL_RESPONSES_DIR = PROJECT_ROOT / "Individual_responses"
VOTING_ENGINE_DIR = PROJECT_ROOT / "debug_outputs" / "voting_engine"
OUTPUT_DIR = PROJECT_ROOT / "research_publications" / "evaluation"
RAW_IMAGES_DIR = PROJECT_ROOT / "data" / "raw_images"

JUDGE_MODEL = "gemini-2.5-flash"  # Gemini 2.5 Flash
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or "AIza_REDACTED_GOOGLE_KEY"


def find_question_image(question_id: str, year: int) -> Optional[Path]:
    """Find the image file for a question if it exists."""
    year_dir = RAW_IMAGES_DIR / str(year)
    if not year_dir.exists():
        return None
    
    # Try different naming patterns
    patterns = [
        f"{question_id}.png",
        f"{question_id}.jpg",
        f"GATE_AE_{year}_{question_id.split('_')[-1]}.png",  # Old format
        f"GATE_{year}_AE_{question_id.split('_')[-1]}.png",  # 2009+ format
    ]
    
    for pattern in patterns:
        img_path = year_dir / pattern
        if img_path.exists():
            return img_path
    
    # Search for any matching file
    for ext in [".png", ".jpg", ".jpeg"]:
        for img_file in year_dir.glob(f"*{ext}"):
            # Extract question number from filename
            q_num = question_id.split("_")[-1]  # e.g., "Q20" from "GATE_2009_AE_Q20"
            if q_num in img_file.stem:
                return img_file
    
    return None

# 10 fields split into 2 batches of 5
FIELD_BATCHES = [
    # Batch 1: Core Quality + Student Basics
    [
        {"name": "answer_reasoning", "path": "tier_1_core_research.answer_validation.reasoning", "criteria": "accuracy, clarity, depth of explanation"},
        {"name": "step_by_step", "path": "tier_1_core_research.explanation.step_by_step", "criteria": "logical flow, completeness, clarity"},
        {"name": "common_mistakes", "path": "tier_2_student_learning.common_mistakes", "criteria": "relevance to question, helpfulness, specificity"},
        {"name": "mnemonics", "path": "tier_2_student_learning.mnemonics_memory_aids", "criteria": "memorability, relevance, creativity"},
        {"name": "flashcards", "path": "tier_2_student_learning.flashcards", "criteria": "question clarity, answer accuracy, learning value"},
    ],
    # Batch 2: Context + Enhanced
    [
        {"name": "real_world_context", "path": "tier_2_student_learning.real_world_context", "criteria": "industry relevance, accuracy, educational value"},
        {"name": "exam_strategy", "path": "tier_2_student_learning.exam_strategy", "criteria": "actionability, practicality, time-saving value"},
        {"name": "real_world_applications", "path": "tier_1_core_research.real_world_applications", "criteria": "industry accuracy, aerospace relevance, specificity"},
        {"name": "key_insights", "path": "tier_1_core_research.step_by_step_solution.key_insights", "criteria": "insightfulness, uniqueness, learning value"},
        {"name": "difficulty_factors", "path": "tier_1_core_research.difficulty_analysis.difficulty_factors", "criteria": "validity, specificity, calibration accuracy"},
    ]
]


def get_nested_value(data: Dict, path: str) -> Any:
    """Get value from nested dict using dot notation."""
    keys = path.split(".")
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None
    return current


def format_content(value: Any, max_chars: int = 1500) -> str:
    """Format content for prompt, truncating if needed."""
    if value is None:
        return "[EMPTY/MISSING]"
    if isinstance(value, list):
        items = [v for v in value if v is not None]
        if not items:
            return "[EMPTY LIST]"
        text = json.dumps(items, indent=2, ensure_ascii=False)
    elif isinstance(value, dict):
        text = json.dumps(value, indent=2, ensure_ascii=False)
    else:
        text = str(value)
    
    if len(text) > max_chars:
        return text[:max_chars] + "... [TRUNCATED]"
    return text


def build_pairwise_prompt(
    question_text: str,
    correct_answer: str,
    question_type: str,
    field_batch: List[Dict],
    output_a: Dict,
    output_b: Dict,
    name_a: str,
    name_b: str
) -> str:
    """Build pairwise comparison prompt for a batch of 5 fields."""
    
    sections = []
    
    for field in field_batch:
        val_a = get_nested_value(output_a, field["path"])
        val_b = get_nested_value(output_b, field["path"])
        
        sections.append(f"""
### Field: {field['name']}
**Judge on:** {field['criteria']}

**OUTPUT A ({name_a}):**
```
{format_content(val_a)}
```

**OUTPUT B ({name_b}):**
```
{format_content(val_b)}
```
""")
    
    prompt = f"""You are an expert evaluator for educational content in Aerospace Engineering.

══════════════════════════════════════════════════════════════════════════════
CONTEXT
══════════════════════════════════════════════════════════════════════════════

**Question ({question_type}):**
{question_text[:1000]}

**Correct Answer:** {correct_answer}

══════════════════════════════════════════════════════════════════════════════
TASK: Compare OUTPUT A vs OUTPUT B for each field below.
══════════════════════════════════════════════════════════════════════════════

{"".join(sections)}

══════════════════════════════════════════════════════════════════════════════
INSTRUCTIONS
══════════════════════════════════════════════════════════════════════════════

1. ONLY evaluate what is shown. Do NOT add your own knowledge or perspective.
2. For EACH field, decide: A is better, B is better, or TIE
3. Provide a reason (max 75 words) explaining WHY you chose the winner.
4. If one output is empty/missing and the other has content, the one with content wins.
5. Be objective and consistent.

══════════════════════════════════════════════════════════════════════════════
OUTPUT FORMAT - STRICTLY FOLLOW THIS JSON STRUCTURE
══════════════════════════════════════════════════════════════════════════════

CRITICAL: You MUST respond with ONLY valid JSON. No text before or after.
Do NOT use markdown code blocks. Output raw JSON only.

{{
  "comparisons": [
    {{"field": "answer_reasoning", "winner": "A", "reason": "Your reason here max 75 words"}},
    {{"field": "step_by_step", "winner": "B", "reason": "Your reason here max 75 words"}},
    {{"field": "common_mistakes", "winner": "TIE", "reason": "Your reason here max 75 words"}},
    {{"field": "mnemonics", "winner": "A", "reason": "Your reason here max 75 words"}},
    {{"field": "flashcards", "winner": "A", "reason": "Your reason here max 75 words"}}
  ],
  "overall_winner": "A",
  "overall_reason": "Summary max 75 words"
}}

Replace the field names with the actual field names from above.
Winner must be exactly "A", "B", or "TIE" (uppercase).
"""
    return prompt


def find_individual_responses(question_id: str, year: int) -> Dict[str, Dict]:
    """Find all individual model responses for a question."""
    year_dir = INDIVIDUAL_RESPONSES_DIR / str(year)
    question_dir = year_dir / question_id
    
    if not question_dir.exists():
        return {}
    
    responses = {}
    for model_dir in question_dir.iterdir():
        if model_dir.is_dir():
            response_file = model_dir / "response.json"
            if response_file.exists():
                try:
                    with open(response_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if "response_json" in data:
                            responses[model_dir.name] = data["response_json"]
                        else:
                            responses[model_dir.name] = data
                except Exception as e:
                    pass
    return responses


def find_consensus_output(question_id: str) -> Optional[Dict]:
    """Find the final consensus output for a question."""
    pattern = f"{question_id}_*_final_json.json"
    matches = list(VOTING_ENGINE_DIR.glob(pattern))
    if matches:
        try:
            with open(matches[0], "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            pass
    return None


def collect_all_questions() -> List[Dict]:
    """Collect questions with both individual responses and consensus output."""
    questions = []
    
    for year_dir in sorted(INDIVIDUAL_RESPONSES_DIR.iterdir()):
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue
        year = int(year_dir.name)
        
        for question_dir in sorted(year_dir.iterdir()):
            if not question_dir.is_dir():
                continue
            question_id = question_dir.name
            
            individual_responses = find_individual_responses(question_id, year)
            if not individual_responses:
                continue
            
            consensus = find_consensus_output(question_id)
            if consensus is None:
                continue
            
            questions.append({
                "question_id": question_id,
                "year": year,
                "individual_responses": individual_responses,
                "consensus": consensus,
            })
    
    return questions


class LLMJudge:
    """LLM judge using pairwise comparison."""
    
    def __init__(self, api_key: str, model_name: str = JUDGE_MODEL):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.request_count = 0
        self.last_request_time = 0
    
    def _rate_limit(self, min_delay: float = 0.5):
        elapsed = time.time() - self.last_request_time
        if elapsed < min_delay:
            time.sleep(min_delay - elapsed)
        self.last_request_time = time.time()
    

    def compare_pairwise(
        self,
        question_text: str,
        correct_answer: str,
        question_type: str,
        field_batch: List[Dict],
        consensus_output: Dict,
        model_output: Dict,
        model_name: str,
        image_path: Optional[Path] = None,
        swap_position: bool = False
    ) -> Optional[Dict]:
        """Compare FLSS vs single model on a batch of fields.
        
        Args:
            swap_position: If True, Model is A and FLSS is B.
                          If False, FLSS is A and Model is B.
        """
        
        # Determine A and B based on swap_position
        if swap_position:
            # Model is A, FLSS is B
            output_a = model_output
            output_b = consensus_output
            name_a = model_name
            name_b = "FLSS Consensus"
        else:
            # FLSS is A, Model is B
            output_a = consensus_output
            output_b = model_output
            name_a = "FLSS Consensus"
            name_b = model_name
            
        prompt = build_pairwise_prompt(
            question_text, correct_answer, question_type,
            field_batch, output_a, output_b, name_a, name_b
        )
        
        self._rate_limit()
        
        try:
            # Build content with optional image
            content_parts = []
            
            if image_path and image_path.exists():
                # Upload image for multimodal evaluation
                try:
                    img_file = genai.upload_file(str(image_path))
                    # Wait for upload to complete
                    import time
                    while img_file.state.name == "PROCESSING":
                        time.sleep(0.5)
                        img_file = genai.get_file(img_file.name)
                    content_parts.append(img_file)
                    content_parts.append(f"\n[IMAGE ABOVE: This is the question's diagram/figure]\n\n{prompt}")
                except Exception as img_err:
                    print(f"    Warning: Could not upload image: {img_err}")
                    content_parts.append(prompt)
            else:
                content_parts.append(prompt)
            
            response = self.model.generate_content(
                content_parts,
                generation_config=genai.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=4000,
                    response_mime_type="application/json",
                )
            )
            self.request_count += 1
            
            text = response.text.strip()
            
            # Try direct parse first
            try:
                res = json.loads(text)
                return self._swap_result_winners(res) if swap_position else res
            except json.JSONDecodeError as e:
                pass
            
            # Try extracting from code block
            import re
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
            if json_match:
                try:
                    res = json.loads(json_match.group(1).strip())
                    return self._swap_result_winners(res) if swap_position else res
                except json.JSONDecodeError:
                    pass
            
            # Try finding the outermost JSON object
            first_brace = text.find('{')
            last_brace = text.rfind('}')
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                json_str = text[first_brace:last_brace + 1]
                try:
                    res = json.loads(json_str)
                    return self._swap_result_winners(res) if swap_position else res
                except json.JSONDecodeError as e:
                    print(f"    Warning: JSON parse error at position {e.pos}: {e.msg}")
                    # Try to fix common issues - remove control characters
                    json_str_clean = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)
                    try:
                        res = json.loads(json_str_clean)
                        return self._swap_result_winners(res) if swap_position else res
                    except:
                        pass
                    
                    # FALLBACK: Extract just the winners using regex
                    print(f"    Attempting regex fallback extraction...")
                    winners = re.findall(r'"field"\s*:\s*"([^"]+)"[^}]*"winner"\s*:\s*"([ABT][A-Z]*)"', json_str, re.IGNORECASE)
                    if winners:
                        comparisons = [{"field": f, "winner": w.upper(), "reason": "Extracted via fallback"} for f, w in winners]
                        overall_match = re.search(r'"overall_winner"\s*:\s*"([ABT][A-Z]*)"', json_str, re.IGNORECASE)
                        result = {
                            "comparisons": comparisons,
                            "overall_winner": overall_match.group(1).upper() if overall_match else "TIE",
                            "overall_reason": "Partial extraction due to JSON parse error",
                            "_fallback_extraction": True
                        }
                        
                        if swap_position:
                            return self._swap_result_winners(result)
                        return result
            
            print(f"    Warning: Could not parse JSON. Response length: {len(text)}, starts with: {text[:100]}")
            return None
            
        except Exception as e:
            print(f"    Warning: Judge error: {type(e).__name__}: {e}")
            return None

    def _swap_result_winners(self, result: Dict) -> Dict:
        """Swap A/B winners back to standard form (A=FLSS, B=Model)."""
        if not result:
            return result
        
        # Swap comparisons
        for comp in result.get("comparisons", []):
            if comp["winner"] == "A":
                comp["winner"] = "B"
            elif comp["winner"] == "B":
                comp["winner"] = "A"
        
        # Swap overall
        if result.get("overall_winner") == "A":
            result["overall_winner"] = "B"
        elif result.get("overall_winner") == "B":
            result["overall_winner"] = "A"
            
        return result


# ============================================================================
# STATISTICAL ANALYSIS FUNCTIONS
# ============================================================================

def calculate_win_rate(results: List[Dict], model_name: str) -> Dict:
    """Calculate win rate of FLSS vs a specific model."""
    wins, losses, ties = 0, 0, 0
    
    for r in results:
        if model_name not in r.get("comparisons", {}):
            continue
        for batch_result in r["comparisons"][model_name]:
            if batch_result is None:
                continue
            for comp in batch_result.get("comparisons", []):
                winner = comp.get("winner", "").upper()
                if winner == "A":
                    wins += 1
                elif winner == "B":
                    losses += 1
                else:
                    ties += 1
    
    total = wins + losses + ties
    if total == 0:
        return {"wins": 0, "losses": 0, "ties": 0, "win_rate": 0, "loss_rate": 0, "tie_rate": 0, "total": 0}
    
    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "total": total,
        "win_rate": wins / total,
        "loss_rate": losses / total,
        "tie_rate": ties / total,
    }


def bootstrap_confidence_interval(data: List[float], confidence: float = 0.95, n_bootstrap: int = 1000) -> Tuple[float, float]:
    """Calculate bootstrap confidence interval."""
    if not data or len(data) < 2:
        return (0.0, 0.0)
    
    bootstrap_means = []
    n = len(data)
    
    for _ in range(n_bootstrap):
        sample = [random.choice(data) for _ in range(n)]
        bootstrap_means.append(statistics.mean(sample))
    
    bootstrap_means.sort()
    lower_idx = int((1 - confidence) / 2 * n_bootstrap)
    upper_idx = int((1 + confidence) / 2 * n_bootstrap)
    
    return (bootstrap_means[lower_idx], bootstrap_means[upper_idx])


def wilcoxon_signed_rank_test(wins: int, losses: int) -> float:
    """Simplified Wilcoxon signed-rank test (sign test approximation)."""
    n = wins + losses
    if n == 0:
        return 1.0
    
    # Using normal approximation for sign test
    # Under null hypothesis, wins ~ Binomial(n, 0.5)
    expected = n / 2
    std = math.sqrt(n / 4)
    
    if std == 0:
        return 1.0
    
    z = (wins - expected) / std
    
    # Two-tailed p-value approximation
    # Using standard normal CDF approximation
    p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return p_value


def cohens_d(wins: int, losses: int, ties: int) -> float:
    """Calculate Cohen's d effect size from win/loss counts."""
    # Convert to pseudo-scores: win=1, tie=0.5, loss=0
    n = wins + losses + ties
    if n == 0:
        return 0.0
    
    # Mean of A's "performance"
    mean_a = (wins * 1 + ties * 0.5) / n
    mean_b = (losses * 1 + ties * 0.5) / n
    
    # Pooled std approximation
    std = 0.5  # Since scores are 0, 0.5, or 1
    
    if std == 0:
        return 0.0
    
    return (mean_a - mean_b) / std


def interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


# ============================================================================
# PARALLEL PROCESSING HELPER
# ============================================================================

def evaluate_single_question(q: Dict, judge: LLMJudge) -> Dict:
    """Evaluate a single question across all models. Safe for parallel execution."""
    try:
        question_text = q["consensus"].get("question_text", "")
        correct_answer = get_nested_value(q["consensus"], "tier_1_core_research.answer_validation.correct_answer") or ""
        question_type = get_nested_value(q["consensus"], "tier_0_classification.content_type") or "unknown"
        media_type = get_nested_value(q["consensus"], "tier_0_classification.media_type") or "text_only"
        
        # Find image for image-based questions
        image_path = None
        if media_type == "image_based":
            image_path = find_question_image(q["question_id"], q["year"])
            if image_path:
                print(f"      [{q['question_id']}] Including image: {image_path.name}")
        
        question_result = {
            "question_id": q["question_id"],
            "year": q["year"],
            "has_image": image_path is not None,
            "comparisons": {}
        }
        
        # Compare FLSS vs each model
        for model_name, model_output in q["individual_responses"].items():
            model_comparisons = []
            
            # Determines if we should swap positions (Model A=Model, Model B=FLSS)
            # to avoid positional bias. 50% chance.
            swap = random.choice([True, False])
            
            for batch in FIELD_BATCHES:
                result = judge.compare_pairwise(
                    question_text, correct_answer, question_type,
                    batch, q["consensus"], model_output, model_name,
                    image_path=image_path,
                    swap_position=swap
                )
                model_comparisons.append(result)
            
            question_result["comparisons"][model_name] = model_comparisons
            
        return question_result
        
    except Exception as e:
        print(f"Error processing {q['question_id']}: {e}")
        return {
            "question_id": q["question_id"],
            "year": q["year"],
            "error": str(e),
            "comparisons": {}
        }


# ============================================================================
# MAIN EVALUATION
# ============================================================================

def run_pairwise_evaluation(sample_size: Optional[int] = None, year_filter: Optional[int] = None, output_dir: Optional[Path] = None) -> Dict:
    """Run pairwise LLM-as-Judge evaluation."""
    
    print("=" * 70)
    print("LLM-as-Judge: Pairwise Comparison Evaluation")
    print("=" * 70)
    
    # Collect questions
    print("\n[1/5] Collecting questions...")
    questions = collect_all_questions()
    print(f"    Found {len(questions)} questions with complete data")
    
    # Filter by year if specified
    if year_filter:
        questions = [q for q in questions if q["year"] == year_filter]
        print(f"    Filtered to year {year_filter}: {len(questions)} questions")
    else:
        # Get unique years for display
        years = sorted(set(q["year"] for q in questions))
        print(f"    Running ALL years: {years[0]}-{years[-1]} ({len(years)} years)")
    
    if sample_size and sample_size < len(questions):
        random.seed(42)
        questions = random.sample(questions, sample_size)
        print(f"    Using sample of {sample_size} questions")
    
    # Set up output directory for incremental saves
    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    incremental_path = output_dir / "llm_judge_incremental.json"
    
    # Initialize judge
    print("\n[2/5] Initializing LLM judge (Gemini 2.5 Flash)...")
    judge = LLMJudge(api_key=GEMINI_API_KEY)
    
    # Run evaluations
    print("\n[3/5] Running pairwise comparisons (Parallel: 3 workers)...")
    all_results = []
    
    # Thread-safe lock for file writing
    write_lock = threading.Lock()
    completed_count = 0
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Map futures to question IDs for error tracking
        future_to_qid = {
            executor.submit(evaluate_single_question, q, judge): q["question_id"] 
            for q in questions
        }
        
        for future in as_completed(future_to_qid):
            qid = future_to_qid[future]
            try:
                question_result = future.result()
                all_results.append(question_result)
                
                completed_count += 1
                if completed_count % 5 == 0:
                    print(f"    Processed {completed_count}/{len(questions)} questions...")

                # Incremental save (thread-safe)
                with write_lock:
                    with open(incremental_path, "w", encoding="utf-8") as f:
                        json.dump({
                            "completed": completed_count,
                            "total": len(questions),
                            "year_filter": year_filter,
                            "results": all_results,
                        }, f, indent=2, default=str)
                        
            except Exception as exc:
                print(f"    Request for {qid} generated an exception: {exc}")
    
    # Calculate statistics
    print("\n[4/5] Calculating statistics...")
    
    model_stats = {}
    all_models = set()
    for r in all_results:
        all_models.update(r["comparisons"].keys())
    
    for model_name in all_models:
        win_data = calculate_win_rate(all_results, model_name)
        
        # Statistical tests
        p_value = wilcoxon_signed_rank_test(win_data["wins"], win_data["losses"])
        effect_size = cohens_d(win_data["wins"], win_data["losses"], win_data["ties"])
        
        # Bootstrap CI for win rate
        win_indicators = [1] * win_data["wins"] + [0] * win_data["losses"] + [0.5] * win_data["ties"]
        ci_low, ci_high = bootstrap_confidence_interval(win_indicators)
        
        model_stats[model_name] = {
            **win_data,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "effect_size": effect_size,
            "effect_interpretation": interpret_effect_size(effect_size),
            "ci_95_low": ci_low,
            "ci_95_high": ci_high,
        }
    
    # Per-field breakdown
    print("\n[5/5] Generating per-field breakdown...")
    field_stats = defaultdict(lambda: defaultdict(lambda: {"wins": 0, "losses": 0, "ties": 0}))
    
    for r in all_results:
        for model_name, batch_results in r["comparisons"].items():
            for batch_result in batch_results:
                if batch_result is None:
                    continue
                for comp in batch_result.get("comparisons", []):
                    field = comp.get("field", "unknown")
                    winner = comp.get("winner", "").upper()
                    if winner == "A":
                        field_stats[field][model_name]["wins"] += 1
                    elif winner == "B":
                        field_stats[field][model_name]["losses"] += 1
                    else:
                        field_stats[field][model_name]["ties"] += 1
    
    return {
        "question_count": len(questions),
        "api_calls": judge.request_count,
        "model_stats": model_stats,
        "field_stats": dict(field_stats),
        "detailed_results": all_results,
        "timestamp": datetime.now().isoformat(),
    }


def generate_report(results: Dict, output_path: Path):
    """Generate comprehensive markdown report."""
    
    lines = [
        "# LLM-as-Judge Ablation Report: FLSS vs Single Models",
        "",
        f"**Generated:** {results['timestamp']}",
        f"**Questions Evaluated:** {results['question_count']}",
        f"**API Calls Made:** {results['api_calls']}",
        f"**Judge Model:** Gemini 2.5 Flash",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        "This evaluation uses **pairwise comparison** to determine whether FLSS consensus",
        "outputs are better than individual model outputs. For each field, the judge picks",
        "A (FLSS) or B (single model) or TIE.",
        "",
        "> **Key Finding:** See win rates below to determine if FLSS adds value.",
        "",
        "---",
        "",
        "## 1. Overall Win Rates (FLSS vs Each Model)",
        "",
        "| Model | FLSS Wins | Model Wins | Ties | **Win Rate** | p-value | Significant? | Effect Size |",
        "|-------|-----------|------------|------|--------------|---------|--------------|-------------|",
    ]
    
    for model_name, stats in sorted(results["model_stats"].items()):
        sig = "✅ Yes" if stats["significant"] else "❌ No"
        lines.append(
            f"| {model_name} | {stats['wins']} | {stats['losses']} | {stats['ties']} | "
            f"**{stats['win_rate']:.1%}** | {stats['p_value']:.4f} | {sig} | "
            f"{stats['effect_size']:.2f} ({stats['effect_interpretation']}) |"
        )
    
    lines.extend([
        "",
        "**Interpretation:**",
        "- **Win Rate > 50%**: FLSS is better than this model",
        "- **p-value < 0.05**: Result is statistically significant",
        "- **Effect Size**: negligible (<0.2), small (0.2-0.5), medium (0.5-0.8), large (>0.8)",
        "",
        "---",
        "",
        "## 2. Per-Field Win Rates",
        "",
    ])
    
    # Get all fields and models
    all_fields = list(results["field_stats"].keys())
    all_models = set()
    for field_data in results["field_stats"].values():
        all_models.update(field_data.keys())
    all_models = sorted(all_models)
    
    for field in all_fields:
        lines.append(f"### {field}")
        lines.append("")
        lines.append("| Model | FLSS Wins | Model Wins | Ties | Win Rate |")
        lines.append("|-------|-----------|------------|------|----------|")
        
        for model in all_models:
            data = results["field_stats"][field].get(model, {"wins": 0, "losses": 0, "ties": 0})
            total = data["wins"] + data["losses"] + data["ties"]
            wr = data["wins"] / total if total > 0 else 0
            lines.append(f"| {model} | {data['wins']} | {data['losses']} | {data['ties']} | {wr:.1%} |")
        
        lines.append("")
    
    lines.extend([
        "---",
        "",
        "## 3. Statistical Notes",
        "",
        "- **Test Used:** Sign test (Wilcoxon signed-rank approximation)",
        "- **Confidence Intervals:** 95% bootstrap CI (1000 iterations)",
        "- **Effect Size:** Cohen's d calculated from win/loss proportions",
        "",
        "---",
        "",
        "## 4. Methodology",
        "",
        "1. **Pairwise Comparison:** FLSS (A) vs each model (B) separately",
        "2. **5 Fields per API Call:** Prevents skimming, ensures focused evaluation",
        "3. **Judge Instructions:** Evaluate only provided content, no hallucination",
        "4. **10 Fields Evaluated:**",
        "   - answer_reasoning, step_by_step, common_mistakes, mnemonics, flashcards",
        "   - real_world_context, exam_strategy, real_world_applications, key_insights, difficulty_factors",
        "",
        "---",
        "",
        "*Generated by `llm_judge_evaluation.py` v2*",
    ])
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print(f"\nReport saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="LLM-as-Judge Pairwise Evaluation")
    parser.add_argument("--sample", type=int, default=None, help="Number of questions to sample (default: all)")
    parser.add_argument("--year", type=int, help="Filter by specific year (e.g., 2024)")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run evaluation
    results = run_pairwise_evaluation(sample_size=args.sample, year_filter=args.year, output_dir=output_dir)
    
    # Save raw results
    results_path = output_dir / "llm_judge_pairwise_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nRaw results saved to: {results_path}")
    
    # Generate report
    report_path = output_dir / "llm_judge_pairwise_report.md"
    generate_report(results, report_path)
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY: FLSS Win Rates vs Each Model")
    print("=" * 70)
    
    for model_name, stats in sorted(results["model_stats"].items()):
        sig = "***" if stats["significant"] else ""
        print(f"  vs {model_name}: {stats['win_rate']:.1%} ({stats['wins']}/{stats['total']}) {sig}")
    
    print("\n*** = statistically significant (p < 0.05)")
    print("=" * 70)


if __name__ == "__main__":
    main()
