"""
Ablation Analysis Script for FLSS Research Paper
================================================

This script compares Field-Level Structural Synthesis (FLSS) consensus outputs
against single-model baselines to demonstrate the value of the multi-LLM
consensus approach.

Metrics computed:
1. Field Completeness - % of non-null fields
2. Array Richness - Average count of items in array fields
3. Mnemonic/Flashcard Count - Pedagogical content quantity
4. Text Length - Explanation detail level
5. Confidence Scores - Model self-assessed confidence

Baselines:
- Best-of-N: Take the highest-confidence single model's output
- Single Model: Each individual model (Gemini, Claude, DeepSeek)
- FLSS Consensus: The final merged output

Usage:
    python ablation_analysis.py [--sample N] [--output-dir DIR]
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Optional
import statistics
import csv
from datetime import datetime


# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INDIVIDUAL_RESPONSES_DIR = PROJECT_ROOT / "Individual_responses"
VOTING_ENGINE_DIR = PROJECT_ROOT / "debug_outputs" / "voting_engine"
OUTPUT_DIR = PROJECT_ROOT / "research_publications" / "evaluation"

# Fields to analyze for completeness and richness
ARRAY_FIELDS = [
    "tier_1_core_research.explanation.step_by_step",
    "tier_1_core_research.explanation.formulas_used",
    "tier_1_core_research.hierarchical_tags.concepts",
    "tier_1_core_research.prerequisites.essential",
    "tier_1_core_research.prerequisites.helpful",
    "tier_1_core_research.textbook_references",
    "tier_1_core_research.video_references",
    "tier_1_core_research.formulas_principles",
    "tier_1_core_research.real_world_applications.industry_examples",
    "tier_2_student_learning.common_mistakes",
    "tier_2_student_learning.mnemonics_memory_aids",
    "tier_2_student_learning.flashcards",
    "tier_2_student_learning.real_world_context",
    "tier_3_enhanced_learning.search_keywords",
    "tier_3_enhanced_learning.alternative_methods",
    "tier_3_enhanced_learning.deeper_dive_topics",
]

TEXT_FIELDS = [
    "tier_1_core_research.answer_validation.reasoning",
    "tier_1_core_research.real_world_applications.practical_relevance",
]

NUMERIC_FIELDS = [
    "tier_1_core_research.answer_validation.confidence",
    "tier_0_classification.classification_confidence",
    "tier_1_core_research.difficulty_analysis.score",
]


def get_nested_value(data: Dict, path: str) -> Any:
    """Get a value from a nested dictionary using dot notation."""
    keys = path.split(".")
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None
    return current


def count_non_null_fields(data: Dict, prefix: str = "") -> tuple:
    """Recursively count non-null fields in a dictionary."""
    total = 0
    non_null = 0
    
    if isinstance(data, dict):
        for key, value in data.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                t, n = count_non_null_fields(value, path)
                total += t
                non_null += n
            elif isinstance(value, list):
                total += 1
                if value and any(v is not None for v in value):
                    non_null += 1
            else:
                total += 1
                if value is not None and value != "" and value != []:
                    non_null += 1
    
    return total, non_null


def analyze_single_response(data: Dict) -> Dict[str, Any]:
    """Analyze a single model response or consensus output."""
    # Handle individual response format (has response_json wrapper)
    if "response_json" in data:
        data = data["response_json"]
    
    metrics = {}
    
    # 1. Field Completeness
    total_fields, non_null_fields = count_non_null_fields(data)
    metrics["field_completeness"] = non_null_fields / total_fields if total_fields > 0 else 0
    metrics["total_fields"] = total_fields
    metrics["non_null_fields"] = non_null_fields
    
    # 2. Array Richness (count items in array fields)
    array_counts = {}
    for field_path in ARRAY_FIELDS:
        value = get_nested_value(data, field_path)
        if isinstance(value, list):
            # Filter out None values
            valid_items = [v for v in value if v is not None]
            array_counts[field_path.split(".")[-1]] = len(valid_items)
        else:
            array_counts[field_path.split(".")[-1]] = 0
    
    metrics["array_counts"] = array_counts
    metrics["total_array_items"] = sum(array_counts.values())
    
    # 3. Pedagogical content counts
    metrics["mnemonic_count"] = array_counts.get("mnemonics_memory_aids", 0)
    metrics["flashcard_count"] = array_counts.get("flashcards", 0)
    metrics["common_mistakes_count"] = array_counts.get("common_mistakes", 0)
    metrics["step_count"] = array_counts.get("step_by_step", 0)
    metrics["formulas_count"] = array_counts.get("formulas_principles", 0) + array_counts.get("formulas_used", 0)
    
    # 4. Text lengths
    text_lengths = {}
    for field_path in TEXT_FIELDS:
        value = get_nested_value(data, field_path)
        if isinstance(value, str):
            text_lengths[field_path.split(".")[-1]] = len(value)
        else:
            text_lengths[field_path.split(".")[-1]] = 0
    
    metrics["text_lengths"] = text_lengths
    metrics["total_text_length"] = sum(text_lengths.values())
    
    # 5. Confidence scores
    for field_path in NUMERIC_FIELDS:
        key = field_path.split(".")[-1]
        value = get_nested_value(data, field_path)
        if isinstance(value, (int, float)):
            metrics[key] = value
        else:
            metrics[key] = None
    
    return metrics


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
                        responses[model_dir.name] = json.load(f)
                except (json.JSONDecodeError, Exception) as e:
                    print(f"Warning: Could not load {response_file}: {e}")
    
    return responses


def find_consensus_output(question_id: str) -> Optional[Dict]:
    """Find the final consensus output for a question."""
    # Look for *_03_final_json.json files
    pattern = f"{question_id}_*_final_json.json"
    matches = list(VOTING_ENGINE_DIR.glob(pattern))
    
    if matches:
        try:
            with open(matches[0], "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, Exception) as e:
            print(f"Warning: Could not load {matches[0]}: {e}")
    
    return None


def get_best_of_n(responses: Dict[str, Dict]) -> Optional[Dict]:
    """Select the response with highest confidence as Best-of-N baseline."""
    best_response = None
    best_confidence = -1
    
    for model_name, response in responses.items():
        data = response.get("response_json", response)
        confidence = get_nested_value(data, "tier_1_core_research.answer_validation.confidence")
        
        if confidence is not None and confidence > best_confidence:
            best_confidence = confidence
            best_response = response
    
    return best_response


def collect_all_questions() -> List[Dict]:
    """Collect all questions that have both individual responses and consensus output."""
    questions = []
    
    for year_dir in sorted(INDIVIDUAL_RESPONSES_DIR.iterdir()):
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue
        
        year = int(year_dir.name)
        
        for question_dir in sorted(year_dir.iterdir()):
            if not question_dir.is_dir():
                continue
            
            question_id = question_dir.name
            
            # Check if we have individual responses
            individual_responses = find_individual_responses(question_id, year)
            if not individual_responses:
                continue
            
            # Check if we have consensus output
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


def aggregate_metrics(all_metrics: List[Dict]) -> Dict:
    """Aggregate metrics across all questions."""
    if not all_metrics:
        return {}
    
    aggregated = {}
    
    # Get all keys from first metrics dict
    sample_keys = [k for k in all_metrics[0].keys() 
                   if not isinstance(all_metrics[0][k], dict)]
    
    for key in sample_keys:
        values = [m[key] for m in all_metrics if m.get(key) is not None]
        if values:
            if isinstance(values[0], (int, float)):
                aggregated[key] = {
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "stdev": statistics.stdev(values) if len(values) > 1 else 0,
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                }
    
    return aggregated


def run_ablation_analysis(sample_size: Optional[int] = None) -> Dict:
    """Run the full ablation analysis."""
    print("=" * 60)
    print("FLSS Ablation Analysis")
    print("=" * 60)
    
    # Collect all questions
    print("\n[1/4] Collecting questions...")
    questions = collect_all_questions()
    print(f"    Found {len(questions)} questions with complete data")
    
    if sample_size and sample_size < len(questions):
        import random
        random.seed(42)  # For reproducibility
        questions = random.sample(questions, sample_size)
        print(f"    Using sample of {sample_size} questions")
    
    # Analyze each question
    print("\n[2/4] Analyzing responses...")
    results = {
        "consensus": [],
        "best_of_n": [],
        "models": defaultdict(list),
    }
    
    for i, q in enumerate(questions):
        if (i + 1) % 100 == 0:
            print(f"    Processed {i + 1}/{len(questions)} questions...")
        
        # Analyze consensus
        consensus_metrics = analyze_single_response(q["consensus"])
        results["consensus"].append(consensus_metrics)
        
        # Analyze individual models
        for model_name, response in q["individual_responses"].items():
            model_metrics = analyze_single_response(response)
            results["models"][model_name].append(model_metrics)
        
        # Analyze Best-of-N
        best_response = get_best_of_n(q["individual_responses"])
        if best_response:
            bon_metrics = analyze_single_response(best_response)
            results["best_of_n"].append(bon_metrics)
    
    # Aggregate results
    print("\n[3/4] Aggregating metrics...")
    aggregated = {
        "consensus": aggregate_metrics(results["consensus"]),
        "best_of_n": aggregate_metrics(results["best_of_n"]),
        "models": {name: aggregate_metrics(metrics) 
                   for name, metrics in results["models"].items()},
    }
    
    # Compute relative improvements
    print("\n[4/4] Computing improvements...")
    if aggregated["best_of_n"] and aggregated["consensus"]:
        improvements = {}
        for key in aggregated["consensus"]:
            if key in aggregated["best_of_n"]:
                bon_val = aggregated["best_of_n"][key]["mean"]
                cons_val = aggregated["consensus"][key]["mean"]
                if bon_val > 0:
                    improvements[key] = ((cons_val - bon_val) / bon_val) * 100
                else:
                    improvements[key] = None
        aggregated["improvements_vs_best_of_n"] = improvements
    
    return {
        "summary": aggregated,
        "question_count": len(questions),
        "models_analyzed": list(results["models"].keys()),
        "timestamp": datetime.now().isoformat(),
    }


def generate_report(results: Dict, output_path: Path):
    """Generate a markdown report of the ablation analysis."""
    
    summary = results["summary"]
    
    report = [
        "# FLSS Ablation Analysis Report",
        "",
        f"**Generated:** {results['timestamp']}",
        f"**Questions Analyzed:** {results['question_count']}",
        f"**Models:** {', '.join(results['models_analyzed'])}",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        "This analysis compares the Field-Level Structural Synthesis (FLSS) consensus outputs",
        "against single-model baselines to quantify the value of multi-LLM consensus.",
        "",
        "---",
        "",
        "## 1. Field Completeness",
        "",
        "| Method | Mean | Median | Std Dev |",
        "|--------|------|--------|---------|",
    ]
    
    # Add field completeness rows
    fc = summary["consensus"].get("field_completeness", {})
    report.append(f"| **FLSS Consensus** | {fc.get('mean', 0):.2%} | {fc.get('median', 0):.2%} | {fc.get('stdev', 0):.2%} |")
    
    fc_bon = summary["best_of_n"].get("field_completeness", {})
    report.append(f"| Best-of-N | {fc_bon.get('mean', 0):.2%} | {fc_bon.get('median', 0):.2%} | {fc_bon.get('stdev', 0):.2%} |")
    
    for model_name, model_summary in sorted(summary["models"].items()):
        fc_model = model_summary.get("field_completeness", {})
        report.append(f"| {model_name} | {fc_model.get('mean', 0):.2%} | {fc_model.get('median', 0):.2%} | {fc_model.get('stdev', 0):.2%} |")
    
    report.extend([
        "",
        "---",
        "",
        "## 2. Pedagogical Content Richness",
        "",
        "### 2.1 Mnemonic Count",
        "",
        "| Method | Mean | Median | Max |",
        "|--------|------|--------|-----|",
    ])
    
    # Mnemonic counts
    mc = summary["consensus"].get("mnemonic_count", {})
    report.append(f"| **FLSS Consensus** | {mc.get('mean', 0):.2f} | {mc.get('median', 0):.1f} | {mc.get('max', 0)} |")
    
    mc_bon = summary["best_of_n"].get("mnemonic_count", {})
    report.append(f"| Best-of-N | {mc_bon.get('mean', 0):.2f} | {mc_bon.get('median', 0):.1f} | {mc_bon.get('max', 0)} |")
    
    for model_name, model_summary in sorted(summary["models"].items()):
        mc_model = model_summary.get("mnemonic_count", {})
        report.append(f"| {model_name} | {mc_model.get('mean', 0):.2f} | {mc_model.get('median', 0):.1f} | {mc_model.get('max', 0)} |")
    
    report.extend([
        "",
        "### 2.2 Flashcard Count",
        "",
        "| Method | Mean | Median | Max |",
        "|--------|------|--------|-----|",
    ])
    
    # Flashcard counts
    fcc = summary["consensus"].get("flashcard_count", {})
    report.append(f"| **FLSS Consensus** | {fcc.get('mean', 0):.2f} | {fcc.get('median', 0):.1f} | {fcc.get('max', 0)} |")
    
    fcc_bon = summary["best_of_n"].get("flashcard_count", {})
    report.append(f"| Best-of-N | {fcc_bon.get('mean', 0):.2f} | {fcc_bon.get('median', 0):.1f} | {fcc_bon.get('max', 0)} |")
    
    for model_name, model_summary in sorted(summary["models"].items()):
        fcc_model = model_summary.get("flashcard_count", {})
        report.append(f"| {model_name} | {fcc_model.get('mean', 0):.2f} | {fcc_model.get('median', 0):.1f} | {fcc_model.get('max', 0)} |")
    
    report.extend([
        "",
        "### 2.3 Common Mistakes Identified",
        "",
        "| Method | Mean | Median | Max |",
        "|--------|------|--------|-----|",
    ])
    
    # Common mistakes counts
    cmc = summary["consensus"].get("common_mistakes_count", {})
    report.append(f"| **FLSS Consensus** | {cmc.get('mean', 0):.2f} | {cmc.get('median', 0):.1f} | {cmc.get('max', 0)} |")
    
    cmc_bon = summary["best_of_n"].get("common_mistakes_count", {})
    report.append(f"| Best-of-N | {cmc_bon.get('mean', 0):.2f} | {cmc_bon.get('median', 0):.1f} | {cmc_bon.get('max', 0)} |")
    
    for model_name, model_summary in sorted(summary["models"].items()):
        cmc_model = model_summary.get("common_mistakes_count", {})
        report.append(f"| {model_name} | {cmc_model.get('mean', 0):.2f} | {cmc_model.get('median', 0):.1f} | {cmc_model.get('max', 0)} |")
    
    report.extend([
        "",
        "---",
        "",
        "## 3. Total Array Items (All Pedagogical Content)",
        "",
        "| Method | Mean | Median | Std Dev |",
        "|--------|------|--------|---------|",
    ])
    
    tai = summary["consensus"].get("total_array_items", {})
    report.append(f"| **FLSS Consensus** | {tai.get('mean', 0):.1f} | {tai.get('median', 0):.1f} | {tai.get('stdev', 0):.1f} |")
    
    tai_bon = summary["best_of_n"].get("total_array_items", {})
    report.append(f"| Best-of-N | {tai_bon.get('mean', 0):.1f} | {tai_bon.get('median', 0):.1f} | {tai_bon.get('stdev', 0):.1f} |")
    
    for model_name, model_summary in sorted(summary["models"].items()):
        tai_model = model_summary.get("total_array_items", {})
        report.append(f"| {model_name} | {tai_model.get('mean', 0):.1f} | {tai_model.get('median', 0):.1f} | {tai_model.get('stdev', 0):.1f} |")
    
    # Improvements section
    if "improvements_vs_best_of_n" in summary:
        improvements = summary["improvements_vs_best_of_n"]
        report.extend([
            "",
            "---",
            "",
            "## 4. FLSS Improvement Over Best-of-N",
            "",
            "| Metric | Improvement |",
            "|--------|-------------|",
        ])
        
        for metric, improvement in sorted(improvements.items()):
            if improvement is not None:
                sign = "+" if improvement > 0 else ""
                report.append(f"| {metric} | {sign}{improvement:.1f}% |")
    
    report.extend([
        "",
        "---",
        "",
        "## Methodology",
        "",
        "- **FLSS Consensus**: Final merged output from the voting engine",
        "- **Best-of-N**: Single model response with highest answer confidence",
        "- **Individual Models**: Raw outputs from each model before consensus",
        "",
        "### Metrics Explained",
        "",
        "- **Field Completeness**: Percentage of schema fields with non-null values",
        "- **Mnemonic Count**: Number of memory aids generated",
        "- **Flashcard Count**: Number of study flashcards created",
        "- **Common Mistakes**: Student error patterns identified",
        "- **Total Array Items**: Sum of all pedagogical list items",
        "",
        "---",
        "",
        "*Generated by `ablation_analysis.py`*",
    ])
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    
    print(f"\nReport saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="FLSS Ablation Analysis")
    parser.add_argument("--sample", type=int, help="Number of questions to sample (default: all)")
    parser.add_argument("--output-dir", type=str, help="Output directory for reports")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run analysis
    results = run_ablation_analysis(sample_size=args.sample)
    
    # Save raw results
    results_path = output_dir / "ablation_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nRaw results saved to: {results_path}")
    
    # Generate markdown report
    report_path = output_dir / "ablation_report.md"
    generate_report(results, report_path)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    summary = results["summary"]
    
    fc_cons = summary["consensus"].get("field_completeness", {}).get("mean", 0)
    fc_bon = summary["best_of_n"].get("field_completeness", {}).get("mean", 0)
    
    print(f"\nField Completeness:")
    print(f"  FLSS Consensus: {fc_cons:.1%}")
    print(f"  Best-of-N:      {fc_bon:.1%}")
    if fc_bon > 0:
        print(f"  Improvement:    {((fc_cons - fc_bon) / fc_bon) * 100:+.1f}%")
    
    tai_cons = summary["consensus"].get("total_array_items", {}).get("mean", 0)
    tai_bon = summary["best_of_n"].get("total_array_items", {}).get("mean", 0)
    
    print(f"\nTotal Pedagogical Items (avg per question):")
    print(f"  FLSS Consensus: {tai_cons:.1f}")
    print(f"  Best-of-N:      {tai_bon:.1f}")
    if tai_bon > 0:
        print(f"  Improvement:    {((tai_cons - tai_bon) / tai_bon) * 100:+.1f}%")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
