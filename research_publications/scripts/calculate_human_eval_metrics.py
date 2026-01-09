"""
Human Evaluation Metrics Calculator
====================================
Calculates Precision, Recall, Hallucination Rate, and Quality scores
from the GPT-4o filled human evaluation CSV.

Usage:
    python research_publications/scripts/calculate_human_eval_metrics.py
"""

import csv
from pathlib import Path
from collections import defaultdict
import json

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_CSV = PROJECT_ROOT / "research_publications" / "evaluation" / "human_eval_gpt4o_filled.csv"
OUTPUT_JSON = PROJECT_ROOT / "research_publications" / "evaluation" / "human_eval_metrics.json"
OUTPUT_MD = PROJECT_ROOT / "research_publications" / "evaluation" / "human_eval_report.md"

def load_data():
    """Load the filled CSV."""
    rows = []
    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def calculate_metrics(rows):
    """Calculate precision, hallucination rate, and quality scores."""
    
    # Overall metrics
    total = len(rows)
    correct_count = 0
    hallucination_count = 0
    quality_scores = []
    
    # Per-category breakdown
    category_stats = defaultdict(lambda: {
        "total": 0,
        "correct": 0,
        "hallucinations": 0,
        "quality_scores": []
    })
    
    for row in rows:
        # Parse values (handle empty strings)
        is_correct = row.get("Is_Correct (1/0)", "").strip()
        is_hallucination = row.get("Is_Hallucination (1/0)", "").strip()
        quality = row.get("Quality_Rating (1-5)", "").strip()
        category = row.get("Category", "Unknown").strip()
        
        # Skip rows without data
        if not is_correct:
            continue
            
        # Count correct
        if is_correct == "1":
            correct_count += 1
            category_stats[category]["correct"] += 1
        
        # Count hallucinations
        if is_hallucination == "1":
            hallucination_count += 1
            category_stats[category]["hallucinations"] += 1
        
        # Quality scores
        if quality:
            try:
                q = float(quality)
                quality_scores.append(q)
                category_stats[category]["quality_scores"].append(q)
            except ValueError:
                pass
        
        category_stats[category]["total"] += 1
    
    # Calculate overall metrics
    evaluated = sum(1 for r in rows if r.get("Is_Correct (1/0)", "").strip())
    
    metrics = {
        "overall": {
            "total_items": total,
            "evaluated_items": evaluated,
            "precision": round(correct_count / evaluated * 100, 2) if evaluated > 0 else 0,
            "hallucination_rate": round(hallucination_count / evaluated * 100, 2) if evaluated > 0 else 0,
            "avg_quality": round(sum(quality_scores) / len(quality_scores), 2) if quality_scores else 0,
            "correct_count": correct_count,
            "hallucination_count": hallucination_count
        },
        "by_category": {}
    }
    
    # Per-category metrics
    for cat, stats in category_stats.items():
        if stats["total"] > 0:
            metrics["by_category"][cat] = {
                "total": stats["total"],
                "precision": round(stats["correct"] / stats["total"] * 100, 2),
                "hallucination_rate": round(stats["hallucinations"] / stats["total"] * 100, 2),
                "avg_quality": round(sum(stats["quality_scores"]) / len(stats["quality_scores"]), 2) if stats["quality_scores"] else 0
            }
    
    return metrics

def generate_report(metrics):
    """Generate a markdown report."""
    report = """# Human Evaluation Metrics Report

## Overall Results

| Metric | Value | Target |
|--------|-------|--------|
| **Precision (Edge Correctness)** | {precision}% | >90% |
| **Hallucination Rate** | {hallucination_rate}% | <1% |
| **Average Quality Score** | {avg_quality}/5 | >4.5 |
| Total Items Evaluated | {evaluated_items} | - |

## Results by Category

| Category | Precision | Hallucination Rate | Avg Quality | Count |
|----------|-----------|-------------------|-------------|-------|
""".format(**metrics["overall"])

    for cat, stats in sorted(metrics["by_category"].items()):
        report += f"| {cat} | {stats['precision']}% | {stats['hallucination_rate']}% | {stats['avg_quality']}/5 | {stats['total']} |\n"

    report += """
## Interpretation

"""
    # Add interpretation based on results
    overall = metrics["overall"]
    
    if overall["precision"] >= 90:
        report += "✅ **Precision Target Met:** Edge correctness exceeds 90% threshold.\n\n"
    else:
        report += f"⚠️ **Precision Below Target:** {overall['precision']}% < 90% target.\n\n"
    
    if overall["hallucination_rate"] <= 1:
        report += "✅ **Hallucination Target Met:** Rate is below 1% threshold.\n\n"
    else:
        report += f"⚠️ **Hallucination Rate Above Target:** {overall['hallucination_rate']}% > 1% target.\n\n"
    
    if overall["avg_quality"] >= 4.5:
        report += "✅ **Quality Target Met:** Average quality exceeds 4.5/5 threshold.\n\n"
    else:
        report += f"⚠️ **Quality Below Target:** {overall['avg_quality']}/5 < 4.5/5 target.\n\n"
    
    return report

def main():
    print("Loading human evaluation data...")
    rows = load_data()
    print(f"Loaded {len(rows)} rows.")
    
    print("Calculating metrics...")
    metrics = calculate_metrics(rows)
    
    # Save JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to: {OUTPUT_JSON}")
    
    # Save Report
    report = generate_report(metrics)
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Saved report to: {OUTPUT_MD}")
    
    # Print summary
    print("\n" + "="*50)
    print("HUMAN EVALUATION SUMMARY")
    print("="*50)
    print(f"Precision:         {metrics['overall']['precision']}% (Target: >90%)")
    print(f"Hallucination Rate: {metrics['overall']['hallucination_rate']}% (Target: <1%)")
    print(f"Avg Quality:       {metrics['overall']['avg_quality']}/5 (Target: >4.5)")
    print("="*50)

if __name__ == "__main__":
    main()
