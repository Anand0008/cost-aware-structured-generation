
import json
import glob
from pathlib import Path
from typing import Dict, List, Any
import statistics

# Configuration
OUTPUTS_DIR = Path("c:/Users/anand/Downloads/qbt/debug_outputs/voting_engine")

# Pricing from models_config.yaml (USD per 1M tokens)
PRICING = {
    # The Mix
    "gemini_2.5_pro":    {"input": 0.15, "output": 0.60},
    "deepseek_r1":       {"input": 0.55, "output": 2.19},
    "claude_sonnet_4.5": {"input": 3.00, "output": 15.00},
    "gpt_5_1":           {"input": 1.25, "output": 10.00},
    
    # Router
    "gemini_flash":      {"input": 0.00, "output": 0.00}
}


class CostAnalyzer:
    def __init__(self):
        self.pricing = PRICING
        
    def calculate_cost(self, model_key: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost in USD"""
        p = self.pricing.get(model_key, {"input": 0, "output": 0})
        cost_in = (input_tokens / 1_000_000) * p["input"]
        cost_out = (output_tokens / 1_000_000) * p["output"]
        return cost_in + cost_out

    def scan_questions(self) -> List[str]:
        """Scan all processed question outputs"""
        print(f"Scanning {OUTPUTS_DIR}...")
        files = glob.glob(str(OUTPUTS_DIR / "*_03_final_json.json"))
        return files

    def _estimate_token_usage(self, data: Dict) -> Dict:
        """Estimate tokens based on content length"""
        # 1. Classification (Gemini Flash - Cheap/Free)
        tier_0 = data.get("tier_0_classification", {})
        t0_text = str(tier_0)
        t0_out = len(t0_text) // 4
        t0_in = 1000  # Prompt overhead
        
        # 2. Core Generation
        # Tier 1, 2, 3 content
        tier_content = ""
        for key in ["tier_1_core_research", "tier_2_student_learning", "tier_3_enhanced_learning"]:
            tier_content += str(data.get(key, ""))
            
        gen_out = len(tier_content) // 4
        gen_in = gen_out * 4 # RAG Context Assumption (Input is usually 4x Output)
        
        return {
            "router": {"input": t0_in, "output": t0_out},
            "generation": {"input": gen_in, "output": gen_out}
        }

    def process_data(self):
        """Process all scanned files"""
        files = self.scan_questions()
        results = []
        print(f"Found {len(files)} files. Processing...")
        
        for f in files:
            try:
                with open(f, 'r', encoding='utf-8') as fd:
                    data = json.load(fd)
                
                usage = self._estimate_token_usage(data)
                
                # --- ACTUAL PIPELINE COST ---
                # A. Router (Tier 0)
                cost_router = self.calculate_cost("gemini_flash", usage["router"]["input"], usage["router"]["output"])
                
                # B. Ensemble (3 Models: Gemini Pro + DeepSeek + Claude)
                # They all get the same prompt (gen_in) and generate roughly same output (gen_out)
                c_gem = self.calculate_cost("gemini_2.5_pro", usage["generation"]["input"], usage["generation"]["output"])
                c_deep = self.calculate_cost("deepseek_r1", usage["generation"]["input"], usage["generation"]["output"])
                c_claude = self.calculate_cost("claude_sonnet_4.5", usage["generation"]["input"], usage["generation"]["output"])
                
                # C. GPT-5.1 (Conditional)
                # Check if it was used based on flag
                use_gpt = data.get("tier_0_classification", {}).get("use_gpt51", False)
                c_gpt = 0
                if use_gpt:
                    c_gpt = self.calculate_cost("gpt_5_1", usage["generation"]["input"], usage["generation"]["output"])
                
                # D. Synthesis (Zero Cost Logic)
                # Script logic confirms clean JSON merging in Python, no extra LLM call.
                c_synth = 0
                
                actual_total = cost_router + c_gem + c_deep + c_claude + c_gpt + c_synth
                
                # --- NAIVE BASELINE COSTS ---
                # Baseline 1: Single SOTA (1x Claude) - The "Minimum Viable" approach
                baseline_single_claude = c_claude
                
                # Baseline 2: Self-Consistency SOTA (3x Claude) - The "Maximum Accuracy" approach
                # (Standard research technique: simple output voting to boost reasoning)
                baseline_ensemble_claude = 3 * c_claude
                
                results.append({
                    "id": data.get("question_id", "Unknown"),
                    "actual_cost": actual_total,
                    "baseline_single": baseline_single_claude,
                    "baseline_ensemble": baseline_ensemble_claude,
                    "models_used": 4 if use_gpt else 3
                })
                
            except Exception as e:
                pass
                
        self.generate_report(results)
        
    def generate_report(self, results):
        if not results:
            print("No data found.")
            return
            
        total_actual = sum(r['actual_cost'] for r in results)
        total_base_single = sum(r['baseline_single'] for r in results)
        total_base_ensemble = sum(r['baseline_ensemble'] for r in results)
        
        avg_actual = total_actual / len(results)
        
        # Savings/Cost Delta Calculation
        # vs Single: Usually negative savings (we are more expensive)
        delta_single = ((total_base_single - total_actual) / total_base_single) * 100 if total_base_single else 0
        
        # vs Ensemble: Positive savings
        savings_ensemble = ((total_base_ensemble - total_actual) / total_base_ensemble) * 100 if total_base_ensemble else 0
            
        report = {
            "total_questions": len(results),
            "total_actual_cost": total_actual,
            "avg_cost_per_q": avg_actual,
            
            "baseline_single_total": total_base_single,
            "delta_vs_single_pct": delta_single,
            
            "baseline_ensemble_total": total_base_ensemble,
            "savings_vs_ensemble_pct": savings_ensemble,
            
            "explanation": "Mixed Ensemble vs (1) Single SOTA and (2) Self-Consistency SOTA (3x)"
        }
        
        with open("cost_report.json", "w") as f:
            json.dump(report, f, indent=2)
            
        print("Report saved to cost_report.json")
        print(f"Actual Total: ${total_actual:.2f}")
        print(f"vs Single SOTA (1x): ${total_base_single:.2f} ({delta_single:+.1f}%)")
        print(f"vs Self-Consistency (3x): ${total_base_ensemble:.2f} ({savings_ensemble:+.1f}%)")

if __name__ == "__main__":
    analyzer = CostAnalyzer()
    analyzer.process_data()
