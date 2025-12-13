"""
============================================================================
COST TRACKER UTILITY
============================================================================
Purpose: Track API costs across all models and stages
Features:
    - Per-model cost calculation
    - Per-stage cost aggregation
    - Real-time cost monitoring
    - Budget alerts
    - Cost breakdown reporting
    - Token usage tracking

Usage:
    from utils.cost_tracker import CostTracker
    
    tracker = CostTracker(models_config)
    
    # Calculate cost for API call
    cost = tracker.calculate_cost(
        model_name="gemini_2.5_pro",
        input_tokens=10000,
        output_tokens=4000
    )
    
    # Track cost
    tracker.track_cost(
        model_name="gemini_2.5_pro",
        stage="stage_5",
        cost=cost
    )
    
    # Get report
    report = tracker.get_cost_report()

Author: GATE AE SOTA Pipeline
============================================================================
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from collections import defaultdict

from utils.logging_utils import setup_logger

logger = setup_logger("cost_tracker")


class CostTracker:
    """
    Track costs across all models and pipeline stages
    
    Pricing (per 1M tokens):
    - Gemini 2.5 Pro: $1.25 input, $10.00 output
    - Claude Sonnet 4.5: $3.00 input, $15.00 output
    - DeepSeek R1: $0.55 input, $2.19 output
    - GPT-5.1: $1.25 input, $10.00 output
    - Gemini 2.0 Flash Exp: FREE (or negligible)
    """
    
    def __init__(self, models_config: Dict):
        """
        Args:
            models_config: Models configuration dict with pricing info
        """
        self.models_config = models_config
        
        # Extract pricing for each model
        self.pricing = {}
        for model_name, config in models_config.get('models', {}).items():
            # Get pricing from nested 'pricing' dict with correct keys
            pricing_config = config.get('pricing', {})
            self.pricing[model_name] = {
                'input': pricing_config.get('input_per_1m', 0),
                'output': pricing_config.get('output_per_1m', 0)
            }
        
        # Cost tracking structures
        self.costs_by_model = defaultdict(float)
        self.costs_by_stage = defaultdict(float)
        self.costs_by_question = {}
        
        # Token tracking
        self.tokens_by_model = defaultdict(lambda: {'input': 0, 'output': 0})
        
        # API call counts
        self.api_calls_by_model = defaultdict(int)
        
        # Metadata
        self.start_time = datetime.utcnow()
        self.total_questions_processed = 0
    
    def calculate_cost(
        self,
        model_name: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """
        Calculate cost for an API call
        
        Args:
            model_name: Model name (e.g., "gemini_2.5_pro")
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
        
        Returns:
            float: Cost in USD
        
        Example:
            cost = tracker.calculate_cost(
                model_name="claude_sonnet_4.5",
                input_tokens=10000,
                output_tokens=4000
            )
            # Returns: 0.09 (= $3/1M * 10k + $15/1M * 4k)
        """
        if model_name not in self.pricing:
            logger.warning(f"Unknown model for pricing: {model_name}")
            return 0.0
        
        pricing = self.pricing[model_name]
        
        # Calculate cost (pricing is per 1M tokens)
        input_cost = (input_tokens / 1_000_000) * pricing['input']
        output_cost = (output_tokens / 1_000_000) * pricing['output']
        
        total_cost = input_cost + output_cost
        
        return total_cost
    
    def track_cost(
        self,
        model_name: str,
        stage: str,
        cost: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        question_id: Optional[str] = None
    ):
        """
        Track cost for an API call
        
        Args:
            model_name: Model name
            stage: Pipeline stage (e.g., "stage_5", "stage_7_debate")
            cost: Cost in USD
            input_tokens: Input tokens used
            output_tokens: Output tokens used
            question_id: Question ID (optional, for per-question tracking)
        """
        # Track by model
        self.costs_by_model[model_name] += cost
        
        # Track by stage
        self.costs_by_stage[stage] += cost
        
        # Track tokens
        self.tokens_by_model[model_name]['input'] += input_tokens
        self.tokens_by_model[model_name]['output'] += output_tokens
        
        # Track API calls
        self.api_calls_by_model[model_name] += 1
        
        # Track by question
        if question_id:
            if question_id not in self.costs_by_question:
                self.costs_by_question[question_id] = {
                    'total': 0.0,
                    'by_model': defaultdict(float),
                    'by_stage': defaultdict(float)
                }
            
            self.costs_by_question[question_id]['total'] += cost
            self.costs_by_question[question_id]['by_model'][model_name] += cost
            self.costs_by_question[question_id]['by_stage'][stage] += cost
    
    def get_total_cost(self) -> float:
        """
        Get total cost across all models and stages
        
        Returns:
            float: Total cost in USD
        """
        return sum(self.costs_by_model.values())
    
    def get_cost_by_model(self, model_name: str) -> float:
        """Get cost for specific model"""
        return self.costs_by_model.get(model_name, 0.0)
    
    def get_cost_by_stage(self, stage: str) -> float:
        """Get cost for specific stage"""
        return self.costs_by_stage.get(stage, 0.0)
    
    def get_cost_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive cost report
        
        Returns:
            dict: {
                "summary": {...},
                "by_model": {...},
                "by_stage": {...},
                "by_question": {...},
                "tokens": {...},
                "api_calls": {...}
            }
        """
        total_cost = self.get_total_cost()
        elapsed_time = (datetime.utcnow() - self.start_time).total_seconds()
        
        # Calculate total tokens
        total_input_tokens = sum(t['input'] for t in self.tokens_by_model.values())
        total_output_tokens = sum(t['output'] for t in self.tokens_by_model.values())
        total_tokens = total_input_tokens + total_output_tokens
        
        # Calculate total API calls
        total_api_calls = sum(self.api_calls_by_model.values())
        
        report = {
            "summary": {
                "total_cost_usd": round(total_cost, 4),
                "total_questions_processed": self.total_questions_processed,
                "avg_cost_per_question": round(
                    total_cost / max(self.total_questions_processed, 1), 4
                ),
                "total_api_calls": total_api_calls,
                "total_tokens": total_tokens,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "elapsed_time_seconds": round(elapsed_time, 1),
                "timestamp": datetime.utcnow().isoformat()
            },
            "by_model": {
                model: {
                    "cost_usd": round(cost, 4),
                    "percentage": round((cost / total_cost * 100) if total_cost > 0 else 0, 2),
                    "api_calls": self.api_calls_by_model[model],
                    "input_tokens": self.tokens_by_model[model]['input'],
                    "output_tokens": self.tokens_by_model[model]['output'],
                    "total_tokens": (
                        self.tokens_by_model[model]['input'] + 
                        self.tokens_by_model[model]['output']
                    )
                }
                for model, cost in sorted(
                    self.costs_by_model.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
            },
            "by_stage": {
                stage: {
                    "cost_usd": round(cost, 4),
                    "percentage": round((cost / total_cost * 100) if total_cost > 0 else 0, 2)
                }
                for stage, cost in sorted(
                    self.costs_by_stage.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
            },
            "top_expensive_questions": self._get_top_expensive_questions(10),
            "pricing_reference": self.pricing
        }
        
        return report
    
    def _get_top_expensive_questions(self, top_n: int = 10) -> list:
        """Get top N most expensive questions"""
        sorted_questions = sorted(
            self.costs_by_question.items(),
            key=lambda x: x[1]['total'],
            reverse=True
        )
        
        return [
            {
                "question_id": qid,
                "cost_usd": round(data['total'], 4),
                "by_model": {k: round(v, 4) for k, v in data['by_model'].items()},
                "by_stage": {k: round(v, 4) for k, v in data['by_stage'].items()}
            }
            for qid, data in sorted_questions[:top_n]
        ]
    
    def print_report(self):
        """Print formatted cost report to console"""
        report = self.get_cost_report()
        
        print("\n" + "="*80)
        print("COST REPORT")
        print("="*80)
        
        # Summary
        summary = report['summary']
        print(f"\nTotal Cost: ${summary['total_cost_usd']:.4f}")
        print(f"Questions Processed: {summary['total_questions_processed']}")
        print(f"Avg Cost/Question: ${summary['avg_cost_per_question']:.4f}")
        print(f"Total API Calls: {summary['total_api_calls']}")
        print(f"Total Tokens: {summary['total_tokens']:,}")
        print(f"  - Input: {summary['total_input_tokens']:,}")
        print(f"  - Output: {summary['total_output_tokens']:,}")
        
        # By model
        print(f"\n{'Model':<25} {'Cost':<12} {'%':<8} {'API Calls':<12} {'Tokens':<15}")
        print("-"*80)
        
        for model, data in report['by_model'].items():
            print(
                f"{model:<25} "
                f"${data['cost_usd']:<11.4f} "
                f"{data['percentage']:<7.2f}% "
                f"{data['api_calls']:<12} "
                f"{data['total_tokens']:<15,}"
            )
        
        # By stage
        print(f"\n{'Stage':<30} {'Cost':<12} {'%':<8}")
        print("-"*80)
        
        for stage, data in report['by_stage'].items():
            print(
                f"{stage:<30} "
                f"${data['cost_usd']:<11.4f} "
                f"{data['percentage']:<7.2f}%"
            )
        
        # Top expensive questions
        if report['top_expensive_questions']:
            print(f"\nTop 10 Most Expensive Questions:")
            print("-"*80)
            
            for i, q in enumerate(report['top_expensive_questions'], 1):
                print(f"{i}. {q['question_id']}: ${q['cost_usd']:.4f}")
        
        print("="*80 + "\n")
    
    def save_report(self, output_path: str):
        """
        Save cost report to JSON file
        
        Args:
            output_path: Path to save report
        """
        report = self.get_cost_report()
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Cost report saved to: {output_path}")
    
    def check_budget(self, budget_limit: float) -> Dict[str, Any]:
        """
        Check if cost is within budget
        
        Args:
            budget_limit: Budget limit in USD
        
        Returns:
            dict: {
                "within_budget": bool,
                "total_cost": float,
                "budget_limit": float,
                "remaining": float,
                "percentage_used": float,
                "alert_level": str (OK, WARNING, CRITICAL, EXCEEDED)
            }
        """
        total_cost = self.get_total_cost()
        remaining = budget_limit - total_cost
        percentage_used = (total_cost / budget_limit * 100) if budget_limit > 0 else 0
        
        # Determine alert level
        if percentage_used >= 100:
            alert_level = "EXCEEDED"
        elif percentage_used >= 90:
            alert_level = "CRITICAL"
        elif percentage_used >= 75:
            alert_level = "WARNING"
        else:
            alert_level = "OK"
        
        return {
            "within_budget": total_cost <= budget_limit,
            "total_cost": round(total_cost, 4),
            "budget_limit": budget_limit,
            "remaining": round(remaining, 4),
            "percentage_used": round(percentage_used, 2),
            "alert_level": alert_level
        }
    
    def increment_questions_processed(self):
        """Increment total questions processed counter"""
        self.total_questions_processed += 1
    
    def reset(self):
        """Reset all tracking data"""
        self.costs_by_model.clear()
        self.costs_by_stage.clear()
        self.costs_by_question.clear()
        self.tokens_by_model.clear()
        self.api_calls_by_model.clear()
        self.start_time = datetime.utcnow()
        self.total_questions_processed = 0
        
        logger.info("Cost tracker reset")


# Example usage
if __name__ == "__main__":
    # Mock models config
    models_config = {
        "models": {
            "gemini_2.5_pro": {
                "cost_per_million_input_tokens": 1.25,
                "cost_per_million_output_tokens": 10.00
            },
            "claude_sonnet_4.5": {
                "cost_per_million_input_tokens": 3.00,
                "cost_per_million_output_tokens": 15.00
            },
            "deepseek_r1": {
                "cost_per_million_input_tokens": 0.55,
                "cost_per_million_output_tokens": 2.19
            },
            "gpt_5.1": {
                "cost_per_million_input_tokens": 1.25,
                "cost_per_million_output_tokens": 10.00
            }
        }
    }
    
    # Create tracker
    tracker = CostTracker(models_config)
    
    # Simulate some API calls
    for i in range(100):
        # Gemini call
        cost = tracker.calculate_cost("gemini_2.5_pro", 10000, 4000)
        tracker.track_cost(
            "gemini_2.5_pro",
            "stage_5",
            cost,
            10000,
            4000,
            f"Q{i}"
        )
        
        # Claude call
        cost = tracker.calculate_cost("claude_sonnet_4.5", 10000, 4000)
        tracker.track_cost(
            "claude_sonnet_4.5",
            "stage_5",
            cost,
            10000,
            4000,
            f"Q{i}"
        )
        
        # DeepSeek call
        cost = tracker.calculate_cost("deepseek_r1", 10000, 4000)
        tracker.track_cost(
            "deepseek_r1",
            "stage_5",
            cost,
            10000,
            4000,
            f"Q{i}"
        )
        
        tracker.increment_questions_processed()
    
    # Print report
    tracker.print_report()
    
    # Check budget
    budget_check = tracker.check_budget(420.0)
    print(f"\nBudget Check: {budget_check}")