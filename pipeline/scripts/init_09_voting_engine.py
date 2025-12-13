"""
============================================================================
STAGE 6: HYBRID CONSENSUS ENGINE (VOTING)
============================================================================
Purpose: Robustly combine model responses using a Hybrid Consensus approach.

Core Logic:
1.  **Red Line Safety Check**: If ANY model says `is_correct: False`, DISPUTE immediately.
2.  **Deterministic Processing**:
    -   Identity fields: Strict Majority Vote.
    -   Numeric fields: Weighted Average.
3.  **Semantic Synthesis** (Field-Level Batching):
    -   Fields split into ~8 batches for granular control.
    -   LLM synthesizes each batch with STRICT instructions.
    -   **STRICT AUDIT**: Verifies every key exists in response.

Fixes Applied:
- Issue 1: Redundant Steps → LLM prompt instructs single chronological merge
- Issue 4: Missing URLs → LLM prompt preserves all URLs exactly
- Issue 5: Difficulty Mismatch → Syncs tier_0 score with tier_1 average
- Issue 6: Semantic Duplicates → LLM prompt deduplicates semantically
- Issue 7: Consensus Label → Sets to "synthesized" in list items
- Issue 8: Formula Formatting → LLM prompt requires pure LaTeX only

Used by: 99_pipeline_runner.py (Stage 6)
Author: GATE AE SOTA Pipeline
============================================================================
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Set
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Project setup
PROJECT_ROOT = Path(__file__).parent.parent
import sys
sys.path.append(str(PROJECT_ROOT))

from utils.logging_utils import setup_logger, log_stage

logger = setup_logger("09_voting_engine")

# Debug output directory
DEBUG_DIR = PROJECT_ROOT.parent / "debug_outputs" / "voting_engine"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)


class IncompleteLLMResponseError(Exception):
    """Raised when LLM response misses required keys"""
    pass


# ============================================================================
# SYNTHESIZER SYSTEM PROMPT (Fixes Issues 1, 4, 6, 8)
# ============================================================================
SYNTHESIZER_SYSTEM_PROMPT = """You are a Robust Consensus Engine for GATE exam question tagging.

TASK: Synthesize a single consensus value for EACH field path from multiple model outputs.

## STRICT RULES (FOLLOW EXACTLY):

### 1. DEDUPLICATION (Fixes duplicate concepts)
- For lists like prerequisites, mistakes, concepts: IDENTIFY semantically identical items and MERGE them.
- If two items mean the same thing but have slightly different wording, keep only ONE.
- Example: "Hooke's Law" and "Generalized Hooke's Law" → Keep "Generalized Hooke's Law" only.

### 2. STEP-BY-STEP SOLUTIONS (Fixes redundant steps)
- For `step_by_step` field: Merge ALL models' logic into ONE SINGLE chronological path.
- DO NOT append Model A's steps after Model B's steps.
- Remove any redundant or repeated steps.
- The final solution should read as ONE coherent walkthrough.

### 3. URL PRESERVATION (Fixes missing URLs)
- PRESERVE all `video_url`, `book`, and reference URLs EXACTLY as found in input.
- If any model has a valid URL, include it in the output.
- NEVER drop valid URLs. NEVER hallucinate new ones.

### 4. FORMULA FORMATTING (Fixes LaTeX issues)
- The `formula` field must contain ONLY pure mathematical LaTeX.
- Do NOT wrap formulas in `$` or `$$` delimiters.
- Move any explanatory text (like "implies incompressible") to `conditions` or `notes` field.
- Example BAD: "K = E/(3(1-2\\nu)) \\implies \\text{incompressible}"
- Example GOOD: "K = \\frac{E}{3(1-2\\nu)}" with conditions: ["When ν = 0.5, material is incompressible"]

### 5. CONFLICT RESOLUTION
- For factual conflicts (different formulas, different numbers): Pick the value from the HIGHEST-WEIGHTED model.
- For subjective text: Synthesize best explanation combining insights.

### 6. OUTPUT FORMAT
- Return a JSON object where keys are the exact field paths provided.
- Each value is the synthesized consensus value.
- You MUST return a value for EVERY field path listed. Do NOT skip any.
- CONSENSUS LABELS: For any object with a 'consensus' field, set the value to 'ensemble'.
"""


class VotingEngine:
    """
    Hybrid Consensus Engine with Field-Level Batching.
    
    Key Features:
    - Red Line safety check
    - Weighted averaging for numeric stats
    - Field-level batching (~8 batches) for synthesis
    - Strict nested field audit
    - Difficulty score synchronization (tier_0 ↔ tier_1)
    - Debug output saving
    """
    
    def __init__(self, configs: Dict, clients: Dict):
        self.configs = configs
        self.clients = clients
        self.genai = clients.get('google_genai')
        
        # Store weights for metadata (Fix Issue 2)
        self.current_weights = {}
        
        # Load synthesis model config
        self.synth_model_config = configs['models_config']['models']['gemini_2.5_pro']
        
        # Identity fields (strict majority)
        self.identity_fields = ['question_id', 'year', 'subject', 'exam_name']
        
        # Numeric fields (weighted average)
        self.numeric_fields = [
            'tier_0_classification.classification_confidence',
            'tier_0_classification.difficulty_score',
            'tier_1_core_research.answer_validation.confidence',
            'tier_1_core_research.difficulty_analysis.score',
            'tier_1_core_research.explanation.estimated_time_minutes',
            'tier_4_metadata_and_future.accuracy_percent'
        ]
        
        # Current question_id for debug saving
        self.current_question_id = None

    @log_stage("Stage 6: Hybrid Consensus")
    def vote_on_responses(
        self,
        model_responses: Dict[str, Dict],
        weights: Dict[str, float],
        question: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Main voting flow with field-level batching."""
        self.current_question_id = question.get('question_id', 'unknown')
        self.current_weights = weights  # Store for metadata (Fix Issue 2)
        logger.info(f"Processing Consensus for {self.current_question_id}")
        synthesis_logs = []
        
        # 1. Extract model JSONs
        model_jsons = {
            k: v.get('response_json', {}) 
            for k, v in model_responses.items() if v
        }
        
        if not model_jsons:
            logger.error("No valid model responses to vote on.")
            return {"disputed_fields": [], "consensus_score": 0.0, "final_json": {}, "synthesis_logs": [], "weights_used": weights}
        
        # Save raw model responses for debugging
        self._save_debug("01_model_jsons", model_jsons)
        
        # 2. RED LINE CHECK
        if self._check_red_line(model_jsons):
            logger.warning("🚨 RED LINE TRIGGERED: Answer validation failure detected.")
            synthesis_logs.append("Red Line Triggered - Answer marked incorrect by at least one model")
            return {
                "converged_fields": {},
                "disputed_fields": ["RED_LINE_FAILURE"],
                "consensus_score": 0.0,
                "final_json": None,
                "synthesis_logs": synthesis_logs,
                "weights_used": weights
            }
        
        # 3. Initialize final output structure from best model (as template)
        best_model = max(weights, key=weights.get)
        final_json = json.loads(json.dumps(model_jsons.get(best_model, {})))  # Deep copy
        
        # 4. PHASE 2: Identity Fields (Strict Majority)
        for field in self.identity_fields:
            majority_val = self._get_majority_value(model_jsons, field)
            if majority_val is not None:
                final_json[field] = majority_val
        
        # 5. PHASE 2: Numeric Fields (Weighted Average)
        averaged_stats = {}
        for path in self.numeric_fields:
            avg_val = self._calculate_weighted_average(model_jsons, path, weights)
            if avg_val is not None:
                self._set_nested_value(final_json, path, avg_val)
                averaged_stats[path] = avg_val
        
        # FIX ISSUE 5: Sync difficulty scores
        # Use tier_1 difficulty_analysis.score as the authoritative source
        tier1_difficulty = averaged_stats.get('tier_1_core_research.difficulty_analysis.score')
        if tier1_difficulty is not None:
            self._set_nested_value(final_json, 'tier_0_classification.difficulty_score', tier1_difficulty)
            averaged_stats['tier_0_classification.difficulty_score'] = tier1_difficulty
            logger.info(f"  Synced difficulty_score to {tier1_difficulty}")
        
        # 6. PHASE 3: Field-Level Synthesis (8 Batches)
        all_field_paths = self._get_all_leaf_paths(model_jsons)
        
        # Remove already-handled fields
        paths_to_synthesize = [
            p for p in all_field_paths 
            if not any(p.startswith(nf) or p == nf for nf in self.numeric_fields)
            and p not in self.identity_fields
        ]
        
        # Split into ~8 batches
        batches = self._split_into_batches(paths_to_synthesize, num_batches=8)
        self._save_debug("02_batches", {"batch_count": len(batches), "batches": batches})
        
        for i, batch_paths in enumerate(batches):
            batch_name = f"batch_{i+1}_of_{len(batches)}"
            logger.info(f"  Synthesizing {batch_name} ({len(batch_paths)} fields)")
            
            batch_input = self._prepare_field_batch(model_jsons, batch_paths, weights)
            
            try:
                batch_result = self._call_synthesizer_with_audit(batch_name, batch_input, batch_paths)
                
                # FIX ISSUE 7: Set consensus label to "synthesized" for list items
                batch_result = self._fix_consensus_labels(batch_result)
                
                for path, value in batch_result.items():
                    self._set_nested_value(final_json, path, value)
                
                synthesis_logs.append(f"{batch_name}: Success ({len(batch_paths)} fields)")
                
            except Exception as e:
                logger.error(f"Failed to synthesize {batch_name}: {e}")
                synthesis_logs.append(f"{batch_name}: Failed ({e}) - Using best model fallback")
        
        # Save final output for debugging
        self._save_debug("03_final_json", final_json)
        self._save_debug("04_synthesis_logs", synthesis_logs)
        
        return {
            "converged_fields": averaged_stats,
            "disputed_fields": [],
            "consensus_score": 1.0,
            "final_json": final_json,
            "synthesis_logs": synthesis_logs,
            "weights_used": weights  # Pass weights through for metadata (Fix Issue 2)
        }

    def _fix_consensus_labels(self, batch_result: Dict) -> Dict:
        """Fix Issue 7: Set consensus field to 'synthesized' in list items."""
        def fix_recursive(obj):
            if isinstance(obj, dict):
                if 'consensus' in obj:
                    obj['consensus'] = 'ensemble'
                for v in obj.values():
                    fix_recursive(v)
            elif isinstance(obj, list):
                for item in obj:
                    fix_recursive(item)
        
        fix_recursive(batch_result)
        return batch_result

    def _check_red_line(self, model_jsons: Dict[str, Dict]) -> bool:
        """Return True if ANY model says is_correct=False"""
        path = "tier_1_core_research.answer_validation.is_correct"
        for name, data in model_jsons.items():
            val = self._get_nested_value(data, path)
            if val is False:
                logger.warning(f"  Model {name} flagged answer as INCORRECT.")
                return True
        return False

    def _get_majority_value(self, model_jsons: Dict, field: str):
        """Simple majority vote for identity fields"""
        values = [self._get_nested_value(data, field) for data in model_jsons.values()]
        values = [v for v in values if v is not None]
        if not values:
            return None
        return max(set(values), key=values.count)

    def _calculate_weighted_average(self, model_jsons: Dict, path: str, weights: Dict) -> float:
        """Calculate weighted average for numeric fields"""
        total_weight = 0.0
        weighted_sum = 0.0
        
        for name, data in model_jsons.items():
            val = self._get_nested_value(data, path)
            if isinstance(val, (int, float)):
                w = weights.get(name, 0.0)
                weighted_sum += val * w
                total_weight += w
        
        if total_weight == 0:
            return None
        return round(weighted_sum / total_weight, 2)

    def _get_all_leaf_paths(self, model_jsons: Dict) -> List[str]:
        """Get all unique leaf field paths from all models"""
        all_paths = set()
        for data in model_jsons.values():
            paths = self._extract_leaf_paths(data, "")
            all_paths.update(paths)
        return sorted(all_paths)

    def _extract_leaf_paths(self, data: Any, prefix: str) -> Set[str]:
        """Recursively extract all leaf paths"""
        paths = set()
        
        if isinstance(data, dict):
            for key, value in data.items():
                new_prefix = f"{prefix}.{key}" if prefix else key
                if isinstance(value, (dict, list)) and value:
                    paths.update(self._extract_leaf_paths(value, new_prefix))
                else:
                    paths.add(new_prefix)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                new_prefix = f"{prefix}[{i}]"
                if isinstance(item, (dict, list)) and item:
                    paths.update(self._extract_leaf_paths(item, new_prefix))
                else:
                    paths.add(new_prefix)
        
        return paths

    def _split_into_batches(self, paths: List[str], num_batches: int) -> List[List[str]]:
        """Split paths into N roughly equal batches"""
        if not paths:
            return []
        batch_size = max(1, len(paths) // num_batches)
        batches = []
        for i in range(0, len(paths), batch_size):
            batches.append(paths[i:i + batch_size])
        return batches

    def _prepare_field_batch(self, model_jsons: Dict, paths: List[str], weights: Dict) -> Dict:
        """Prepare batch input with model values for each path"""
        batch_data = {}
        for path in paths:
            batch_data[path] = {}
            for name, data in model_jsons.items():
                val = self._get_nested_value(data, path)
                if val is not None:
                    batch_data[path][name] = {
                        "value": val,
                        "weight": weights.get(name, 0.25)
                    }
        return batch_data

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((IncompleteLLMResponseError, Exception))
    )
    def _call_synthesizer_with_audit(self, batch_name: str, batch_input: Dict, expected_paths: List[str]) -> Dict:
        """Call LLM to synthesize with strict audit."""
        user_prompt = f"""
FIELD PATHS TO SYNTHESIZE:
{json.dumps(expected_paths, indent=2)}

MODEL VALUES (with weights):
{json.dumps(batch_input, indent=2)}

Return ONLY the JSON object with synthesized values for each path. Do NOT skip any field."""

        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        try:
            model = self.genai.GenerativeModel(
                model_name=self.synth_model_config['model_id'],
                generation_config={
                    "temperature": 0.1,
                    "response_mime_type": "application/json"
                },
                safety_settings=safety_settings,
                system_instruction=SYNTHESIZER_SYSTEM_PROMPT
            )
            
            response = model.generate_content(user_prompt)
            result = json.loads(response.text)
            
            # STRICT AUDIT
            self._audit_response_strict(expected_paths, result, batch_name)
            
            return result
            
        except IncompleteLLMResponseError as e:
            logger.warning(f"{batch_name} failed audit: {e}. Retrying...")
            self._save_debug(f"audit_failure_{batch_name}", {"error": str(e)})
            raise e
        except Exception as e:
            logger.error(f"LLM Call failed for {batch_name}: {e}")
            raise e

    def _audit_response_strict(self, expected_paths: List[str], response_json: Dict, batch_name: str):
        """Verify ALL expected paths exist in response"""
        missing = [p for p in expected_paths if p not in response_json]
        
        if missing:
            self._save_debug(f"audit_missing_{batch_name}", {"missing": missing, "total": len(expected_paths)})
            raise IncompleteLLMResponseError(f"Missing {len(missing)}/{len(expected_paths)} fields")

    def _get_nested_value(self, data: Dict, path: str) -> Any:
        """Get value from nested dict using path string"""
        try:
            parts = path.replace('[', '.').replace(']', '').split('.')
            value = data
            for part in parts:
                if part.isdigit():
                    value = value[int(part)]
                else:
                    value = value[part]
            return value
        except (KeyError, IndexError, TypeError):
            return None

    def _set_nested_value(self, data: Dict, path: str, value: Any):
        """Set value in nested dict using path string"""
        parts = path.replace('[', '.[').split('.')
        parts = [p for p in parts if p]
        
        current = data
        for i, part in enumerate(parts[:-1]):
            if part.startswith('['):
                idx = int(part[1:-1])
                while len(current) <= idx:
                    current.append({})
                current = current[idx]
            else:
                if part not in current:
                    next_part = parts[i+1] if i+1 < len(parts) else ""
                    if next_part.startswith('['):
                        current[part] = []
                    else:
                        current[part] = {}
                current = current[part]
        
        final_part = parts[-1]
        if final_part.startswith('['):
            idx = int(final_part[1:-1])
            while len(current) <= idx:
                current.append(None)
            current[idx] = value
        else:
            current[final_part] = value

    def _save_debug(self, name: str, data: Any):
        """Save debug output to file"""
        if self.current_question_id:
            filename = f"{self.current_question_id}_{name}.json"
        else:
            filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{name}.json"
        
        filepath = DEBUG_DIR / filename
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            logger.warning(f"Failed to save debug {filename}: {e}")