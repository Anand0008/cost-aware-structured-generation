"""
============================================================================
STAGE 8: SYNTHESIS ENGINE
============================================================================
Purpose: Merge all sources into final complete JSON output
Process:
    1. Take converged fields from voting (Stage 6)
    2. Take resolved fields from debate (Stage 7)
    3. Intelligently merge arrays/objects from all models
    4. Calculate tier_4_metadata (quality, costs, timing)
    5. Validate against complete schema
    6. Return final 200+ field JSON

Sources merged:
    - Voting engine consensus (Stage 6)
    - Debate resolution (Stage 7)
    - Direct model responses (for arrays/complex fields)
    - RAG metadata
    - Processing metadata

Used by: 99_pipeline_runner.py (Stage 8)
Author: GATE AE SOTA Pipeline

Output: Complete JSON with tier_0 through tier_4 (~200+ fields)
Processing Time: <1 second (no API calls)
============================================================================
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Union
from datetime import datetime
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from utils.logging_utils import setup_logger, log_stage
from utils.similarity_utils import merge_arrays_by_similarity
from utils.json_validator import validate_complete_schema

logger = setup_logger("11_synthesis_engine")


class SynthesisEngine:
    """
    Synthesize final output from all pipeline stages
    
    Responsibilities:
    1. Field-by-field resolution using priority order
    2. Intelligent array merging (deduplicate, combine unique items)
    3. Quality score calculation (5 metrics weighted)
    4. Metadata generation (costs, tokens, timing)
    5. Schema validation
    """
    
    def __init__(self, configs: Dict):
        """
        Args:
            configs: All configuration dictionaries
        """
        self.configs = configs
        
        # Quality score weights
        self.quality_weights = {
            'avg_model_confidence': 0.25,
            'consensus_rate': 0.30,
            'debate_efficiency': 0.20,
            'rag_relevance': 0.15,
            'field_completeness': 0.10
        }
    
    @log_stage("Stage 8: Synthesis Engine")
    def synthesize(
        self,
        question: Dict[str, Any],
        classification: Dict[str, Any],
        voting_result: Dict[str, Any],
        debate_result: Dict[str, Any],
        model_responses: Dict[str, Dict],
        rag_chunks: List[Dict],
        stage_timings: Dict[str, float],
        total_cost: float
    ) -> Dict[str, Any]:
        """
        Synthesize final complete JSON
        
        Args:
            question: Original question (14 root fields)
            classification: tier_0_classification (8 fields)
            voting_result: Results from Stage 6
            debate_result: Results from Stage 7
            model_responses: All model responses with metadata
            rag_chunks: Retrieved chunks
            stage_timings: Time taken per stage
            total_cost: Total cost so far
        
        Returns:
            dict: Complete JSON with tier_0 through tier_4
        """
        question_id = question.get('question_id', 'unknown')
        logger.info(f"Synthesizing final output for: {question_id}")
        
        # Start with root fields and tier_0
        final_output = {
            **self._get_root_fields(question),
            "tier_0_classification": classification
        }
        
        # Resolve all tier_1 through tier_3 fields
        for tier_num in range(1, 4):
            tier_name = f"tier_{tier_num}"
            
            tier_data = self._resolve_tier(
                tier_name=tier_name,
                voting_result=voting_result,
                debate_result=debate_result,
                model_responses=model_responses
            )
            
            final_output[tier_name + self._get_tier_suffix(tier_num)] = tier_data
        
        # Generate tier_4_metadata
        final_output["tier_4_metadata_and_future"] = self._generate_tier4_metadata(
            model_responses=model_responses,
            voting_result=voting_result,
            debate_result=debate_result,
            rag_chunks=rag_chunks,
            stage_timings=stage_timings,
            total_cost=total_cost,
            classification=classification
        )
        
        # Validate schema
        is_valid, errors = validate_complete_schema(final_output)
        
        if not is_valid:
            logger.warning(f"Schema validation found issues: {errors[:3]}")
        else:
            logger.info("  ✓ Schema validation passed")
        
        # Calculate statistics
        total_fields = self._count_fields(final_output)
        logger.info(f"  ✓ Synthesis complete: {total_fields} total fields")
        
        return final_output
    
    def _get_root_fields(self, question: Dict) -> Dict:
        """Extract 14 root fields from question"""
        # Strip base64 from image metadata for final output (too large)
        image_meta = None
        if question.get('image_metadata'):
            image_meta = question['image_metadata'].copy()
            if 'base64' in image_meta:
                del image_meta['base64']

        return {
            "question_id": question['question_id'],
            "exam_name": question.get('exam_name', 'GATE AE'),
            "subject": question.get('subject', 'Aerospace Engineering'),
            "year": question['year'],
            "question_number": question.get('question_number'),
            "question_text": question['question_text'],
            "question_text_latex": question.get('question_text_latex'),
            "question_type": question['question_type'],
            "marks": question['marks'],
            "negative_marks": question.get('negative_marks', 0),
            "options": question.get('options'),
            "answer_key": question.get('answer_key'),
            "has_question_image": question.get('has_question_image', False),
            "image_metadata": image_meta
        }
    
    def _get_tier_suffix(self, tier_num: int) -> str:
        """Get tier name suffix"""
        suffixes = {
            1: "_core_research",
            2: "_student_learning", 
            3: "_enhanced_learning"
        }
        return suffixes.get(tier_num, "")
    
    def _resolve_tier(
        self,
        tier_name: str,
        voting_result: Dict,
        debate_result: Dict,
        model_responses: Dict
    ) -> Dict:
        """
        Resolve all fields in a tier
        
        Priority order:
        1. VotingEngine's synthesized final_json (NEW - highest priority)
        2. Debate resolved fields
        3. Voting converged fields
        4. Merged from all models (for arrays/objects)
        
        Args:
            tier_name: e.g., "tier_1"
            voting_result: Voting results
            debate_result: Debate results
            model_responses: All model responses
        
        Returns:
            dict: Complete tier data
        """
        # NEW: Check if VotingEngine already synthesized this tier
        final_json = voting_result.get('final_json', {})
        full_tier_name = tier_name + self._get_tier_suffix_from_name(tier_name)
        
        if final_json and full_tier_name in final_json:
            tier_data = final_json.get(full_tier_name)
            if tier_data and isinstance(tier_data, dict) and len(tier_data) > 0:
                logger.info(f"  Using pre-synthesized data for {full_tier_name} ({len(tier_data)} top-level keys)")
                return tier_data
        
        # Fallback to original logic
        tier_data = {}
        
        # Get all fields that should be in this tier from model responses
        all_model_tiers = [
            resp['response_json'].get(full_tier_name, {})
            for resp in model_responses.values()
            if resp and 'response_json' in resp
        ]
        
        if not all_model_tiers:
            logger.warning(f"No model provided {tier_name}")
            return {}
        
        # Get all unique field paths in this tier
        field_paths = set()
        for tier in all_model_tiers:
            field_paths.update(self._get_field_paths_in_object(tier, tier_name))
        
        # Resolve each field
        for field_path in field_paths:
            value = self._resolve_field(
                field_path=field_path,
                voting_result=voting_result,
                debate_result=debate_result,
                model_responses=model_responses
            )
            
            # Set value in tier_data
            self._set_nested_value(tier_data, field_path, value)
        
        return tier_data
    
    def _get_tier_suffix_from_name(self, tier_name: str) -> str:
        """Extract tier number and return suffix"""
        tier_num = int(tier_name.split('_')[1])
        return self._get_tier_suffix(tier_num)
    
    def _get_field_paths_in_object(self, obj: Any, prefix: str = "") -> List[str]:
        """Get all field paths in an object (non-recursive for top level)"""
        paths = []
        
        if isinstance(obj, dict):
            for key in obj.keys():
                path = f"{prefix}.{key}" if prefix else key
                paths.append(path)
        
        return paths
    
    def _resolve_field(
        self,
        field_path: str,
        voting_result: Dict,
        debate_result: Dict,
        model_responses: Dict
    ) -> Any:
        """
        Resolve a single field using priority order
        
        Priority:
        1. Debate resolved (if exists)
        2. Voting converged (if exists)
        3. Merge from all models
        
        Args:
            field_path: Full field path (e.g., "tier_1.concepts[0].name")
            voting_result: Voting results
            debate_result: Debate results
            model_responses: All model responses
        
        Returns:
            Any: Resolved field value
        """
        # Priority 1: Debate resolved
        if field_path in debate_result.get('resolved_fields', {}):
            return debate_result['resolved_fields'][field_path]
        
        # Priority 2: Voting converged
        if field_path in voting_result.get('converged_fields', {}):
            return voting_result['converged_fields'][field_path]
        
        # Priority 3: Merge from all models
        values_from_models = []
        
        for model_name, resp_data in model_responses.items():
            if not resp_data or 'response_json' not in resp_data:
                continue
            
            value = self._get_nested_value(resp_data['response_json'], field_path)
            
            if value is not None:
                values_from_models.append(value)
        
        # Merge based on type
        if not values_from_models:
            return None
        
        sample_value = values_from_models[0]
        
        if isinstance(sample_value, list):
            # Merge arrays (deduplicate)
            return merge_arrays_by_similarity(values_from_models)
        
        elif isinstance(sample_value, dict):
            # Merge objects (this is complex, use first non-empty)
            for val in values_from_models:
                if val:
                    return val
            return {}
        
        else:
            # Primitive: use most common
            from collections import Counter
            counts = Counter([str(v) for v in values_from_models])
            most_common_str = counts.most_common(1)[0][0]
            
            # Convert back to original type
            for val in values_from_models:
                if str(val) == most_common_str:
                    return val
            
            return values_from_models[0]
    
    def _get_nested_value(self, data: Dict, field_path: str) -> Any:
        """Get value from nested dict using field path"""
        try:
            parts = field_path.replace('[', '.').replace(']', '').split('.')
            value = data
            
            for part in parts:
                if part.isdigit():
                    value = value[int(part)]
                else:
                    value = value[part]
            
            return value
        except (KeyError, IndexError, TypeError):
            return None
    
    def _set_nested_value(self, data: Dict, field_path: str, value: Any):
        """Set value in nested dict using field path"""
        # Remove tier prefix if present
        parts = field_path.split('.')
        if parts[0].startswith('tier_'):
            parts = parts[1:]  # Remove tier prefix
        
        # Navigate to parent
        current = data
        for part in parts[:-1]:
            if '[' in part:
                # Array access
                key = part.split('[')[0]
                idx = int(part.split('[')[1].replace(']', ''))
                
                if key not in current:
                    current[key] = []
                
                # Extend array if needed
                while len(current[key]) <= idx:
                    current[key].append({})
                
                current = current[key][idx]
            else:
                if part not in current:
                    current[part] = {}
                current = current[part]
        
        # Set final value
        final_key = parts[-1]
        if '[' in final_key:
            key = final_key.split('[')[0]
            idx = int(final_key.split('[')[1].replace(']', ''))
            
            if key not in current:
                current[key] = []
            
            while len(current[key]) <= idx:
                current[key].append(None)
            
            current[key][idx] = value
        else:
            current[final_key] = value
    
    def _generate_tier4_metadata(
        self,
        model_responses: Dict,
        voting_result: Dict,
        debate_result: Dict,
        rag_chunks: List,
        stage_timings: Dict,
        total_cost: float,
        classification: Dict
    ) -> Dict:
        """
        Generate tier_4_metadata_and_future
        
        Fields:
        - model_meta (which models, weights, consensus method, etc.)
        - quality_score (overall + 5 metrics + band)
        - cost_breakdown (per model, per stage)
        - token_usage (input/output per model)
        - processing_time (per stage, bottlenecks)
        
        Returns:
            dict: tier_4 data
        """
        # Model metadata
        models_used = list(model_responses.keys())
        
        model_meta = {
            "models_used": models_used,
            "model_count": len(models_used),
            "weight_strategy": classification.get('weight_strategy', 'BALANCED'),
            # FIX ISSUE 2: Use weights from voting_result, not broken metadata lookup
            "weights_applied": voting_result.get('weights_used', {
                name: 0 for name in models_used
            }),
            "consensus_method": "weighted_voting",
            "debate_rounds": debate_result.get('debate_rounds', 0),
            "converged_fields_count": len(voting_result.get('converged_fields', {})),
            "debated_fields_count": len(voting_result.get('disputed_fields', [])),
            "flagged_for_review": debate_result.get('still_disputed', []),
            "gpt51_added_in_debate": debate_result.get('gpt51_added', False),
            "timestamp": datetime.utcnow().isoformat(),
            "pipeline_version": "1.0.0"
        }
        
        # Quality score
        quality_score = self._calculate_quality_score(
            model_responses=model_responses,
            voting_result=voting_result,
            debate_result=debate_result,
            rag_chunks=rag_chunks
        )
        
        # Cost breakdown
        cost_breakdown = {
            "total_cost": total_cost,
            "currency": "USD",
            "per_model": {
                name: resp.get('cost', 0)
                for name, resp in model_responses.items()
            },
            "classification_cost": 0.00005,  # Gemini Flash
            "image_consensus_cost": 0.006 if any(r.get('has_image') for r in []) else 0,
            "debate_cost": debate_result.get('debate_cost', 0),
            "total_api_calls": len(models_used) + (debate_result.get('debate_rounds', 0) * len(models_used))
        }
        
        # Token usage
        token_usage = {
            "total_input_tokens": sum(
                resp.get('input_tokens', 0)
                for resp in model_responses.values()
            ),
            "total_output_tokens": sum(
                resp.get('output_tokens', 0)
                for resp in model_responses.values()
            ),
            "total_tokens": sum(
                resp.get('tokens_used', 0)
                for resp in model_responses.values()
            ),
            "per_model": {
                name: {
                    "input": resp.get('input_tokens', 0),
                    "output": resp.get('output_tokens', 0),
                    "total": resp.get('tokens_used', 0)
                }
                for name, resp in model_responses.items()
            }
        }
        
        # Processing time
        total_time = sum(stage_timings.values())
        
        processing_time = {
            "total_seconds": total_time,
            "per_stage": stage_timings,
            "bottleneck_stage": max(stage_timings.items(), key=lambda x: x[1])[0] if stage_timings else None,
            "parallel_generation_time": stage_timings.get('stage_5', 0),
            "debate_time": stage_timings.get('stage_7', 0)
        }
        
        # Assemble tier 4
        tier_4 = {
            "model_meta": model_meta,
            "quality_score": quality_score,
            "cost_breakdown": cost_breakdown,
            "token_usage": token_usage,
            "processing_time": processing_time
        }
        
        return tier_4
    
    def _calculate_quality_score(
        self,
        model_responses: Dict,
        voting_result: Dict,
        debate_result: Dict,
        rag_chunks: List
    ) -> Dict:
        """
        Calculate quality score (0.0-1.0) from 5 metrics
        
        Metrics:
        1. avg_model_confidence (25%): Average confidence across models
        2. consensus_rate (30%): % of fields that converged
        3. debate_efficiency (20%): Inverse of debate rounds needed
        4. rag_relevance (15%): Average relevance of top 3 RAG chunks
        5. field_completeness (10%): % of expected fields filled
        
        Returns:
            dict: {overall, metrics, band}
        """
        # 1. Average model confidence
        confidences = []
        for resp in model_responses.values():
            if 'response_json' in resp:
                # Extract confidence from tier_1 if available
                tier1 = resp['response_json'].get('tier_1_core_research', {})
                conf = tier1.get('generation_confidence', 0.75)
                confidences.append(conf)
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.75
        
        # 2. Consensus rate
        total_fields = len(voting_result.get('field_details', {}))
        converged = len(voting_result.get('converged_fields', {}))
        consensus_rate = converged / total_fields if total_fields > 0 else 0.9
        
        # 3. Debate efficiency
        debate_rounds = debate_result.get('debate_rounds', 0)
        if debate_rounds == 0:
            debate_efficiency = 1.0  # No debate needed = perfect
        elif debate_rounds == 1:
            debate_efficiency = 0.9
        else:
            debate_efficiency = 0.7
        
        # 4. RAG relevance
        if rag_chunks:
            top3_relevance = [c.get('relevance_score', 0.7) for c in rag_chunks[:3]]
            rag_relevance = sum(top3_relevance) / len(top3_relevance)
        else:
            rag_relevance = 0.7
        
        # 5. Field completeness
        # Count fields from final_json (the synthesized output) instead of converged_fields
        final_json = voting_result.get('final_json', {})
        total_filled = self._count_filled_fields(final_json)
        total_filled += sum(1 for v in debate_result.get('resolved_fields', {}).values() if v is not None)
        
        # Expect ~150 fields for a complete response
        field_completeness = min(total_filled / 150, 1.0)
        
        # Calculate weighted overall
        overall = (
            avg_confidence * self.quality_weights['avg_model_confidence'] +
            consensus_rate * self.quality_weights['consensus_rate'] +
            debate_efficiency * self.quality_weights['debate_efficiency'] +
            rag_relevance * self.quality_weights['rag_relevance'] +
            field_completeness * self.quality_weights['field_completeness']
        )
        
        # Determine band
        if overall >= 0.90:
            band = "GOLD"
        elif overall >= 0.80:
            band = "SILVER"
        elif overall >= 0.70:
            band = "BRONZE"
        else:
            band = "REVIEW"
        
        return {
            "overall": round(overall, 3),
            "band": band,
            "metrics": {
                "avg_model_confidence": round(avg_confidence, 3),
                "consensus_rate": round(consensus_rate, 3),
                "debate_efficiency": round(debate_efficiency, 3),
                "rag_relevance": round(rag_relevance, 3),
                "field_completeness": round(field_completeness, 3)
            }
        }
    
    def _count_fields(self, data: Any, count: int = 0) -> int:
        """Recursively count all fields in JSON"""
        if isinstance(data, dict):
            for value in data.values():
                count = self._count_fields(value, count + 1)
        elif isinstance(data, list):
            for item in data:
                count = self._count_fields(item, count)
        return count

    def _count_filled_fields(self, data: Any, count: int = 0) -> int:
        """Recursively count all non-null leaf fields in JSON"""
        if isinstance(data, dict):
            for value in data.values():
                if value is not None:
                    if isinstance(value, (dict, list)):
                        count = self._count_filled_fields(value, count)
                    else:
                        count += 1
        elif isinstance(data, list):
            for item in data:
                if item is not None:
                    if isinstance(item, (dict, list)):
                        count = self._count_filled_fields(item, count)
                    else:
                        count += 1
        return count


def main():
    """Test synthesis engine"""
    pass


if __name__ == "__main__":
    main()
