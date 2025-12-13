"""
============================================================================
STAGE 7: DEBATE ORCHESTRATOR (MULTI-ROUND CONSENSUS)
============================================================================
Purpose: Resolve disputed fields through multi-round debate
Process:
    Round 1: Models see disagreement, defend/revise/compromise (70% threshold)
    Round 2: If still disputed, add GPT-5.1 if not used, final debate (70%)
    After Round 2: Flag remaining disputes for human review

Key Features:
    - Batches disputed fields (3-5 per call) to reduce API costs
    - Conditionally adds GPT-5.1 in Round 2 as tiebreaker
    - Tracks debate history and reasoning
    - Adjusts weights based on confidence

Used by: 99_pipeline_runner.py (Stage 7, only if disputed_fields exist)
Author: GATE AE SOTA Pipeline

Cost: ~$0.01-0.03 per question with disputes (30% of questions)
Debate Threshold: 70% (lower than initial 80% voting threshold)
============================================================================
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from utils.logging_utils import setup_logger, log_stage
from utils.prompt_builder import PromptBuilder
from utils.api_wrappers import call_gemini, call_claude, call_deepseek, call_gpt
from utils.cost_tracker import CostTracker

logger = setup_logger("10_debate_orchestrator")


class DebateOrchestrator:
    """
    Orchestrate multi-round debate to resolve disputed fields
    
    Debate Flow:
    1. Group disputed fields into batches (3-5 fields per batch)
    2. Round 1: Each model defends/revises with 70% threshold
    3. If resolved → done
    4. If still disputed → Round 2 with possible GPT-5.1 addition
    5. If still disputed → flag for human review
    
    Batching Strategy:
    - 1-3 fields: 1 batch (all together)
    - 4-8 fields: 2 batches (4 each)
    - 9-15 fields: 3 batches (5 each)
    - 16+ fields: 4 batches max, flag rest for human review
    """
    
    def __init__(self, configs: Dict, clients: Dict):
        """
        Args:
            configs: All configuration dictionaries
            clients: All initialized API clients
        """
        self.configs = configs
        self.clients = clients
        self.prompt_builder = PromptBuilder(configs['prompts_config'])
        self.cost_tracker = CostTracker(configs['models_config'])
        
        # Debate threshold (lower than voting threshold)
        self.debate_threshold = configs['thresholds_config']['consensus']['debate_round2_threshold']
        self.round2_threshold = configs['thresholds_config']['consensus']['debate_round2_threshold']
        
        # Batch sizes
        self.max_fields_per_batch = 5
        self.max_batches = 4
    
    @log_stage("Stage 7: Debate Orchestrator")
    def resolve_disputes(
        self,
        disputed_fields: List[str],
        field_details: Dict[str, Dict],
        model_responses: Dict[str, Dict],
        weights: Dict[str, float],
        rag_chunks: List[Dict],
        question: Dict[str, Any],
        classification: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Resolve disputed fields through debate
        
        Args:
            disputed_fields: List of disputed field paths from Stage 6
            field_details: Voting details for each field
            model_responses: Original model responses from Stage 5
            weights: Model weights
            rag_chunks: RAG context (for reference in debate)
            question: Question object
            classification: Classification from Stage 2
        
        Returns:
            dict: {
                "resolved_fields": {field_path: value, ...},
                "still_disputed": [field_path, ...],
                "debate_rounds": 1 or 2,
                "debate_history": {...},
                "gpt51_added": bool,
                "debate_cost": float
            }
        """
        question_id = question.get('question_id', 'unknown')
        logger.info(f"Resolving {len(disputed_fields)} disputed fields for: {question_id}")
        
        if not disputed_fields:
            logger.info("  No disputed fields, skipping debate")
            return {
                "resolved_fields": {},
                "still_disputed": [],
                "debate_rounds": 0,
                "debate_history": {},
                "gpt51_added": False,
                "debate_cost": 0.0
            }
        
        # Create batches
        batches = self._create_batches(disputed_fields)
        logger.info(f"  Created {len(batches)} batches (max {self.max_fields_per_batch} fields each)")
        
        # Track overall results
        all_resolved = {}
        all_still_disputed = []
        total_cost = 0.0
        debate_history = {"round1": {}, "round2": {}}
        
        # Get models used in Stage 5
        models_used = list(model_responses.keys())
        use_gpt51_stage5 = classification.get('use_gpt51', False)
        gpt51_added_round2 = False
        
        # ROUND 1
        logger.info("  === DEBATE ROUND 1 ===")
        
        for batch_num, batch_fields in enumerate(batches, 1):
            logger.info(f"    Batch {batch_num}/{len(batches)}: {len(batch_fields)} fields")
            
            # Run Round 1 debate for this batch
            round1_result = self._run_debate_round(
                round_num=1,
                batch_fields=batch_fields,
                field_details=field_details,
                model_responses=model_responses,
                weights=weights,
                rag_chunks=rag_chunks,
                models_to_use=models_used
            )
            
            # Update results
            all_resolved.update(round1_result['resolved'])
            all_still_disputed.extend(round1_result['still_disputed'])
            total_cost += round1_result['cost']
            debate_history['round1'][f'batch_{batch_num}'] = round1_result['history']
        
        logger.info(f"  Round 1 complete: {len(all_resolved)} resolved, {len(all_still_disputed)} still disputed")
        
        # Check if Round 2 needed
        if not all_still_disputed:
            logger.info("  ✓ All disputes resolved in Round 1")
            return {
                "resolved_fields": all_resolved,
                "still_disputed": [],
                "debate_rounds": 1,
                "debate_history": debate_history,
                "gpt51_added": False,
                "debate_cost": total_cost
            }
        
        # ROUND 2
        logger.info(f"  === DEBATE ROUND 2 === ({len(all_still_disputed)} fields remaining)")
        
        # Decide if GPT-5.1 should be added
        models_round2 = models_used.copy()
        
        if not use_gpt51_stage5:
            # GPT-5.1 wasn't used in Stage 5, add it now as tiebreaker
            models_round2.append("gpt_5_1")
            gpt51_added_round2 = True
            logger.info("  Adding GPT-5.1 as tiebreaker for Round 2")
            
            # Adjust weights (redistribute to include GPT-5.1)
            weights_round2 = self._adjust_weights_with_gpt51(weights)
        else:
            weights_round2 = weights
        
        # Create batches for Round 2
        round2_batches = self._create_batches(all_still_disputed)
        
        round2_resolved = {}
        round2_still_disputed = []
        
        for batch_num, batch_fields in enumerate(round2_batches, 1):
            logger.info(f"    Batch {batch_num}/{len(round2_batches)}: {len(batch_fields)} fields")
            
            # Run Round 2 debate
            round2_result = self._run_debate_round(
                round_num=2,
                batch_fields=batch_fields,
                field_details=field_details,
                model_responses=model_responses,
                weights=weights_round2,
                rag_chunks=rag_chunks,
                models_to_use=models_round2,
                round1_history=debate_history['round1']
            )
            
            # Update results
            round2_resolved.update(round2_result['resolved'])
            round2_still_disputed.extend(round2_result['still_disputed'])
            total_cost += round2_result['cost']
            debate_history['round2'][f'batch_{batch_num}'] = round2_result['history']
        
        # Merge Round 1 and Round 2 results
        all_resolved.update(round2_resolved)
        final_disputed = round2_still_disputed
        
        logger.info(f"  Round 2 complete: {len(round2_resolved)} more resolved")
        logger.info(f"  ✓ Debate complete: {len(all_resolved)} resolved, {len(final_disputed)} flagged for review")
        
        return {
            "resolved_fields": all_resolved,
            "still_disputed": final_disputed,
            "debate_rounds": 2,
            "debate_history": debate_history,
            "gpt51_added": gpt51_added_round2,
            "debate_cost": total_cost
        }
    
    def _create_batches(self, fields: List[str]) -> List[List[str]]:
        """
        Create batches of fields for debate
        
        Strategy:
        - 1-3 fields: 1 batch
        - 4-8 fields: 2 batches (4 each)
        - 9-15 fields: 3 batches (5 each)
        - 16+ fields: 4 batches max, rest flagged
        
        Args:
            fields: List of field paths
        
        Returns:
            list: List of batches (each batch is list of fields)
        """
        total_fields = len(fields)
        
        if total_fields <= 3:
            return [fields]
        
        elif total_fields <= 8:
            # 2 batches
            mid = total_fields // 2
            return [fields[:mid], fields[mid:]]
        
        elif total_fields <= 15:
            # 3 batches
            batch_size = (total_fields + 2) // 3
            return [
                fields[i:i+batch_size]
                for i in range(0, total_fields, batch_size)
            ]
        
        else:
            # 4 batches max (limit to 20 fields total)
            limited_fields = fields[:20]
            if len(fields) > 20:
                logger.warning(f"Too many disputed fields ({len(fields)}), limiting to 20")
            
            batch_size = 5
            return [
                limited_fields[i:i+batch_size]
                for i in range(0, len(limited_fields), batch_size)
            ]
    
    def _run_debate_round(
        self,
        round_num: int,
        batch_fields: List[str],
        field_details: Dict,
        model_responses: Dict,
        weights: Dict,
        rag_chunks: List,
        models_to_use: List[str],
        round1_history: Dict = None
    ) -> Dict:
        """
        Run one round of debate for a batch of fields
        
        Args:
            round_num: 1 or 2
            batch_fields: Fields in this batch
            field_details: Voting details from Stage 6
            model_responses: Original responses
            weights: Model weights
            rag_chunks: RAG context
            models_to_use: Which models to call
            round1_history: Round 1 debate history (for Round 2 only)
        
        Returns:
            dict: {
                "resolved": {field: value},
                "still_disputed": [fields],
                "cost": float,
                "history": {...}
            }
        """
        # Build debate prompt
        prompt_name = f"debate_round{round_num}"
        
        # Prepare disputed fields data
        disputed_fields_data = self._format_disputed_fields(
            batch_fields, field_details, model_responses, weights
        )
        
        # Format RAG context
        rag_context = self._format_rag_context(rag_chunks)
        
        # Build prompt variables
        variables = {
            "batch_number": 1,  # Simplified for this implementation
            "total_batches": 1,
            "field_count": len(batch_fields),
            "disputed_fields_data": disputed_fields_data,
            "rag_context": rag_context,
            "consensus_score": 0,  # Will be filled
            "gpt51_added": "gpt_5_1" in models_to_use and round_num == 2
        }
        
        if round_num == 2:
            variables["round1_debate_history"] = self._format_round1_history(round1_history)
            variables["adjusted_weights_info"] = self._format_weights(weights)
            variables["gpt51_context"] = self._get_gpt51_context(variables["gpt51_added"])
        
        prompt = self.prompt_builder.build_prompt(prompt_name, variables)
        
        # Call all models with debate prompt
        debate_responses = {}
        total_cost = 0.0
        
        for model_name in models_to_use:
            try:
                response, cost = self._call_model_for_debate(model_name, prompt)
                debate_responses[model_name] = response
                total_cost += cost
            except Exception as e:
                logger.error(f"Debate call failed for {model_name}: {e}")
        
        # Vote on revised answers
        threshold = self.round2_threshold if round_num == 2 else self.debate_threshold
        resolved, still_disputed = self._vote_on_debate_responses(
            batch_fields, debate_responses, weights, threshold
        )
        
        return {
            "resolved": resolved,
            "still_disputed": still_disputed,
            "cost": total_cost,
            "history": {
                "fields": batch_fields,
                "responses": debate_responses,
                "resolved_count": len(resolved)
            }
        }
    
    def _format_disputed_fields(
        self,
        fields: List[str],
        field_details: Dict,
        model_responses: Dict,
        weights: Dict
    ) -> str:
        """
        Format disputed fields for debate prompt
        
        Format:
            ## FIELD 1: difficulty_analysis.score
            
            Your Answer: 6
            Other Models:
            - Claude: 8 (weight: 0.30, confidence: 0.85)
            - DeepSeek: 7 (weight: 0.20, confidence: 0.75)
            - GPT-5.1: 8 (weight: 0.30, confidence: 0.90)
            
            Consensus: 50%
        """
        formatted = ""
        
        for i, field_path in enumerate(fields, 1):
            details = field_details.get(field_path, {})
            model_values = details.get('model_values', {})
            consensus_weight = details.get('consensus_weight', 0.0)
            
            formatted += f"\n## FIELD {i}: {field_path}\n\n"
            
            # This will be replaced per model
            formatted += "Your Answer: {{YOUR_ANSWER}}\n\n"
            formatted += "Other Models:\n"
            
            for model_name, value in model_values.items():
                weight = weights.get(model_name, 0)
                formatted += f"- {model_name}: {value} (weight: {weight:.2f})\n"
            
            formatted += f"\nConsensus: {consensus_weight:.0%}\n"
            formatted += "\n---\n"
        
        return formatted
    
    def _format_rag_context(self, rag_chunks: List[Dict]) -> str:
        """Format RAG chunks (abbreviated for debate)"""
        if not rag_chunks:
            return "No RAG context available."
        
        formatted = "# RAG CONTEXT (for reference):\n\n"
        
        for chunk in rag_chunks[:3]:  # Only top 3 for debate
            formatted += f"**{chunk['source_name']}:** {chunk['text'][:200]}...\n\n"
        
        return formatted
    
    def _format_round1_history(self, round1_history: Dict) -> str:
        """Format Round 1 debate history for Round 2 prompt"""
        if not round1_history:
            return "No Round 1 history available."
        
        formatted = "# ROUND 1 DEBATE HISTORY:\n\n"
        
        for batch_key, batch_data in round1_history.items():
            formatted += f"## {batch_key}:\n"
            formatted += f"Fields debated: {batch_data.get('fields', [])}\n"
            formatted += f"Resolved: {batch_data.get('resolved_count', 0)}\n\n"
        
        return formatted
    
    def _format_weights(self, weights: Dict) -> str:
        """Format model weights"""
        return "\n".join([f"- {model}: {weight:.2f}" for model, weight in weights.items()])
    
    def _get_gpt51_context(self, gpt51_added: bool) -> str:
        """Get GPT-5.1 context message"""
        if gpt51_added:
            return "GPT-5.1 has been added to this debate as a tiebreaker."
        return "All models from Stage 5 continue in this round."
    
    def _call_model_for_debate(self, model_name: str, prompt: str) -> Tuple[Dict, float]:
        """
        Call a model for debate
        
        Returns:
            tuple: (parsed_response, cost)
        """
        model_config = self.configs['models_config']['models'].get(model_name)
        
        # Call appropriate API
        if model_name == "gemini_2.5_pro":
            response_text, usage = call_gemini(
                prompt=prompt,
                client=self.clients['google_genai'],
                model_name=model_config['model_id'],
                temperature=0.3,
                max_tokens=2000
            )
        elif model_name == "claude_sonnet_4.5":
            response_text, usage = call_claude(
                prompt=prompt,
                client=self.clients['anthropic'],
                model_name=model_config['model_id'],
                temperature=0.3,
                max_tokens=2000
            )
        elif model_name == "deepseek_r1":
            response_text, usage = call_deepseek(
                prompt=prompt,
                client=self.clients['deepseek'],
                model_name=model_config['model_id'],
                temperature=0.3,
                max_tokens=2000
            )
        elif model_name == "gpt_5_1":
            response_text, usage = call_gpt(
                prompt=prompt,
                client=self.clients['openai'],
                model_name=model_config['model_id'],
                temperature=0.3,
                max_tokens=2000
            )
        
        # Parse JSON
        response_json = self._parse_debate_response(response_text)
        
        # Calculate cost
        cost = self.cost_tracker.calculate_cost(
            model_name=model_name,
            input_tokens=usage['input_tokens'],
            output_tokens=usage['output_tokens']
        )
        
        return response_json, cost
    
    def _parse_debate_response(self, response_text: str) -> Dict:
        """Parse debate response JSON using json_repair"""
        try:
            from json_repair import repair_json
            return repair_json(response_text, return_objects=True)
        except Exception as e:
            logger.error(f"Debate JSON parse error: {e}")
            # Fallback to simple extraction if repair fails (unlikely)
            import json
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "").replace("```", "").strip()
            try:
                start = response_text.index('{')
                end = response_text.rindex('}') + 1
                return json.loads(response_text[start:end])
            except:
                raise ValueError(f"Failed to parse debate JSON: {e}")
    
    def _vote_on_debate_responses(
        self,
        fields: List[str],
        debate_responses: Dict[str, Dict],
        weights: Dict,
        threshold: float
    ) -> Tuple[Dict, List]:
        """
        Vote on debate responses
        
        Returns:
            tuple: (resolved_fields, still_disputed_fields)
        """
        resolved = {}
        still_disputed = []
        
        for field_path in fields:
            # Extract revised answers
            revised_values = {}
            
            for model_name, response in debate_responses.items():
                # Find this field's answer in response
                field_key = field_path.replace('.', '_').replace('[', '_').replace(']', '')
                
                if field_key in response:
                    revised_values[model_name] = response[field_key].get('revised_answer')
            
            # Simple voting (this can be enhanced)
            if not revised_values:
                still_disputed.append(field_path)
                continue
            
            # Check for majority
            value_counts = {}
            for model, value in revised_values.items():
                value_str = str(value)
                if value_str not in value_counts:
                    value_counts[value_str] = {'weight': 0, 'value': value}
                value_counts[value_str]['weight'] += weights.get(model, 0)
            
            # Find top value
            top_value = max(value_counts.items(), key=lambda x: x[1]['weight'])
            
            if top_value[1]['weight'] >= threshold:
                resolved[field_path] = top_value[1]['value']
            else:
                still_disputed.append(field_path)
        
        return resolved, still_disputed
    
    def _adjust_weights_with_gpt51(self, original_weights: Dict) -> Dict:
        """
        Adjust weights to include GPT-5.1
        
        Strategy: Redistribute proportionally
        """
        new_weights = original_weights.copy()
        
        # Give GPT-5.1 weight of 0.30, scale others down
        new_weights['gpt_5_1'] = 0.30
        
        # Scale others to sum to 0.70
        total_original = sum(original_weights.values())
        scale_factor = 0.70 / total_original
        
        for model in original_weights:
            new_weights[model] = original_weights[model] * scale_factor
        
        return new_weights


def main():
    """Test debate orchestrator"""
    pass


if __name__ == "__main__":
    main()