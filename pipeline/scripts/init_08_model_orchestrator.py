"""
============================================================================
STAGE 5: MODEL ORCHESTRATOR (PARALLEL GENERATION)
============================================================================
Purpose: Call all models in parallel to generate tier_1-4 responses
Models: Gemini 2.5 Pro, Claude Sonnet 4.5, DeepSeek R1, GPT-5.1 (conditional)
Process: 
    - Build prompts with RAG context, question, classification
    - Call 3 or 4 models in parallel (based on use_gpt51 flag)
    - Apply model-specific weights from weight_strategy
    - Parse and validate JSON responses
    - Return all model responses for voting/debate
Used by: 99_pipeline_runner.py (Stage 5)
Author: GATE AE SOTA Pipeline

Cost per question (avg):
    - 3 models: ~$0.15 (if use_gpt51=false)
    - 4 models: ~$0.20 (if use_gpt51=true)
============================================================================
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import concurrent.futures
import sys

# Add project root to path
current_file = Path(__file__).resolve()
PROJECT_ROOT = current_file.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Robust imports
try:
    from pipeline.utils.logging_utils import setup_logger, log_stage
    from pipeline.utils.prompt_builder import PromptBuilder
    from pipeline.utils.api_wrappers import call_gemini, call_claude, call_deepseek, call_gpt
    from pipeline.utils.json_validator import validate_complete_schema
    from pipeline.utils.cost_tracker import CostTracker
except ImportError:
    sys.path.append(str(current_file.parent.parent))
    from utils.logging_utils import setup_logger, log_stage
    from utils.prompt_builder import PromptBuilder
    from utils.api_wrappers import call_gemini, call_claude, call_deepseek, call_gpt
    from utils.json_validator import validate_complete_schema
    from utils.cost_tracker import CostTracker

# Try to import json_repair
try:
    from json_repair import repair_json
except ImportError:
    print("Warning: json_repair not installed. Install with `pip install json_repair`")
    repair_json = None

logger = setup_logger("08_model_orchestrator")


class ModelOrchestrator:
    """
    Orchestrate parallel generation from multiple models
    
    Key responsibilities:
    1. Build complete prompts (system + RAG + question)
    2. Determine which models to call (3 or 4 based on use_gpt51)
    3. Call models in parallel for speed
    4. Parse and validate responses
    5. Apply weight strategy
    6. Track costs and tokens
    
    CRITICAL UPDATE (Linked Answer Questions):
    - Injects Common Data Statement for all parts.
    - Injects explicitly where the question stands in the sequence (Part X of Y).
    - Injects Derived Answer and Reasoning from previous parts.
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
        
        # Get model configurations
        self.models_config = configs['models_config']['models']
        self.weights_config = configs['weights_config']['weight_strategies']
    
    @log_stage("Stage 5: Model Orchestrator")
    def generate_responses(
        self,
        question: Dict[str, Any],
        classification: Dict[str, Any],
        rag_chunks: List[Dict[str, Any]],
        image_consensus: str = None,
        previous_context: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate responses from all models
        
        Args:
            question: Question object from Stage 1
            classification: tier_0_classification from Stage 2
            rag_chunks: Top 6 chunks from Stage 4
            image_consensus: Consensus image description from Stage 4.5 (if applicable)
        
        Returns:
            dict: {
                "model_responses": {model_name: response_json, ...},
                "weights": {model_name: weight, ...},
                "metadata": {tokens, costs, timing}
            }
        """
        question_id = question.get('question_id', 'unknown')
        logger.info(f"Generating model responses for: {question_id}")
        
        # Determine which models to call
        use_gpt51 = classification.get('use_gpt51', False)
        models_to_call = self._get_models_to_call(use_gpt51)
        
        logger.info(f"  Models to call: {models_to_call}")
        logger.info(f"  Weight strategy: {classification.get('weight_strategy', 'BALANCED')}")
        
        # Get weights for this question type
        weights = self._get_model_weights(classification.get('weight_strategy', 'BALANCED'), models_to_call)
        logger.info(f"  Model weights: {weights}")
        
        # Build prompt
        prompt = self._build_prompt(question, classification, rag_chunks, image_consensus, previous_context)
        
        # Call models in parallel
        logger.info("  Calling models in parallel...")
        model_responses = self._call_models_parallel(
            models=models_to_call,
            prompt=prompt,
            question=question,
            previous_context=previous_context
        )
        
        # DEBUG: Save raw model responses before any injection
        self._save_debug_output(question_id, "01_raw_model_responses", model_responses)
        
        # --- FIX 1: Inject Stage 2 classification data into tier_0_classification ---
        # The models follow the prompt which doesn't ask for difficulty_score, 
        # complexity_flags, or use_gpt51, but the validator requires them.
        for model_name, data in model_responses.items():
            if data and data.get('response_json'):
                tier_0 = data['response_json'].get('tier_0_classification', {})
                
                # Inject missing fields from Stage 2 Classification
                tier_0['difficulty_score'] = classification.get('difficulty_score', 5)
                tier_0['complexity_flags'] = classification.get('complexity_flags', {})
                tier_0['use_gpt51'] = classification.get('use_gpt51', False)
                
                # Extended injection for validator compliance
                tier_0['classification_reasoning'] = classification.get('classification_reasoning', 'Reasoning not provided')
                tier_0['classification_confidence'] = classification.get('classification_confidence', 0.8)
                tier_0['weight_strategy'] = classification.get('weight_strategy', 'BALANCED')
                tier_0['content_type'] = classification.get('content_type', 'conceptual_application')
                tier_0['media_type'] = classification.get('media_type', 'text_only')
                
                # Update the response
                data['response_json']['tier_0_classification'] = tier_0
        
        logger.info("  [OK] Injected Stage 2 classification fields into tier_0_classification")
        
        # DEBUG: Save after injection
        self._save_debug_output(question_id, "02_after_injection", model_responses)
        
        # --- FIX 2: Inject/Fix Tier 4 metadata from question object ---
        # The validator requires specific tier_4 fields that we can populate from the question
        import datetime
        for model_name, data in model_responses.items():
            if data and data.get('response_json'):
                tier_4 = data['response_json'].get('tier_4_metadata_and_future', {})
                
                # Ensure question_metadata exists and has required fields from question object
                if 'question_metadata' not in tier_4 or tier_4['question_metadata'] is None:
                    tier_4['question_metadata'] = {}
                qm = tier_4['question_metadata']
                qm['id'] = qm.get('id') or question.get('question_id', question_id)
                qm['year'] = qm.get('year') or question.get('year', 2024)
                qm['marks'] = qm.get('marks') or question.get('marks', 1.0)
                
                # Ensure syllabus_mapping exists
                if 'syllabus_mapping' not in tier_4 or tier_4['syllabus_mapping'] is None:
                    tier_4['syllabus_mapping'] = {}
                if 'gate_section' not in tier_4['syllabus_mapping']:
                    tier_4['syllabus_mapping']['gate_section'] = classification.get('subject', 'General')
                
                # Ensure rag_quality exists
                if 'rag_quality' not in tier_4 or tier_4['rag_quality'] is None:
                    tier_4['rag_quality'] = {}
                if 'relevance_score' not in tier_4['rag_quality']:
                    tier_4['rag_quality']['relevance_score'] = 0.5
                if 'notes' not in tier_4['rag_quality']:
                    tier_4['rag_quality']['notes'] = "Auto-generated"
                if 'sources_distribution' not in tier_4['rag_quality']:
                    tier_4['rag_quality']['sources_distribution'] = {}
                
                # Ensure model_meta exists
                if 'model_meta' not in tier_4 or tier_4['model_meta'] is None:
                    tier_4['model_meta'] = {}
                if 'timestamp' not in tier_4['model_meta']:
                    tier_4['model_meta']['timestamp'] = datetime.datetime.now().isoformat()
                if 'version' not in tier_4['model_meta']:
                    tier_4['model_meta']['version'] = "GATE_AE_SOTA_v1.0"
                
                # Ensure future_questions_potential list exists
                if 'future_questions_potential' not in tier_4 or tier_4['future_questions_potential'] is None:
                    tier_4['future_questions_potential'] = []
                
                # Update the response
                data['response_json']['tier_4_metadata_and_future'] = tier_4
        
        logger.info("  [OK] Injected/Fixed Tier 4 metadata fields")
        # -------------------------------------------------------------------------------
        
        # Validate responses
        model_responses = self._validate_responses(model_responses, question_id)
        
        # Calculate metadata
        metadata = self._calculate_metadata(model_responses, weights)
        
        logger.info(f"  ✓ Generated {len(model_responses)} responses")
        logger.info(f"    - Total tokens: {metadata['total_tokens']}")
        logger.info(f"    - Total cost: ${metadata['total_cost']:.4f}")
        
        return {
            "model_responses": model_responses,
            "weights": weights,
            "metadata": metadata
        }
    
    def _get_models_to_call(self, use_gpt51: bool) -> List[str]:
        """
        Determine which models to call based on use_gpt51 flag
        
        Base 3 models (always used):
            - gemini_2.5_pro
            - claude_sonnet_4.5
            - deepseek_r1
        
        Conditional 4th model:
            - gpt_5.1 (if use_gpt51=true)
        
        Args:
            use_gpt51: Boolean from classification
        
        Returns:
            list: Model names to call
        """
        base_models = [
            "gemini_2.5_pro",
            "claude_sonnet_4.5",
            "deepseek_r1"
        ]
        
        if use_gpt51:
            return base_models + ["gpt_5_1"]  # Must match config key (underscore, not dot)
        else:
            return base_models
    
    def _get_model_weights(self, weight_strategy: str, models: List[str]) -> Dict[str, float]:
        """
        Get model weights from weight strategy
        
        Args:
            weight_strategy: e.g., "NUMERICAL_WEIGHTED", "CONCEPTUAL_WEIGHTED"
            models: List of model names being used
        
        Returns:
            dict: {model_name: weight}
        """
        strategy_weights = self.weights_config.get(weight_strategy)
        
        if not strategy_weights:
            logger.warning(f"Unknown weight strategy: {weight_strategy}, using BALANCED")
            strategy_weights = self.weights_config.get("BALANCED", {})
        
        # Extract weights for models being used
        weights = {}
        for model in models:
            weights[model] = strategy_weights.get(model, 0.25)
        
        # Normalize weights to sum to 1.0
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        
        return weights
    
    def _build_prompt(
        self,
        question: Dict,
        classification: Dict,
        rag_chunks: List[Dict],
        image_consensus: str = None,
        previous_context: List[Dict] = None
    ) -> str:
        """
        Build complete prompt for models
        
        Components:
            1. System prompt base (4,000 tokens)
            2. RAG context formatted (4,800 tokens from 6 chunks)
            3. Question details (500 tokens)
            4. Image description if applicable (500 tokens)
            5. Classification metadata (50 tokens)
        
        Total: ~9,800-10,000 tokens
        
        Args:
            question: Question object
            classification: Classification results
            rag_chunks: Retrieved chunks
            image_consensus: Image description (optional)
        
        Returns:
            str: Complete prompt
        """
        # Format RAG context
        rag_context = self._format_rag_context(rag_chunks)
        
        # Format options
        if question.get('options'):
            if isinstance(question['options'], dict):
                options_text = "\n".join([
                    f"{key}. {value}" 
                    for key, value in question['options'].items()
                ])
            else:
                options_text = str(question['options'])
        else:
            options_text = "N/A (Numerical Answer Type)"
        
        # Format image description
        image_desc = ""
        if image_consensus:
            image_desc = f"\n\n{image_consensus}\n"
        # ------------------------------------------------------------------
        # LINKED ANSWER QUESTION LOGIC (Fixes Hallucination & Consistency)
        # ------------------------------------------------------------------
        
        # 1. COMMON DATA INJECTION (Unconditional)
        common_data_section = ""
        if question.get('common_data_statement'):
            common_data_section = (
                f"### COMMON DATA (APPLIES TO ALL LINKED QUESTIONS):\n"
                f"{question.get('common_data_statement')}\n"
                f"------------------------------------------------------------\n\n"
            )
        
        # 2. LINKED METADATA INJECTION
        linked_meta_section = ""
        link_info = question.get('linked_question_group', {})
        if link_info:
            if isinstance(link_info, dict):
                group_id = link_info.get('group_id', 'unknown')
                current_idx = link_info.get('question_index', '?')
                total = link_info.get('total_questions', '?')
            else:
                # Handle case where link_info is a string (like in 2007 data)
                group_id = str(link_info)
                current_idx = 'unknown'
                total = 'unknown'
            
            linked_meta_section = (
                f"[CONTEXTUAL INSTRUCTION]\n"
                f"This is a LINKED ANSWER QUESTION. It is part of a group: {group_id}.\n"
                f"You are solving Part {current_idx} of {total}.\n"
                f"Consistency with previous parts is mandatory.\n\n"
            )

        # 3. PREVIOUS CONTEXT INJECTION (The Bridge)
        linked_context_text = ""
        if previous_context:
            linked_context_text = "### CONTEXT FROM PREVIOUS LINKED QUESTIONS (Chronological Order)\n"
            for i, prev_entry in enumerate(previous_context, 1):
                # prev_entry is {raw_question: {...}, final_json: {...}}
                raw_q = prev_entry.get('raw_question', {})
                fj = prev_entry.get('final_json', {})
                
                p_id = raw_q.get('question_id', f"Linked-Q{i}")
                p_text = raw_q.get('question_text', 'N/A')
                
                # Format options if present
                p_opts = raw_q.get('options', {})
                p_opts_text = ""
                if p_opts and isinstance(p_opts, dict):
                    p_opts_text = ", ".join([f"{k}: {v}" for k, v in p_opts.items()])
                
                # Get answer and physics from final_json Tier 1
                t1 = fj.get('tier_1_core_research', {})
                p_ans = t1.get('answer_validation', {}).get('derived_answer', 'Not Found')
                p_phys = t1.get('key_physics_concepts', 'Not Found')
                if isinstance(p_phys, list):
                    p_phys = ", ".join([c.get('name', str(c)) if isinstance(c, dict) else str(c) for c in p_phys])
                
                # Check if previous question had an image
                p_has_image = raw_q.get('has_question_image', False)
                p_image_note = "Yes (Image was provided to vision models)" if p_has_image else "No"
                
                linked_context_text += (
                    f"### PREVIOUS QUESTION CONTEXT ({p_id}):\n"
                    f"- Question Text: {p_text}\n"
                    f"- Options: {p_opts_text or 'N/A'}\n"
                    f"- Had Image: {p_image_note}\n"
                    f"- Derived Answer: {p_ans}\n"
                    f"- Key Physics/Reasoning: {p_phys}\n\n"
                )
            
            linked_context_text += (
                "[INSTRUCTION]: Use the derived values and physical context from the previous question "
                "to solve this one. Do not re-derive established facts unless necessary.\n"
                "------------------------------------------------------------\n\n"
            )

        # Combine for injection
        # We prepend this to question_text so it appears prominently in the "Question" section of the prompt
        full_context_injection = common_data_section + linked_meta_section + linked_context_text
        
        # Build variable dictionary
        variables = {
            "rag_context": rag_context,
            "question_id": question.get('question_id', 'unknown'),
            "year": question.get('year', 'unknown'),
            "marks": question.get('marks', 1),
            "negative_marks": question.get('negative_marks', 0),
            "question_type": question.get('question_type', 'MCQ'),
            "question_text": full_context_injection + question['question_text'],  # Inject Here
            "options": options_text,
            "answer_key": question.get('answer_key', 'Not provided'),
            "image_description": image_desc,
            "content_type": classification.get('content_type', 'unknown'),
            "media_type": classification.get('media_type', 'unknown'),
            "weight_strategy": classification.get('weight_strategy', 'unknown')
        }
        
        # Build prompt from template
        prompt = self.prompt_builder.build_prompt(
            prompt_name="base_system",
            variables=variables
        )
        
        return prompt
    
    def _format_rag_context(self, rag_chunks: List[Dict]) -> str:
        """
        Format RAG chunks for prompt
        
        Format each chunk as:
            CHUNK {rank} (Relevance: {score:.2f}):
            Source: {source_type} - {source_name}
            Reference: {reference}
            
            {text}
            
            ---
        
        Args:
            rag_chunks: List of retrieved chunks (6 chunks)
        
        Returns:
            str: Formatted RAG context
        """
        if not rag_chunks:
            return "No relevant context retrieved."
        
        formatted = "# RETRIEVED KNOWLEDGE:\n\n"
        
        for chunk in rag_chunks:
            # Handle potential missing keys gracefully
            rank = chunk.get('rank', '?')
            score = chunk.get('relevance_score', 0.0)
            stype = chunk.get('source_type', 'unknown')
            sname = chunk.get('source_name', 'unknown')
            ref = chunk.get('reference', 'unknown')
            text = chunk.get('text', '')
            
            # Extract metadata for preservation
            metadata = chunk.get('metadata', {})
            video_url = metadata.get('video_url', '')
            book = metadata.get('book', '')
            professor = metadata.get('professor', '')
            timestamp = ""
            if metadata.get('timestamp_start'):
                timestamp = f" [{metadata.get('timestamp_start')} - {metadata.get('timestamp_end', '')}]"

            formatted += f"## CHUNK {rank} (Relevance: {score:.2f})\n"
            formatted += f"**Source:** {stype} - {sname}\n"
            formatted += f"**Reference:** {ref}\n"
            
            # Include video URL if available (CRITICAL for Issue 4)
            if video_url:
                formatted += f"**Video URL:** {video_url}{timestamp}\n"
                if professor:
                    formatted += f"**Professor:** {professor}\n"
            elif book:
                formatted += f"**Book:** {book}\n"
            
            formatted += f"\n{text}\n\n"
            formatted += "---\n\n"
        
        formatted += "# End of retrieved knowledge.\n"
        
        return formatted
    
    def _call_models_parallel(
        self,
        models: List[str],
        prompt: str,
        question: Dict,
        previous_context: List[Dict] = None
    ) -> Dict[str, Dict]:
        """
        Call all models in parallel using ThreadPoolExecutor
        
        Why parallel?
        - Reduces total time from ~15s (sequential) to ~5s (parallel)
        - API calls are I/O bound, not CPU bound
        - Thread-safe since each call is independent
        
        Args:
            models: List of model names to call
            prompt: Complete prompt
            question: Question object
        
        Returns:
            dict: {model_name: parsed_response, ...}
        """
        responses = {}
        
        # Use ThreadPoolExecutor for parallel API calls
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all calls
            future_to_model = {
                executor.submit(
                    self._call_single_model,
                    model_name,
                    prompt,
                    question,
                    previous_context
                ): model_name
                for model_name in models
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    response = future.result()
                    responses[model_name] = response
                    logger.info(f"    ✓ {model_name}: {response['tokens_used']} tokens")
                except Exception as e:
                    # Better error logging to identify root cause
                    import traceback
                    error_details = traceback.format_exc()
                    logger.error(f"    ✗ {model_name} failed: {str(e)}")
                    logger.debug(f"    Full traceback for {model_name}:\n{error_details}")
                    
                    # Check if it's a config/key error
                    if "'" in str(e) and "model" in str(e).lower():
                        logger.error(f"    Model config issue - check models_config.yaml for '{model_name}'")
                        logger.error(f"    Available models: {list(self.models_config.keys())}")
                    
                    responses[model_name] = None
        
        return responses
    
    def _call_single_model(
        self,
        model_name: str,
        prompt: str,
        question: Dict,
        previous_context: List[Dict] = None
    ) -> Dict:
        """
        Call a single model and parse response
        
        Args:
            model_name: Name of model to call
            prompt: Complete prompt
            question: Question object (for image if needed)
        
        Returns:
            dict: {
                "model_name": str,
                "response_json": dict,
                "tokens_used": int,
                "cost": float,
                "time_seconds": float
            }
        """
        import time
        start_time = time.time()
        
        model_config = self.models_config.get(model_name)
        if not model_config:
            available_models = list(self.models_config.keys())
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available models: {available_models}. "
                f"Check models_config.yaml for correct model names."
            )
        
        # Validate model_config has required fields
        if 'model_id' not in model_config:
            raise ValueError(
                f"Model config for '{model_name}' missing 'model_id' field. "
                f"Config keys: {list(model_config.keys())}"
            )
        
        # Get images list (previous context + current)
        images_list = []
        if model_name != "deepseek_r1":
            # Add previous linked images first (from raw_question in combined cache object)
            if previous_context:
                for prev_entry in previous_context:
                    raw_q = prev_entry.get('raw_question', {})
                    if raw_q.get('has_question_image') and raw_q.get('image_metadata'):
                        img_b64 = raw_q['image_metadata'].get('base64')
                        if img_b64:
                            images_list.append(img_b64)
            
            # Add current image
            if question.get('has_question_image') and question.get('image_metadata'):
                images_list.append(question['image_metadata'].get('base64'))
        
        # Call appropriate API wrapper
        if model_name == "gemini_2.5_pro":
            response_text, usage = call_gemini(
                prompt=prompt,
                images=images_list,  # Use list
                client=self.clients['google_genai'],
                model_name=model_config['model_id'],
                temperature=0.3,
                max_tokens=16000
            )
        
        elif model_name == "claude_sonnet_4.5":
            # Check if using Bedrock
            use_bedrock = self.configs.get('use_bedrock_for_claude', False)
            client = self.clients.get('bedrock_runtime') if use_bedrock else self.clients.get('anthropic')
            
            # Bedrock specific ID handling
            if use_bedrock:
                model_id = os.getenv("BEDROCK_CLAUDE_MODEL_ID", model_config['model_id'])
            else:
                model_id = model_config['model_id']
                
            response_text, usage = call_claude(
                prompt=prompt,
                images=images_list, # Use list
                client=client,
                model_name=model_id,
                temperature=0.3,
                max_tokens=8000
            )
        
        elif model_name == "deepseek_r1":
            # DeepSeek gets text description only
            response_text, usage = call_deepseek(
                prompt=prompt,
                client=self.clients['deepseek'],
                model_name=model_config['model_id'],
                temperature=0.3,
                max_tokens=16000
            )
        
        elif model_name == "gpt_5_1":
            response_text, usage = call_gpt(
                prompt=prompt,
                images=images_list, # Use list
                client=self.clients['openai'],
                model_name=model_config['model_id'],
                temperature=0.3,
                max_tokens=16000
            )
        
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Parse JSON response
        response_json = self._parse_json_response(response_text, model_name)

        # --- FIX: INJECT ROOT FIELDS ---
        if isinstance(response_json, dict):
            response_json['question_id'] = question.get('question_id')
            response_json['year'] = question.get('year')
            response_json['subject'] = question.get('subject')
            response_json['exam_name'] = question.get('exam_name')
            response_json['question_type'] = question.get('question_type')
            response_json['question_text'] = question.get('question_text')
        # -------------------------------
        
        # Calculate cost
        cost = self.cost_tracker.calculate_cost(
            model_name=model_name,
            input_tokens=usage.get('input_tokens', 0),
            output_tokens=usage.get('output_tokens', 0)
        )
        
        elapsed_time = time.time() - start_time
        
        return {
            "model_name": model_name,
            "response_json": response_json,
            "tokens_used": usage.get('input_tokens', 0) + usage.get('output_tokens', 0),
            "input_tokens": usage.get('input_tokens', 0),
            "output_tokens": usage.get('output_tokens', 0),
            "cost": cost,
            "time_seconds": elapsed_time
        }
    
    def _parse_json_response(self, response_text: str, model_name: str) -> Dict:
        """
        Parse model response as JSON using robust repair
        """
        # Try json_repair first if available
        if repair_json:
            try:
                # repair_json automatically handles markdown blocks, missing braces, etc.
                response_json = repair_json(response_text, return_objects=True)
                
                # Check if it returned valid dict
                if isinstance(response_json, dict):
                    return response_json
                elif isinstance(response_json, list) and len(response_json) > 0:
                     # Sometimes returns list if input was list-like
                    if isinstance(response_json[0], dict):
                        return response_json[0]
            except Exception as e:
                logger.warning(f"json_repair failed for {model_name}: {e}")

        # Fallback to manual parsing logic
        try:
            # Strip whitespace
            response_text = response_text.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "", 1)
            if response_text.startswith("```"):
                response_text = response_text.replace("```", "", 1)
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            # Find JSON object (starts with {, ends with })
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}')
            
            if start_idx == -1 or end_idx == -1:
                # If repair failed and we can't find braces, it's likely not JSON
                raise ValueError("No JSON object found in response")
            
            json_text = response_text[start_idx:end_idx+1]
            
            # Parse JSON
            response_json = json.loads(json_text)
            
            return response_json
            
        except json.JSONDecodeError as e:
            logger.error(f"{model_name} returned invalid JSON: {e}")
            logger.debug(f"Response snippet: {response_text[:500]}...")
            raise ValueError(f"{model_name} JSON parse error: {e}")
    
    def _validate_responses(
        self,
        model_responses: Dict[str, Dict],
        question_id: str
    ) -> Dict[str, Dict]:
        """
        Validate all model responses against schema
        
        Checks:
        - Required tiers present (tier_1 through tier_4)
        - Required fields in each tier
        - Data types correct
        
        Args:
            model_responses: {model_name: response_data, ...}
            question_id: For logging
        
        Returns:
            dict: Validated responses (removes failed responses)
        """
        validated = {}
        
        for model_name, response_data in model_responses.items():
            if response_data is None:
                continue
            
            try:
                response_json = response_data['response_json']
                
                # Use validator utility
                is_valid, errors = validate_complete_schema(response_json)
                
                if not is_valid:
                    # Log specific missing tiers
                    required_tiers = [
                        'tier_1_core_research',
                        'tier_2_student_learning',
                        'tier_3_enhanced_learning',
                        'tier_4_metadata_and_future'
                    ]
                    missing = [t for t in required_tiers if t not in response_json]
                    
                    if missing:
                        present_tiers = [t for t in required_tiers if t in response_json]
                        
                        # Partial Acceptance Logic:
                        # If Tier 1 (answer/explanation) exists, we accept it but flag it
                        if 'tier_1_core_research' in response_json:
                            logger.warning(
                                f"Accepting partial response from {model_name}: "
                                f"Present: {present_tiers}, Missing: {missing}. "
                                f"This may cause more fields to go to debate."
                            )
                            validated[model_name] = response_data
                        else:
                            logger.error(f"Rejecting {model_name}: Missing Tier 1 (critical tier)")
                    else:
                        # Other validation error, log but accept if parsed
                        logger.warning(f"{model_name} validation errors: {errors[:3]}")
                        validated[model_name] = response_data
                else:
                    validated[model_name] = response_data
                
            except Exception as e:
                logger.error(f"Validation failed for {model_name}: {e}")
                continue
        
        if len(validated) < 2:
            raise RuntimeError(f"Only {len(validated)} models provided valid responses, need at least 2")
        
        return validated
    
    def _calculate_metadata(
        self,
        model_responses: Dict[str, Dict],
        weights: Dict[str, float]
    ) -> Dict:
        """
        Calculate aggregate metadata
        
        Returns:
            dict: {
                total_tokens, total_cost, total_time,
                per_model_breakdown
            }
        """
        total_tokens = sum(r.get('tokens_used', 0) for r in model_responses.values())
        total_cost = sum(r.get('cost', 0) for r in model_responses.values())
        
        # Safe max calculation
        times = [r.get('time_seconds', 0) for r in model_responses.values()]
        max_time = max(times) if times else 0
        
        per_model = {
            name: {
                'tokens': data.get('tokens_used', 0),
                'cost': data.get('cost', 0),
                'time': data.get('time_seconds', 0),
                'weight': weights.get(name, 0)
            }
            for name, data in model_responses.items()
        }
        
        return {
            'total_tokens': total_tokens,
            'total_cost': total_cost,
            'max_time_seconds': max_time,
            'per_model': per_model
        }


    def _save_debug_output(self, question_id: str, name: str, data: Any):
        """Save debug output to file for debugging purposes."""
        debug_dir = PROJECT_ROOT / "debug_outputs" / "model_orchestrator"
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{question_id}_{name}.json"
        filepath = debug_dir / filename
        
        try:
            # Prepare data for JSON serialization
            serializable_data = {}
            for model_name, response in data.items():
                serializable_data[model_name] = {
                    'response_json': response.get('response_json') if response else None,
                    'tokens_used': response.get('tokens_used') if response else None,
                    'cost': response.get('cost') if response else None
                }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=2, ensure_ascii=False, default=str)
            logger.debug(f"Saved debug: {filepath}")
        except Exception as e:
            logger.warning(f"Failed to save debug {filename}: {e}")


def main():
    """
    Test model orchestrator
    """
    import argparse
    from pipeline.scripts.init_00_initialization import PipelineInitializer
    from pipeline.scripts.init_01_question_loader import QuestionLoader
    from pipeline.scripts.init_02_question_classifier import QuestionClassifier
    from pipeline.scripts.init_04_retrieval_dense import DenseRetriever
    from pipeline.scripts.init_05_retrieval_sparse import SparseRetriever
    from pipeline.scripts.init_06_retrieval_merger import RetrievalMerger
    
    parser = argparse.ArgumentParser(description="Test Model Orchestrator")
    parser.add_argument("--question-json", required=True, help="Path to question JSON file")
    
    args = parser.parse_args()
    
    # Initialize
    print("\n" + "="*80)
    print("INITIALIZING")
    print("="*80 + "\n")
    
    initializer = PipelineInitializer()
    components = initializer.initialize_all()
    
    # Load question
    print("\n" + "="*80)
    print("LOADING QUESTION")
    print("="*80 + "\n")
    
    loader = QuestionLoader(os.path.dirname(args.question_json))
    question = loader.load_question(args.question_json)
    
    # Classify
    print("\n" + "="*80)
    print("CLASSIFYING")
    print("="*80 + "\n")
    
    classifier = QuestionClassifier(components['configs'], components['clients'])
    classification = classifier.classify_question(question)
    
    # Retrieve RAG
    print("\n" + "="*80)
    print("RETRIEVING RAG CONTEXT")
    print("="*80 + "\n")
    
    dense = DenseRetriever(components['clients']['qdrant'], components['embedding_model'], components['configs'])
    sparse = SparseRetriever(components['clients']['redis'], components['configs'])
    merger = RetrievalMerger(components['configs'])
    
    dense_res = dense.retrieve(question)
    sparse_res = sparse.retrieve(question)
    rag_chunks = merger.merge(dense_res, sparse_res, question)
    
    # Run Orchestrator
    print("\n" + "="*80)
    print("RUNNING MODEL ORCHESTRATOR")
    print("="*80 + "\n")
    
    orchestrator = ModelOrchestrator(components['configs'], components['clients'])
    result = orchestrator.generate_responses(
        question=question,
        classification=classification,
        rag_chunks=rag_chunks
    )
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Total Cost: ${result['metadata']['total_cost']:.4f}")
    print(f"Total Tokens: {result['metadata']['total_tokens']}")
    
    for model, response in result['model_responses'].items():
        print(f"\nModel: {model}")
        print(f"Status: {'Success' if response else 'Failed'}")
        if response:
            print(f"Tokens: {response['tokens_used']}")
            # Print first 200 chars of JSON to verify
            print(f"Response Preview: {str(response['response_json'])[:200]}...")


if __name__ == "__main__":
    main()