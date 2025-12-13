"""
============================================================================
STAGE 2: QUESTION CLASSIFICATION
============================================================================
Purpose: Classify question into content_type, media_type, difficulty, determine GPT-5.1 usage
Used by: 99_pipeline_runner.py
Model: Gemini 2.0 Flash Exp (cheap/free)
Author: GATE AE SOTA Pipeline
============================================================================
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List
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
    from pipeline.utils.api_wrappers import call_gemini
except ImportError:
    # Fallback for direct execution
    sys.path.append(str(current_file.parent.parent))
    from utils.logging_utils import setup_logger, log_stage
    from utils.prompt_builder import PromptBuilder
    from utils.api_wrappers import call_gemini

logger = setup_logger("02_question_classifier")


class QuestionClassifier:
    """
    Classify GATE AE questions and determine processing requirements.
    Maps classification results to weight strategies defined in weights_config.yaml.
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
        
        # Load weight strategies configuration
        self.weight_strategies = configs['weights_config']['weight_strategies']
        # Load type mapping (question type -> strategy name)
        self.type_mapping = configs['weights_config']['question_type_to_strategy']
    
    @log_stage("Stage 2: Classification")
    def classify_question(self, question: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify question and determine GPT-5.1 usage
        
        Args:
            question: Question object from Stage 1
        
        Returns:
            dict: tier_0_classification (8 fields)
        """
        question_id = question.get('question_id', 'unknown')
        logger.info(f"Classifying question: {question_id}")
        
        # Build classification prompt
        prompt = self._build_prompt(question)
        
        # Call Gemini 2.0 Flash Exp
        logger.info("  Calling Gemini 2.0 Flash Exp for classification...")
        try:
            # Note: Using max_tokens=500 as classification response is small
            response_text, usage = call_gemini(
                prompt=prompt,
                client=self.clients['google_genai'],
                model_name="gemini-2.0-flash-exp",
                temperature=0.1,  # Low temperature for consistent classification
                max_tokens=500
            )
            
            # Parse response
            classification = self._parse_response(response_text)
            
            # Determine weight strategy based on content/media type
            classification = self._determine_weight_strategy(classification)
            
            # Add metadata
            classification['classification_method'] = "llm_assisted"
            classification['classifier_model'] = "gemini_2.0_flash_exp"
            
            # Validate
            self._validate_classification(classification)
            
            logger.info(f"  ✓ Classification complete:")
            logger.info(f"    - Content: {classification.get('content_type')}")
            logger.info(f"    - Media: {classification.get('media_type')}")
            logger.info(f"    - Difficulty: {classification.get('difficulty_score')}/10")
            logger.info(f"    - Use GPT-5.1: {classification.get('use_gpt51')}")
            logger.info(f"    - Weight Strategy: {classification.get('weight_strategy')}")
            
            return classification
            
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            logger.warning("Using fallback classification (BALANCED)")
            return self._create_fallback_classification()
    
    def _build_prompt(self, question: Dict) -> str:
        """Build classification prompt with question data"""
        
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
        
        # Fill variables
        variables = {
            "question_text": question['question_text'],
            "options": options_text,
            "has_image": str(question.get('has_question_image', False))
        }
        
        prompt = self.prompt_builder.build_prompt(
            prompt_name="classification",
            variables=variables
        )
        
        return prompt
    
    def _parse_response(self, response: str) -> Dict:
        """
        Parse Gemini response JSON
        
        Expected fields:
            - content_type
            - media_type
            - difficulty_score
            - complexity_flags (dict of 7 booleans)
            - use_gpt51
            - classification_confidence
            - classification_reasoning
        """
        try:
            # Handle tuple response (from API wrapper returns)
            if isinstance(response, tuple):
                response = response[0] if len(response) > 0 else ""
            
            # Ensure response is a string
            response = str(response)
            
            # Strip markdown if present
            response = response.strip()
            if response.startswith("```json"):
                response = response.replace("```json", "").replace("```", "").strip()
            elif response.startswith("```"):
                response = response.replace("```", "").strip()
            
            classification = json.loads(response)
            
            return classification
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse classification JSON: {e}")
            logger.error(f"Response: {response[:500]}")
            raise ValueError(f"Invalid JSON response from classifier: {e}")
    
    def _determine_weight_strategy(self, classification: Dict) -> Dict:
        """
        Determine weight strategy based on content_type and media_type.
        
        Logic:
        1. Construct 'combined_type' key (e.g., 'mathematical_derivation_text_only')
        2. Lookup strategy name in 'question_type_to_strategy' mapping from config
        3. Verify strategy exists in 'weight_strategies'
        4. Fallback to 'BALANCED' if not found
        """
        content_type = classification.get('content_type', 'unknown')
        media_type = classification.get('media_type', 'text_only')
        
        # Create combined type key
        combined_type = f"{content_type}_{media_type}"
        classification['combined_type'] = combined_type
        
        # Lookup strategy name from config mapping
        # This uses the 'question_type_to_strategy' section from weights_config.yaml
        strategy_name = self.type_mapping.get(combined_type)
        
        # If direct match fails, try fallback logic based on content
        if not strategy_name:
            if media_type == "image_based":
                if "conceptual" in content_type:
                    strategy_name = "VISION_WEIGHTED"
                else:
                    strategy_name = "MATH_IMAGE_HYBRID"
            elif "mathematical" in content_type or "numerical" in content_type:
                strategy_name = "MATH_WEIGHTED"
            elif "theory" in content_type:
                strategy_name = "CONCEPTUAL_WEIGHTED"
            else:
                strategy_name = "BALANCED"
            
            logger.warning(f"Strategy mapping not found for '{combined_type}', inferred: {strategy_name}")

        # Validate strategy exists in defined strategies
        if strategy_name not in self.weight_strategies:
            logger.warning(f"Unknown weight strategy: {strategy_name}, defaulting to BALANCED")
            strategy_name = "BALANCED"
        
        classification['weight_strategy'] = strategy_name
        return classification
    
    def _validate_classification(self, classification: Dict):
        """
        Validate classification output
        
        Required fields:
            - content_type (one of 4 options)
            - media_type (one of 2 options)
            - combined_type
            - weight_strategy
            - difficulty_score (1-10)
            - complexity_flags (dict)
            - use_gpt51 (boolean)
            - classification_confidence (0.0-1.0)
            - classification_method
            - classifier_model
            - classification_reasoning
        """
        required_fields = [
            'content_type', 'media_type', 'combined_type', 'weight_strategy',
            'difficulty_score', 'complexity_flags', 'use_gpt51',
            'classification_confidence'
        ]
        
        missing = [f for f in required_fields if f not in classification]
        if missing:
            raise ValueError(f"Classification missing fields: {missing}")
        
        # Validate content_type
        valid_content_types = [
            "conceptual_theory",
            "mathematical_derivation",
            "numerical_calculation",
            "conceptual_application"
        ]
        if classification['content_type'] not in valid_content_types:
            logger.warning(f"Invalid content_type: {classification['content_type']}, using conceptual_application")
            classification['content_type'] = "conceptual_application"
        
        # Validate media_type
        valid_media_types = ["text_only", "image_based"]
        if classification['media_type'] not in valid_media_types:
            logger.warning(f"Invalid media_type: {classification['media_type']}, using text_only")
            classification['media_type'] = "text_only"
        
        # Validate difficulty_score
        try:
            score = int(classification['difficulty_score'])
            if not (1 <= score <= 10):
                classification['difficulty_score'] = 5
                logger.warning("Invalid difficulty_score range, reset to 5")
            else:
                classification['difficulty_score'] = score
        except (ValueError, TypeError):
            classification['difficulty_score'] = 5
            
        # Validate confidence
        try:
            conf = float(classification['classification_confidence'])
            if not (0.0 <= conf <= 1.0):
                classification['classification_confidence'] = 0.5
            else:
                classification['classification_confidence'] = conf
        except (ValueError, TypeError):
            classification['classification_confidence'] = 0.5
            
        # Ensure complexity_flags is dict
        if not isinstance(classification.get('complexity_flags'), dict):
            classification['complexity_flags'] = {}
            
        # Ensure use_gpt51 is boolean
        if not isinstance(classification.get('use_gpt51'), bool):
            classification['use_gpt51'] = False

    def batch_classify(self, questions: list) -> list:
        """
        Classify multiple questions
        
        Args:
            questions: List of question objects
        
        Returns:
            list: List of tier_0_classification dicts
        """
        logger.info(f"Batch classifying {len(questions)} questions...")
        
        classifications = []
        
        for i, question in enumerate(questions, 1):
            try:
                classification = self.classify_question(question)
                classifications.append(classification)
                
                if i % 50 == 0:
                    logger.info(f"  Progress: {i}/{len(questions)} classified")
                    
            except Exception as e:
                logger.error(f"Failed to classify {question.get('question_id')}: {e}")
                # Create fallback classification
                classifications.append(self._create_fallback_classification())
        
        logger.info(f"✓ Batch classification complete: {len(classifications)}/{len(questions)}")
        
        return classifications
    
    def _create_fallback_classification(self) -> Dict:
        """
        Create fallback classification if API call fails
        """
        return {
            "content_type": "conceptual_application",
            "media_type": "text_only",
            "combined_type": "conceptual_application_text_only",
            "weight_strategy": "BALANCED",
            "difficulty_score": 5,
            "complexity_flags": {
                "requires_derivation": False,
                "multi_concept_integration": False,
                "ambiguous_wording": False,
                "image_interpretation_complex": False,
                "edge_case_scenario": False,
                "multi_step_reasoning": False,
                "approximation_needed": False
            },
            "use_gpt51": False,
            "classification_confidence": 0.0,
            "classification_method": "fallback",
            "classifier_model": "none",
            "classification_reasoning": "API call failed, using fallback classification"
        }


def main():
    """
    Test question classifier
    """
    import argparse
    from pipeline.scripts.init_00_initialization import PipelineInitializer
    from pipeline.scripts.init_01_question_loader import QuestionLoader
    
    parser = argparse.ArgumentParser(description="Test Question Classifier")
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
    
    loader = QuestionLoader(
        questions_dir=os.path.dirname(args.question_json)
    )
    question = loader.load_question(args.question_json)
    
    # Classify
    print("\n" + "="*80)
    print("CLASSIFYING QUESTION")
    print("="*80 + "\n")
    
    classifier = QuestionClassifier(
        configs=components['configs'],
        clients=components['clients']
    )
    
    classification = classifier.classify_question(question)
    
    # Print result
    print("\n" + "="*80)
    print("CLASSIFICATION RESULT")
    print("="*80)
    print(json.dumps(classification, indent=2))


if __name__ == "__main__":
    main()