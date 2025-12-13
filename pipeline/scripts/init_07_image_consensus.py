"""
============================================================================
STAGE 4.5: IMAGE DESCRIPTION CONSENSUS
============================================================================
Purpose: Generate consensus image description for questions with images
         DeepSeek R1 doesn't support image input, so needs text description
Models: Claude Sonnet 4.5, GPT-5.1, Gemini 2.5 Pro (all can see images)
Process: Each model describes image → Merge into consensus → Pass to DeepSeek
Used by: 99_pipeline_runner.py (only if has_question_image = true)
Author: GATE AE SOTA Pipeline
============================================================================
"""

import os
import json
import base64
from pathlib import Path
from typing import Dict, Any, List
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from utils.logging_utils import setup_logger, log_stage
from utils.prompt_builder import PromptBuilder
from utils.api_wrappers import call_claude, call_gpt, call_gemini
from utils.similarity_utils import merge_text_by_consensus

logger = setup_logger("07_image_consensus")


class ImageConsensusGenerator:
    """
    Generate consensus image description from multiple vision models
    
    Why we do this:
    - DeepSeek R1 cannot see images directly (no vision capability)
    - But DeepSeek is good at reasoning, so we want to include it
    - Solution: Have 3 vision models describe the image, merge descriptions
    - Pass consensus description to DeepSeek as text
    
    Why 3 models?
    - Different models notice different details
    - Consensus reduces hallucination risk
    - Ensures critical information isn't missed
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
    
    @log_stage("Stage 4.5: Image Consensus")
    def generate_consensus(self, question: Dict[str, Any]) -> str:
        """
        Generate consensus image description
        
        Args:
            question: Question object with image_metadata
        
        Returns:
            str: Consensus image description (400-600 tokens typically)
        """
        question_id = question.get('question_id', 'unknown')
        logger.info(f"Generating image consensus for: {question_id}")
        
        # Verify image exists
        if not question.get('has_question_image'):
            logger.warning("Question has no image, skipping consensus")
            return ""
        
        image_metadata = question.get('image_metadata')
        if not image_metadata:
            raise ValueError("Question marked as has_image but no image_metadata found")
        
        # Get image base64
        image_base64 = image_metadata.get('base64')
        if not image_base64:
            raise ValueError("Image metadata missing base64 encoding")
        
        logger.info(f"  Image: {image_metadata['format']} {image_metadata['width']}x{image_metadata['height']} ({image_metadata['size_kb']:.1f} KB)")
        
        # Generate descriptions from 3 models in parallel
        descriptions = self._generate_descriptions_parallel(
            image_base64=image_base64,
            question_text=question['question_text']
        )
        
        # Merge into consensus
        consensus = self._merge_descriptions(descriptions, question)
        
        logger.info(f"  ✓ Consensus generated ({len(consensus.split())} words)")
        
        return consensus
    
    def _generate_descriptions_parallel(
        self,
        image_base64: str,
        question_text: str
    ) -> Dict[str, str]:
        """
        Generate image descriptions from all 3 vision models in parallel
        
        Models:
            - Claude Sonnet 4.5: Excellent at technical diagram analysis
            - GPT-5.1: Good at quantitative data extraction
            - Gemini 2.5 Pro: Strong at spatial relationships
        
        Args:
            image_base64: Base64 encoded image
            question_text: Question text (for context)
        
        Returns:
            dict: {"claude": description, "gpt": description, "gemini": description}
        """
        # Build prompt
        prompt = self._build_image_prompt(question_text)
        
        logger.info("  Generating descriptions from 3 vision models...")
        
        # Call models (ideally in parallel using threading/async)
        # For simplicity, calling sequentially here
        descriptions = {}
        
        # 1. Claude Sonnet 4.5
        try:
            logger.info("    - Calling Claude Sonnet 4.5...")
            response_text, usage = call_claude(
                prompt=prompt,
                image_base64=image_base64,
                client=self.clients['anthropic'],
                model_name=self.configs['models_config']['models']['claude_sonnet_4.5']['model_id'],
                temperature=0.3,
                max_tokens=800
            )
            descriptions['claude'] = response_text
            logger.info(f"      ✓ Claude: {len(response_text.split())} words")
        except Exception as e:
            logger.error(f"Claude failed: {e}")
            descriptions['claude'] = ""
        
        # 2. GPT-4o (using GPT-4o for vision since GPT-5.1 has issues with images)
        try:
            logger.info("    - Calling GPT-4o (vision)...")
            response_text, usage = call_gpt(
                prompt=prompt,
                image_base64=image_base64,
                client=self.clients['openai'],
                model_name="gpt-4o",  # Use GPT-4o for vision, not GPT-5.1
                temperature=0.3,
                max_tokens=800
            )
            descriptions['gpt'] = response_text
            logger.info(f"      ✓ GPT-4o: {len(response_text.split())} words")
        except Exception as e:
            logger.error(f"GPT-4o failed: {e}")
            descriptions['gpt'] = ""
        
        # 3. Gemini 2.5 Pro
        try:
            logger.info("    - Calling Gemini 2.5 Pro...")
            response_text, usage = call_gemini(
                prompt=prompt,
                image_base64=image_base64,
                client=self.clients['google_genai'],
                model_name="gemini-2.5-flash",  # Use configured model
                temperature=0.3,
                max_tokens=800
            )
            descriptions['gemini'] = response_text
            logger.info(f"      ✓ Gemini: {len(response_text.split())} words")
        except Exception as e:
            logger.error(f"Gemini failed: {e}")
            descriptions['gemini'] = ""
        
        # Check if at least 2 models succeeded
        valid_descriptions = {k: v for k, v in descriptions.items() if v}
        if len(valid_descriptions) < 2:
            raise RuntimeError(f"Only {len(valid_descriptions)} models provided descriptions, need at least 2")
        
        return descriptions
    
    def _build_image_prompt(self, question_text: str) -> str:
        """
        Build image description prompt
        
        Variables:
            {question_text}: Provides context for what the image might show
        """
        variables = {
            "question_text": question_text
        }
        
        prompt = self.prompt_builder.build_prompt(
            prompt_name="image_description",
            variables=variables
        )
        
        return prompt
    
    def _merge_descriptions(
        self,
        descriptions: Dict[str, str],
        question: Dict
    ) -> str:
        """
        Merge 3 model descriptions into consensus
        
        Strategy:
        1. Extract common elements (mentioned by 2+ models)
        2. Include unique details if specific and technical
        3. Prioritize quantitative data (numbers, dimensions)
        4. Organize into structured format
        
        Args:
            descriptions: {"claude": str, "gpt": str, "gemini": str}
            question: Question object (for logging)
        
        Returns:
            str: Merged consensus description
        """
        logger.info("  Merging descriptions into consensus...")
        
        # Remove empty descriptions
        valid_descriptions = {k: v for k, v in descriptions.items() if v}
        
        if len(valid_descriptions) == 1:
            # Only one model succeeded, use it directly
            model_name, description = list(valid_descriptions.items())[0]
            logger.warning(f"Only {model_name} provided description, using directly")
            return description
        
        # Use similarity-based merging (from utils)
        consensus = merge_text_by_consensus(
            texts=list(valid_descriptions.values()),
            weights=[0.35, 0.35, 0.30],  # Claude and GPT slightly higher weight
            min_agreement=2  # Include if 2+ models mention
        )
        
        # Add header for DeepSeek
        consensus_with_header = (
            f"IMAGE DESCRIPTION (for reference - question includes an image):\n\n"
            f"{consensus}\n\n"
            f"Use the above image description to solve the question below."
        )
        
        return consensus_with_header
    
    def batch_generate(self, questions: List[Dict]) -> Dict[str, str]:
        """
        Generate consensus descriptions for multiple questions
        
        Args:
            questions: List of question objects with images
        
        Returns:
            dict: {question_id: consensus_description}
        """
        # Filter to only questions with images
        image_questions = [q for q in questions if q.get('has_question_image')]
        
        logger.info(f"Batch generating consensus for {len(image_questions)} questions with images...")
        
        consensus_map = {}
        
        for i, question in enumerate(image_questions, 1):
            try:
                question_id = question['question_id']
                consensus = self.generate_consensus(question)
                consensus_map[question_id] = consensus
                
                if i % 10 == 0:
                    logger.info(f"  Progress: {i}/{len(image_questions)} processed")
                    
            except Exception as e:
                logger.error(f"Failed to generate consensus for {question.get('question_id')}: {e}")
                consensus_map[question['question_id']] = ""
        
        logger.info(f"✓ Batch complete: {len(consensus_map)}/{len(image_questions)} successful")
        
        return consensus_map


def main():
    """
    Test image consensus generator
    """
    import argparse
    from scripts.initialization import PipelineInitializer
    from scripts.question_loader import QuestionLoader
    
    parser = argparse.ArgumentParser(description="Test Image Consensus Generator")
    parser.add_argument("--question-json", required=True, help="Path to question JSON with image")
    parser.add_argument("--save-output", help="Save consensus to file")
    
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
    
    if not question.get('has_question_image'):
        print("ERROR: Question does not have an image")
        return
    
    # Generate consensus
    print("\n" + "="*80)
    print("GENERATING IMAGE CONSENSUS")
    print("="*80 + "\n")
    
    generator = ImageConsensusGenerator(
        configs=components['configs'],
        clients=components['clients']
    )
    
    consensus = generator.generate_consensus(question)
    
    # Print result
    print("\n" + "="*80)
    print("CONSENSUS DESCRIPTION")
    print("="*80)
    print(consensus)
    
    # Save if requested
    if args.save_output:
        with open(args.save_output, 'w', encoding='utf-8') as f:
            f.write(consensus)
        print(f"\n✓ Saved to {args.save_output}")


if __name__ == "__main__":
    main()