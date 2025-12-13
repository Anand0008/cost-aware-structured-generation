"""
============================================================================
STAGE 1: QUESTION LOADING
============================================================================
Purpose: Load raw question JSON and image, create standardized question object
Used by: 99_pipeline_runner.py
Author: GATE AE SOTA Pipeline
============================================================================
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image
import base64
from datetime import datetime

import sys
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from utils.logging_utils import setup_logger, log_stage

logger = setup_logger("01_question_loader")


class QuestionLoader:
    """
    Load and validate GATE AE questions from JSON files
    """
    
    def __init__(self, questions_dir: str, images_dir: str = None):
        """
        Args:
            questions_dir: Directory containing question JSON files
            images_dir: Directory containing question images (default: same as questions_dir)
        """
        self.questions_dir = Path(questions_dir)
        self.images_dir = Path(images_dir) if images_dir else self.questions_dir
        
        if not self.questions_dir.exists():
            raise FileNotFoundError(f"Questions directory not found: {self.questions_dir}")
    
    @log_stage("Stage 1: Question Loading")
    def load_question(self, question_file) -> Dict[str, Any]:
        """
        Load a single question from JSON file
        
        Args:
            question_file: Either:
                - str: Path to individual question JSON file
                - tuple: (file_path, question_id, index) for array-based files
        
        Returns:
            dict: Standardized question object with 14 root fields
        """
        # Handle both individual files and array-based files
        if isinstance(question_file, tuple):
            # Array-based file: (file_path, question_id, index)
            file_path, question_id, index = question_file
            question_path = Path(file_path)
            
            logger.info(f"Loading question: {question_id} from {question_path.name}")
            
            # Load array and extract specific question
            with open(question_path, 'r', encoding='utf-8') as f:
                questions_array = json.load(f)
            
            raw_data = questions_array[index]
        else:
            # Individual file
            question_path = self._resolve_path(question_file)
            logger.info(f"Loading question: {question_path.name}")
            
            # Load JSON
            with open(question_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
        
        # Create standardized question object
        question = self._create_question_object(raw_data, question_path)
        
        # Load image if exists
        if question['has_question_image']:
            question = self._load_image(question, question_path)
        
        # Validate
        self._validate_question(question)
        
        logger.info(f"✓ Question loaded: {question['question_id']} ({question['year']})")
        
        return question
    
    def _resolve_path(self, question_file: str) -> Path:
        """Resolve question file path"""
        path = Path(question_file)
        
        if path.is_absolute() and path.exists():
            return path
        
        # Try relative to questions_dir
        relative_path = self.questions_dir / path
        if relative_path.exists():
            return relative_path
        
        raise FileNotFoundError(f"Question file not found: {question_file}")
    
    def _create_question_object(self, raw_data: Dict, question_path: Path) -> Dict[str, Any]:
        """
        Create standardized question object from raw JSON
        
        Returns:
            dict with 14 root fields:
                - question_id, exam_name, subject, year, question_number
                - question_text, question_text_latex, question_type
                - marks, negative_marks, options, answer_key
                - has_question_image, image_metadata
        """
        # Extract base fields with defaults
        question = {
            # Identification
            "question_id": raw_data.get("id") or raw_data.get("question_id") or f"unknown_{datetime.now().timestamp()}",
            "exam_name": raw_data.get("exam_name", "GATE AE"),
            "subject": raw_data.get("subject", "Aerospace Engineering"),
            "year": int(raw_data.get("year", 0)),
            "question_number": raw_data.get("question_number") or raw_data.get("q_num"),
            
            # Question content
            "question_text": raw_data.get("question_text") or raw_data.get("question", ""),
            "question_text_latex": raw_data.get("question_text_latex") or raw_data.get("question_latex"),
            "question_type": raw_data.get("question_type", "MCQ"),  # MCQ, MCQ_LINKED, NAT, MSQ
            
            # Scoring
            "marks": float(raw_data.get("marks") or 1.0),
            "negative_marks": float(raw_data.get("negative_marks") or 0.0),
            
            # Options and answer
            "options": self._parse_options(raw_data),
            "answer_key": raw_data.get("answer_key") or raw_data.get("correct_answer"),
            
            # Image
            "has_question_image": raw_data.get("has_image", False) or raw_data.get("has_question_image", False),
            "image_metadata": raw_data.get("image_metadata"),  # Preserve if exists in JSON
            
            # LINKED QUESTION FIELDS (CRITICAL for context injection)
            "common_data_statement": raw_data.get("common_data_statement"),
            "common_data_statement_latex": raw_data.get("common_data_statement_latex"),
            "linked_question_group": raw_data.get("linked_question_group"),
        }
        
        return question
    
    def _parse_options(self, raw_data: Dict) -> Optional[Dict[str, str]]:
        """
        Parse options from various JSON formats
        
        Supports:
            - {"options": {"A": "...", "B": "...", ...}}
            - {"option_a": "...", "option_b": "...", ...}
            - {"options": ["...", "...", "...", "..."]}
        
        Returns:
            dict: {"A": "text", "B": "text", "C": "text", "D": "text"} or None for NAT
        """
        # For NAT (Numerical Answer Type), no options
        if raw_data.get("question_type") == "NAT":
            return None
        
        # Format 1: {"options": {"A": "...", "B": "...", ...}}
        if "options" in raw_data and isinstance(raw_data["options"], dict):
            return raw_data["options"]
        
        # Format 2: {"option_a": "...", "option_b": "...", ...}
        options = {}
        for key in ["A", "B", "C", "D"]:
            option_key = f"option_{key.lower()}"
            if option_key in raw_data:
                options[key] = raw_data[option_key]
        
        if options:
            return options
        
        # Format 3: {"options": ["...", "...", "...", "..."]}
        if "options" in raw_data and isinstance(raw_data["options"], list):
            labels = ["A", "B", "C", "D"]
            return {labels[i]: opt for i, opt in enumerate(raw_data["options"][:4])}
        
        # No options found
        logger.warning("No options found in question JSON")
        return None
    
    def _load_image(self, question: Dict, question_path: Path) -> Dict:
        """
        Load question image and extract metadata
        
        Searches for image by question number pattern in year folder:
            - Extracts Q number from question_id (e.g., GATE_AE_2007_Q25 -> Q25)
            - Looks in images_dir/{year}/ for any image containing Q25 (case-insensitive)
        
        Updates question object with:
            - image_metadata: {path, format, width, height, size_kb, base64}
        """
        import re
        
        question_id = question.get('question_id', '')
        year = str(question.get('year', ''))
        
        # Extract question number from question_id (e.g., Q25, Q3)
        q_match = re.search(r'Q(\d+)', question_id, re.IGNORECASE)
        if not q_match:
            logger.warning(f"Cannot extract Q number from: {question_id}")
            question['has_question_image'] = False
            return question
        
        q_num = q_match.group(1)  # e.g., "25"
        q_pattern = f"Q{q_num}"   # e.g., "Q25"
        
        # Build list of directories to search
        search_dirs = []
        if year:
            search_dirs.append(self.images_dir / year)
        search_dirs.append(self.images_dir)
        
        # Search for image containing Q{number} in filename
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            
            for img_file in search_dir.iterdir():
                if not img_file.is_file():
                    continue
                
                ext = img_file.suffix.lower()
                if ext not in ['.png', '.jpg', '.jpeg']:
                    continue
                
                # Check if filename contains Q{number} (case-insensitive)
                # Match Q25 but not Q250 or Q125
                pattern = rf'Q{q_num}(?!\d)'
                if re.search(pattern, img_file.stem, re.IGNORECASE):
                    logger.info(f"  Found image: {img_file} (matched {q_pattern})")
                    
                    try:
                        # Load image
                        img = Image.open(img_file)
                        
                        # Convert to RGB if necessary
                        if img.mode not in ['RGB', 'RGBA']:
                            img = img.convert('RGB')
                        
                        # Get file size
                        size_kb = img_file.stat().st_size / 1024
                        
                        # Convert to base64
                        with open(img_file, 'rb') as f:
                            image_bytes = f.read()
                            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                        
                        # Update metadata
                        question['image_metadata'] = {
                            "path": str(img_file),
                            "format": img.format or ext[1:].upper(),
                            "width": img.width,
                            "height": img.height,
                            "size_kb": round(size_kb, 2),
                            "base64": image_base64
                        }
                        
                        logger.info(f"  ✓ Image loaded: {img.width}x{img.height} {img.format} ({size_kb:.1f} KB)")
                        
                        return question
                        
                    except Exception as e:
                        logger.error(f"Failed to load image {img_file}: {e}")
                        question['has_question_image'] = False
                        return question
        
        # Image not found
        logger.warning(f"Image file not found for: {question_id} ({q_pattern} in year {year})")
        question['has_question_image'] = False
        return question
    
    def _validate_question(self, question: Dict):
        """
        Validate question object has required fields
        
        Raises:
            ValueError: If validation fails
        """
        required_fields = [
            "question_id",
            "year",
            "question_text",
            "marks"
        ]
        
        missing = [field for field in required_fields if not question.get(field)]
        
        if missing:
            raise ValueError(f"Missing required fields: {missing}")
        
        # Validate year range
        if not (2000 <= question['year'] <= 2030):
            logger.warning(f"Unusual year value: {question['year']}")
        
        # Validate marks
        if question['marks'] <= 0:
            raise ValueError(f"Invalid marks value: {question['marks']}")
        
        # Validate MCQ has options
        if question['question_type'] == "MCQ" and not question['options']:
            logger.warning("MCQ question has no options")
        
        # Validate has answer key
        if not question.get('answer_key'):
            logger.warning("Question has no answer key")
    
    def load_batch(self, question_files: list) -> list:
        """
        Load multiple questions
        
        Args:
            question_files: List of question file paths
        
        Returns:
            list: List of question objects
        """
        questions = []
        
        logger.info(f"Loading batch of {len(question_files)} questions...")
        
        for i, qfile in enumerate(question_files, 1):
            try:
                question = self.load_question(qfile)
                questions.append(question)
                
                if i % 50 == 0:
                    logger.info(f"  Progress: {i}/{len(question_files)} questions loaded")
                    
            except Exception as e:
                logger.error(f"Failed to load {qfile}: {e}")
                continue
        
        logger.info(f"✓ Batch loaded: {len(questions)}/{len(question_files)} successful")
        
        return questions
    
    def get_all_questions_in_directory(self, pattern: str = "*.json") -> list:
        """
        Get all question files in directory matching pattern
        
        Handles two formats:
        1. Individual question files (e.g., GATE_2024_AE_Q19.json)
        2. Year-based array files (e.g., 2024_json.json containing multiple questions)
        
        Args:
            pattern: File pattern (default: *.json)
        
        Returns:
            list: List of question file paths or tuples (file_path, question_id, index)
        """
        json_files = sorted(self.questions_dir.glob(pattern))
        all_questions = []
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Check if it's an array of questions
                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                    # Year-based file - create tuple for each question
                    for idx, question in enumerate(data):
                        if 'question_id' in question:
                            all_questions.append((str(json_file), question['question_id'], idx))
                else:
                    # Individual question file
                    all_questions.append(str(json_file))
            except:
                # If we can't parse it, treat as individual file
                all_questions.append(str(json_file))
        
        logger.info(f"Found {len(all_questions)} question files in {self.questions_dir}")
        return all_questions


def main():
    """
    Test question loader
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Question Loader")
    parser.add_argument("--questions-dir", required=True, help="Directory with question JSON files")
    parser.add_argument("--images-dir", help="Directory with images (default: same as questions-dir)")
    parser.add_argument("--file", help="Specific question file to load")
    
    args = parser.parse_args()
    
    loader = QuestionLoader(args.questions_dir, args.images_dir)
    
    if args.file:
        # Load single question
        question = loader.load_question(args.file)
        print("\n" + "="*80)
        print("QUESTION LOADED")
        print("="*80)
        print(json.dumps(question, indent=2, default=str))
    else:
        # Load all questions
        files = loader.get_all_questions_in_directory()
        questions = loader.load_batch(files[:10])  # Load first 10 as test
        
        print("\n" + "="*80)
        print(f"LOADED {len(questions)} QUESTIONS")
        print("="*80)
        
        for q in questions[:3]:
            print(f"\n{q['question_id']} ({q['year']}): {q['question_text'][:100]}...")


if __name__ == "__main__":
    main()