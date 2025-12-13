"""
============================================================================
MAIN PIPELINE RUNNER (ORCHESTRATOR)
============================================================================
Purpose: Run complete GATE AE question tagging pipeline (Stages 0-9)
Features:
    - Question-by-question processing
    - Year-by-year filtering
    - Resume from checkpoint (crash recovery)
    - Progress tracking and reporting
    - Cost monitoring and budget alerts
    - Flexible execution modes (single/batch/year/all)

Execution Modes:
    1. Single question: --question-id GATE_AE_2024_Q15
    2. Year filter: --year 2024
    3. Question range: --start-index 0 --end-index 100
    4. All questions: --all
    5. Resume from checkpoint: --resume

Usage Examples:
    # Process single question
    python 99_pipeline_runner.py --question-id GATE_AE_2024_Q15
    
    # Process all 2024 questions
    python 99_pipeline_runner.py --year 2024
    
    # Process first 100 questions
    python 99_pipeline_runner.py --start-index 0 --end-index 100
    
    # Process all 1,300 questions
    python 99_pipeline_runner.py --all
    
    # Resume from last checkpoint
    python 99_pipeline_runner.py --resume

Author: GATE AE SOTA Pipeline
Version: 1.0.0
============================================================================
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import all pipeline stages
from pipeline.scripts.init_00_initialization import PipelineInitializer
from pipeline.scripts.init_01_question_loader import QuestionLoader
from pipeline.scripts.init_02_question_classifier import QuestionClassifier
from pipeline.scripts.init_03_cache_manager import CacheManager
from pipeline.scripts.init_04_retrieval_dense import DenseRetriever
from pipeline.scripts.init_05_retrieval_sparse import SparseRetriever
from pipeline.scripts.init_06_retrieval_merger import RetrievalMerger
from pipeline.scripts.init_07_image_consensus import ImageConsensusGenerator
from pipeline.scripts.init_08_model_orchestrator import ModelOrchestrator
from pipeline.scripts.init_09_voting_engine import VotingEngine
from pipeline.scripts.init_10_debate_orchestrator import DebateOrchestrator
from pipeline.scripts.init_11_synthesis_engine import SynthesisEngine
from pipeline.scripts.init_12_output_manager import OutputManager

# Import utilities
from pipeline.utils.logging_utils import setup_logger
from pipeline.utils.cost_tracker import CostTracker
from pipeline.utils.checkpoint_manager import CheckpointManager
from pipeline.utils.health_monitor import HealthMonitor

logger = setup_logger("99_pipeline_runner")


class PipelineRunner:
    """
    Main pipeline orchestrator
    
    Runs all 10 stages (0-9) for each question:
    0. Initialization
    1. Question Loading
    2. Classification
    3. Cache Check
    4. RAG Retrieval (4A Dense + 4B Sparse + 4C Merge)
    4.5. Image Consensus (if has_image)
    5. Model Orchestration (parallel generation)
    6. Voting Engine
    7. Debate Orchestrator (if disputes)
    8. Synthesis Engine
    9. Output Manager
    """
    
    def __init__(self, config_dir: str = None):
        """
        Args:
            config_dir: Path to config directory
        """
        self.config_dir = config_dir or os.path.join(PROJECT_ROOT, "config")
        
        # Components (initialized in setup)
        self.components = None
        self.stages = {}
        
        # Tracking
        self.cost_tracker = None
        self.checkpoint_manager = None
        self.health_monitor = None
        
        # LINKED QUESTION CONTEXT CACHE (Fix for Q59-Q60 dependency)
        # Key: group_id, Value: List of processed question objects
        self.linked_context_cache = {}
        
        # Statistics
        self.stats = {
            'total_questions': 0,
            'processed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'failed': 0,
            'total_cost': 0.0,
            'total_time': 0.0,
            'start_time': None,
            'end_time': None
        }
    
    def setup(self):
        """
        Initialize all pipeline components (Stage 0)
        """
        logger.info("="*80)
        logger.info("GATE AE SOTA PIPELINE - SETUP")
        logger.info("="*80)
        
        # Initialize all services and models
        initializer = PipelineInitializer(self.config_dir)
        self.components = initializer.initialize_all()
        
        # Initialize cost tracker
        self.cost_tracker = CostTracker(self.components['configs']['models_config'])
        
        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=os.path.join(PROJECT_ROOT, "checkpoints")
        )
        
        # Initialize health monitor
        self.health_monitor = HealthMonitor(
            budget_limit=420.0,  # Total budget $420
            cost_tracker=self.cost_tracker
        )
        
        # Initialize all stage classes
        self._initialize_stages()
        
        logger.info("✓ Pipeline setup complete")
        logger.info("")
    
    def _initialize_stages(self):
        """Initialize all pipeline stage classes"""
        configs = self.components['configs']
        clients = self.components['clients']
        embedding_model = self.components['embedding_model']
        
        self.stages = {
            'question_loader': QuestionLoader(
                questions_dir=os.getenv('QUESTIONS_DIR', './data/raw'),
                images_dir=os.getenv('IMAGES_DIR', './data/raw_images')
            ),
            'classifier': QuestionClassifier(configs, clients),
            'cache_manager': CacheManager(
                redis_client=clients['redis'],
                embedding_model=embedding_model,
                configs=configs
            ),
            'dense_retriever': DenseRetriever(
                qdrant_client=clients['qdrant'],
                embedding_model=embedding_model,
                configs=configs
            ),
            'sparse_retriever': SparseRetriever(
                redis_client=clients['redis'],
                configs=configs
            ),
            'retrieval_merger': RetrievalMerger(configs),
            'image_consensus': ImageConsensusGenerator(configs, clients),
            'model_orchestrator': ModelOrchestrator(configs, clients),
            'voting_engine': VotingEngine(configs, clients),
            'debate_orchestrator': DebateOrchestrator(configs, clients),
            'synthesis_engine': SynthesisEngine(configs),
            'output_manager': OutputManager(
                redis_client=clients['redis'],
                s3_client=clients.get('s3'),
                dynamodb_client=clients.get('dynamodb'),
                embedding_model=embedding_model,
                configs=configs
            )
        }
    
    def run(
        self,
        question_files: List[str] = None,
        question_ids: List[str] = None,
        year: Optional[int] = None,
        start_index: int = 0,
        end_index: Optional[int] = None,
        resume: bool = False
    ) -> Dict[str, Any]:
        """
        Run pipeline on questions
        
        Args:
            question_files: List of question file paths
            question_ids: List of specific question IDs to process
            year: Filter by year
            start_index: Start from this index
            end_index: End at this index
            resume: Resume from last checkpoint
        
        Returns:
            dict: Pipeline statistics and results
        """
        self.stats['start_time'] = datetime.utcnow().isoformat()
        
        # Determine which questions to process
        questions_to_process = self._get_questions_to_process(
            question_files=question_files,
            question_ids=question_ids,
            year=year,
            start_index=start_index,
            end_index=end_index,
            resume=resume
        )
        
        self.stats['total_questions'] = len(questions_to_process)
        
        logger.info("="*80)
        logger.info(f"PROCESSING {len(questions_to_process)} QUESTIONS")
        logger.info("="*80)
        logger.info("")
        
        # Process each question
        for i, question_file in enumerate(questions_to_process, 1):
            try:
                logger.info(f"\n{'='*80}")
                logger.info(f"QUESTION {i}/{len(questions_to_process)}")
                logger.info(f"{'='*80}\n")
                
                # Process single question
                result = self._process_question(question_file)
                
                # Update statistics
                self.stats['processed'] += 1
                
                if result['cache_hit']:
                    self.stats['cache_hits'] += 1
                else:
                    self.stats['cache_misses'] += 1
                
                self.stats['total_cost'] += result['total_cost']
                self.stats['total_time'] += result['total_time']
                
                # Check budget
                if not self.health_monitor.check_budget(self.stats['total_cost']):
                    logger.error("Budget limit exceeded! Stopping pipeline.")
                    break
                
                # Save checkpoint every 10 questions
                if i % 10 == 0:
                    self.checkpoint_manager.save_checkpoint({
                        'last_processed_index': i,
                        'last_question_file': question_file,
                        'stats': self.stats
                    })
                    logger.info(f"\n✓ Checkpoint saved at question {i}")
                
                # Progress report
                self._log_progress(i, len(questions_to_process))
                
            except Exception as e:
                logger.error(f"Failed to process question {question_file}: {e}")
                self.stats['failed'] += 1
                
                # Continue with next question (don't stop entire pipeline)
                continue
        
        # Final statistics
        self.stats['end_time'] = datetime.utcnow().isoformat()
        self._log_final_statistics()
        
        return self.stats
    
    def _get_questions_to_process(
        self,
        question_files: List[str] = None,
        question_ids: List[str] = None,
        year: Optional[int] = None,
        start_index: int = 0,
        end_index: Optional[int] = None,
        resume: bool = False
    ) -> List[str]:
        """
        Determine which questions to process based on filters
        
        Returns:
            list: Question file paths to process
        """
        # If resuming, load checkpoint
        if resume:
            checkpoint = self.checkpoint_manager.load_checkpoint()
            if checkpoint:
                logger.info(f"Resuming from checkpoint: {checkpoint['last_question_file']}")
                start_index = checkpoint['last_processed_index']
        
        # Get all question files
        if question_files:
            all_files = question_files
        else:
            all_files = self.stages['question_loader'].get_all_questions_in_directory()
        
        # Filter by specific question IDs
        if question_ids:
            filtered = []
            for f in all_files:
                # Handle both individual files (str) and array-based files (tuple)
                if isinstance(f, tuple):
                    # Tuple format: (file_path, question_id, index)
                    _, qid, _ = f
                    if any(search_qid in qid for search_qid in question_ids):
                        filtered.append(f)
                else:
                    # String format: check if qid is in filename
                    if any(qid in str(f) for qid in question_ids):
                        filtered.append(f)
            all_files = filtered
        
        # Filter by year
        if year:
            # Load each question to check year (or parse from filename)
            year_filtered = []
            for qfile in all_files:
                try:
                    q = self.stages['question_loader'].load_question(qfile)
                    if q['year'] == year:
                        year_filtered.append(qfile)
                except:
                    continue
            all_files = year_filtered
        
        # Apply index range
        if end_index is not None:
            all_files = all_files[start_index:end_index]
        else:
            all_files = all_files[start_index:]
        
        return all_files
    
    def _process_question(self, question_file: str) -> Dict[str, Any]:
        """
        Process a single question through all 9 stages
        
        Args:
            question_file: Path to question JSON file
        
        Returns:
            dict: {
                "question_id": str,
                "cache_hit": bool,
                "total_cost": float,
                "total_time": float,
                "stage_timings": dict,
                "output_paths": dict
            }
        """
        stage_timings = {}
        total_start = time.time()
        
        # STAGE 1: Load Question
        stage_start = time.time()
        question = self.stages['question_loader'].load_question(question_file)
        stage_timings['stage_1'] = time.time() - stage_start
        
        question_id = question['question_id']
        logger.info(f"Question ID: {question_id}")
        logger.info(f"Year: {question['year']}, Type: {question['question_type']}, Marks: {question['marks']}")
        
        # STAGE 2: Classification
        stage_start = time.time()
        classification = self.stages['classifier'].classify_question(question)
        stage_timings['stage_2'] = time.time() - stage_start
        
        # STAGE 3: Cache Check
        stage_start = time.time()
        cached_result = self.stages['cache_manager'].check_cache(question)
        stage_timings['stage_3'] = time.time() - stage_start
        
        if cached_result:
            # Cache hit - skip to Stage 9
            logger.info("✓ CACHE HIT - Using cached result")
            
            # Still save to ensure all destinations have it
            stage_start = time.time()
            output_paths = self.stages['output_manager'].save_output(
                final_json=cached_result,
                question=question,
                save_to_redis=False  # Already in cache
            )
            stage_timings['stage_9'] = time.time() - stage_start
            
            return {
                "question_id": question_id,
                "cache_hit": True,
                "total_cost": 0.0,  # No API calls
                "total_time": time.time() - total_start,
                "stage_timings": stage_timings,
                "output_paths": output_paths
            }
        
        # STAGE 4: RAG Retrieval
        stage_start = time.time()
        
        # 4A: Dense retrieval
        dense_results = self.stages['dense_retriever'].retrieve(question)
        
        # 4B: Sparse retrieval
        sparse_results = self.stages['sparse_retriever'].retrieve(question)
        
        # 4C: Merge
        rag_chunks = self.stages['retrieval_merger'].merge(
            dense_results=dense_results,
            sparse_results=sparse_results,
            question=question
        )
        
        stage_timings['stage_4'] = time.time() - stage_start
        
        # STAGE 4.5: Image Consensus (if has_image)
        image_consensus = None
        if question.get('has_question_image'):
            stage_start = time.time()
            image_consensus = self.stages['image_consensus'].generate_consensus(question)
            stage_timings['stage_4_5'] = time.time() - stage_start
        
        # STAGE 5: Model Orchestration
        stage_start = time.time()
        
        # Get Linked Context if available
        previous_context = []
        group_id = None
        if 'linked_question_group' in question and question['linked_question_group']:
            link_info = question['linked_question_group']
            if isinstance(link_info, dict):
                group_id = link_info.get('group_id')
            else:
                group_id = str(link_info)
            
            if group_id:
                previous_context = self.linked_context_cache.get(group_id, [])
                if previous_context:
                    logger.info(f"  🔗 Injecting context from {len(previous_context)} queries in group {group_id}")

        generation_result = self.stages['model_orchestrator'].generate_responses(
            question=question,
            classification=classification,
            rag_chunks=rag_chunks,
            image_consensus=image_consensus,
            previous_context=previous_context
        )
        stage_timings['stage_5'] = time.time() - stage_start
        
        model_responses = generation_result['model_responses']
        weights = generation_result['weights']
        generation_metadata = generation_result['metadata']
        
        # Save individual model responses (new request)
        if 'output_manager' in self.stages:
            self.stages['output_manager'].save_individual_responses(
                question=question,
                model_responses=model_responses
            )
        
        # STAGE 6: Voting
        stage_start = time.time()
        voting_result = self.stages['voting_engine'].vote_on_responses(
            model_responses=model_responses,
            weights=weights,
            question=question
        )
        stage_timings['stage_6'] = time.time() - stage_start
        
        # STAGE 7: Debate (if disputes exist)
        debate_result = {
            'resolved_fields': {},
            'still_disputed': [],
            'debate_rounds': 0,
            'debate_history': {},
            'gpt51_added': False,
            'debate_cost': 0.0
        }
        
        if voting_result['disputed_fields']:
            stage_start = time.time()
            debate_result = self.stages['debate_orchestrator'].resolve_disputes(
                disputed_fields=voting_result['disputed_fields'],
                field_details=voting_result.get('field_details', {}),
                model_responses=model_responses,
                weights=weights,
                rag_chunks=rag_chunks,
                question=question,
                classification=classification
            )
            stage_timings['stage_7'] = time.time() - stage_start
        
        # Calculate total cost
        total_cost = generation_metadata['total_cost'] + debate_result['debate_cost']
        
        # STAGE 8: Synthesis
        stage_start = time.time()
        final_json = self.stages['synthesis_engine'].synthesize(
            question=question,
            classification=classification,
            voting_result=voting_result,
            debate_result=debate_result,
            model_responses=model_responses,
            rag_chunks=rag_chunks,
            stage_timings=stage_timings,
            total_cost=total_cost
        )
        stage_timings['stage_8'] = time.time() - stage_start
        
        # STAGE 9: Output Manager
        stage_start = time.time()
        output_paths = self.stages['output_manager'].save_output(
            final_json=final_json,
            question=question
        )
        stage_timings['stage_9'] = time.time() - stage_start
        
        total_time = time.time() - total_start
        
        logger.info(f"\n✓ Question processed in {total_time:.1f}s")
        logger.info(f"  Cost: ${total_cost:.4f}")
        logger.info(f"  Quality: {final_json['tier_4_metadata_and_future']['quality_score']['band']}")
        
        # Update Linked Context Cache
        if group_id:
            if group_id not in self.linked_context_cache:
                self.linked_context_cache[group_id] = []
            # Store COMBINED object: raw question (for image bytes, options) + final_json (for derived answers)
            # This is critical for Q49 to access Q48's image via image_metadata.base64
            self.linked_context_cache[group_id].append({
                'raw_question': question,
                'final_json': final_json
            })
            
        return {
            "question_id": question_id,
            "cache_hit": False,
            "total_cost": total_cost,
            "total_time": total_time,
            "stage_timings": stage_timings,
            "output_paths": output_paths
        }
    
    def _log_progress(self, current: int, total: int):
        """Log progress statistics"""
        progress_pct = (current / total) * 100
        
        logger.info(f"\n{'='*80}")
        logger.info(f"PROGRESS: {current}/{total} ({progress_pct:.1f}%)")
        logger.info(f"{'='*80}")
        logger.info(f"Processed: {self.stats['processed']}")
        logger.info(f"Cache hits: {self.stats['cache_hits']}")
        logger.info(f"Cache misses: {self.stats['cache_misses']}")
        logger.info(f"Failed: {self.stats['failed']}")
        logger.info(f"Total cost: ${self.stats['total_cost']:.2f}")
        logger.info(f"Avg time/question: {self.stats['total_time']/max(current,1):.1f}s")
        
        # Estimate remaining
        if current > 0:
            avg_time = self.stats['total_time'] / current
            remaining_time = avg_time * (total - current)
            logger.info(f"Estimated remaining time: {remaining_time/60:.1f} minutes")
        
        logger.info(f"{'='*80}\n")
    
    def _log_final_statistics(self):
        """Log final pipeline statistics"""
        logger.info("\n" + "="*80)
        logger.info("PIPELINE COMPLETE - FINAL STATISTICS")
        logger.info("="*80)
        logger.info(f"Total questions: {self.stats['total_questions']}")
        logger.info(f"Processed: {self.stats['processed']}")
        logger.info(f"Cache hits: {self.stats['cache_hits']} ({self.stats['cache_hits']/max(self.stats['processed'],1)*100:.1f}%)")
        logger.info(f"Cache misses: {self.stats['cache_misses']}")
        logger.info(f"Failed: {self.stats['failed']}")
        logger.info(f"")
        logger.info(f"Total cost: ${self.stats['total_cost']:.2f}")
        logger.info(f"Avg cost/question: ${self.stats['total_cost']/max(self.stats['processed'],1):.4f}")
        logger.info(f"")
        logger.info(f"Total time: {self.stats['total_time']/60:.1f} minutes")
        logger.info(f"Avg time/question: {self.stats['total_time']/max(self.stats['processed'],1):.1f}s")
        logger.info(f"")
        logger.info(f"Start time: {self.stats['start_time']}")
        logger.info(f"End time: {self.stats['end_time']}")
        logger.info("="*80 + "\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="GATE AE Question Tagging Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Execution mode
    parser.add_argument('--question-id', nargs='+', help="Process specific question(s) by ID (can pass multiple)")
    parser.add_argument('--year', type=int, help="Process all questions from specific year")
    parser.add_argument('--start-index', type=int, default=0, help="Start from this index")
    parser.add_argument('--end-index', type=int, help="End at this index")
    parser.add_argument('--all', action='store_true', help="Process all questions")
    parser.add_argument('--resume', action='store_true', help="Resume from last checkpoint")
    
    # Configuration
    parser.add_argument('--config-dir', help="Path to config directory")
    
    args = parser.parse_args()
    
    # Create pipeline runner
    runner = PipelineRunner(config_dir=args.config_dir)
    
    # Setup
    runner.setup()
    
    # Run pipeline
    if args.question_id:
        # Multiple questions
        runner.run(question_ids=args.question_id)
    
    elif args.year:
        # Year filter (also respects start/end index if provided)
        runner.run(
            year=args.year,
            start_index=args.start_index,
            end_index=args.end_index,
            resume=args.resume
        )
    
    elif args.all:
        # All questions
        runner.run()
    
    else:
        # Range
        runner.run(
            start_index=args.start_index,
            end_index=args.end_index,
            resume=args.resume
        )


if __name__ == "__main__":
    main()

'''# 1. Single question
python 99_pipeline_runner.py --question-id GATE_AE_2024_Q15

# 2. All questions from 2024
python 99_pipeline_runner.py --year 2024

# 3. First 100 questions
python 99_pipeline_runner.py --start-index 0 --end-index 100

# 4. All 1,300 questions
python 99_pipeline_runner.py --all

# 5. Resume from crash
python 99_pipeline_runner.py --resume

# 6. Range with resume
python 99_pipeline_runner.py --start-index 100 --end-index 200 --resume
'''