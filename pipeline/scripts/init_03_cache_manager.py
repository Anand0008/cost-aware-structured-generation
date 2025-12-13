"""
============================================================================
STAGE 3: CACHE MANAGER
============================================================================
Purpose: Check Redis cache for similar questions (97%+ similarity)
Used by: 99_pipeline_runner.py
Author: GATE AE SOTA Pipeline
============================================================================
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from utils.logging_utils import setup_logger, log_stage
from utils.embedding_utils import generate_embedding, cosine_similarity

logger = setup_logger("03_cache_manager")


class CacheManager:
    """
    Manage Redis cache for question results
    """
    
    def __init__(self, redis_client, embedding_model, configs: Dict):
        """
        Args:
            redis_client: Initialized Redis client
            embedding_model: BGE-M3 embedding model
            configs: All configuration dictionaries
        """
        self.redis = redis_client
        self.embedding_model = embedding_model
        self.configs = configs
        
        # Get similarity threshold from config
        self.similarity_threshold = configs['thresholds_config']['cache']['similarity_threshold']
        
        # Cache TTL (time to live) in seconds
        self.cache_ttl = configs['thresholds_config']['cache'].get('ttl_days', 30) * 86400
        
        # Cache key prefix
        self.cache_prefix = "gate_ae_question:"
        self.index_key = "gate_ae_question_index"
    
    @log_stage("Stage 3: Cache Check")
    def check_cache(self, question: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Check if similar question exists in cache
        
        Args:
            question: Question object from Stage 1
        
        Returns:
            dict: Cached result if found (similarity >= threshold)
            None: If no similar question found
        """
        question_id = question.get('question_id', 'unknown')
        logger.info(f"Checking cache for: {question_id}")
        
        # Generate embedding for this question
        question_embedding = self._get_question_embedding(question)
        
        # Search cache for similar questions
        cached_result = self._search_similar_questions(
            question_id=question_id,
            question_embedding=question_embedding
        )
        
        if cached_result:
            logger.info(f"  ✓ CACHE HIT: Found similar question with {cached_result['similarity']:.1%} similarity")
            logger.info(f"    Original question: {cached_result['original_question_id']}")
            return cached_result['data']
        else:
            logger.info("  ✗ CACHE MISS: No similar question found")
            return None
    
    def save_to_cache(self, question: Dict[str, Any], result: Dict[str, Any]):
        """
        Save question result to cache
        
        Args:
            question: Question object
            result: Complete pipeline result (tier_0 through tier_4)
        """
        question_id = question.get('question_id', 'unknown')
        logger.info(f"Saving to cache: {question_id}")
        
        try:
            # Generate embedding
            question_embedding = self._get_question_embedding(question)
            
            # Create cache entry
            cache_entry = {
                "question_id": question_id,
                "question_text": question['question_text'][:200],  # Truncate for storage
                "year": question.get('year'),
                "embedding": question_embedding.tolist(),  # Convert numpy to list
                "data": result,
                "cached_at": self._get_timestamp()
            }
            
            # Generate cache key
            cache_key = self._generate_cache_key(question_id)
            
            # Save to Redis
            self.redis.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(cache_entry)
            )
            
            # Add to index
            self.redis.sadd(self.index_key, cache_key)
            
            logger.info(f"  ✓ Saved to cache (TTL: {self.cache_ttl//86400} days)")
            
        except Exception as e:
            logger.error(f"Failed to save to cache: {e}")
            # Don't raise - caching failure shouldn't stop pipeline
    
    def _get_question_embedding(self, question: Dict) -> np.ndarray:
        """
        Generate embedding for question
        
        Uses question_text as the primary text for embedding
        """
        text = question['question_text']
        
        # Add options if MCQ (helps with similarity matching)
        if question.get('options'):
            options_text = " ".join([
                f"{k}: {v}" for k, v in question['options'].items()
            ])
            text = f"{text} {options_text}"
        
        embedding = generate_embedding(
            text=text,
            model=self.embedding_model
        )
        
        return embedding
    
    def _search_similar_questions(
        self, 
        question_id: str, 
        question_embedding: np.ndarray
    ) -> Optional[Dict]:
        """
        Search cache for questions with similarity >= threshold
        
        Args:
            question_id: Current question ID
            question_embedding: Current question embedding
        
        Returns:
            dict: {original_question_id, similarity, data} if found
            None: If no similar question found
        """
        try:
            # Get all cached question keys
            cached_keys = self.redis.smembers(self.index_key)
            
            if not cached_keys:
                logger.info("  Cache is empty")
                return None
            
            logger.info(f"  Searching through {len(cached_keys)} cached questions...")
            
            best_match = None
            best_similarity = 0.0
            
            for cache_key in cached_keys:
                try:
                    # Load cached entry
                    cached_data = self.redis.get(cache_key)
                    if not cached_data:
                        continue
                    
                    cached_entry = json.loads(cached_data)
                    
                    # Skip if same question
                    if cached_entry['question_id'] == question_id:
                        continue
                    
                    # Calculate similarity
                    cached_embedding = np.array(cached_entry['embedding'])
                    similarity = cosine_similarity(question_embedding, cached_embedding)
                    
                    # Track best match
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = {
                            'original_question_id': cached_entry['question_id'],
                            'similarity': similarity,
                            'data': cached_entry['data']
                        }
                    
                except Exception as e:
                    logger.warning(f"Error processing cache entry {cache_key}: {e}")
                    continue
            
            # Check if best match exceeds threshold
            if best_match and best_similarity >= self.similarity_threshold:
                return best_match
            else:
                if best_match:
                    logger.info(f"  Best match: {best_similarity:.1%} (below {self.similarity_threshold:.1%} threshold)")
                return None
            
        except Exception as e:
            logger.error(f"Cache search failed: {e}")
            return None
    
    def _generate_cache_key(self, question_id: str) -> str:
        """
        Generate Redis key for question
        
        Format: gate_ae_question:{hash}
        """
        # Use hash to avoid special characters in Redis key
        question_hash = hashlib.md5(question_id.encode()).hexdigest()
        return f"{self.cache_prefix}{question_hash}"
    
    def _get_timestamp(self) -> str:
        """Get current ISO timestamp"""
        from datetime import datetime
        return datetime.utcnow().isoformat()
    
    def get_cache_stats(self) -> Dict:
        """
        Get cache statistics
        
        Returns:
            dict: {total_entries, size_mb, oldest_entry, newest_entry}
        """
        try:
            cached_keys = self.redis.smembers(self.index_key)
            
            if not cached_keys:
                return {
                    "total_entries": 0,
                    "size_mb": 0,
                    "oldest_entry": None,
                    "newest_entry": None
                }
            
            timestamps = []
            total_size = 0
            
            for cache_key in cached_keys:
                cached_data = self.redis.get(cache_key)
                if cached_data:
                    total_size += len(cached_data)
                    try:
                        entry = json.loads(cached_data)
                        timestamps.append(entry.get('cached_at'))
                    except:
                        pass
            
            return {
                "total_entries": len(cached_keys),
                "size_mb": round(total_size / (1024 * 1024), 2),
                "oldest_entry": min(timestamps) if timestamps else None,
                "newest_entry": max(timestamps) if timestamps else None
            }
            
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"error": str(e)}
    
    def clear_cache(self, confirm: bool = False):
        """
        Clear all cached questions
        
        Args:
            confirm: Must be True to actually clear cache (safety)
        """
        if not confirm:
            logger.warning("Clear cache called without confirmation - skipping")
            return
        
        try:
            cached_keys = self.redis.smembers(self.index_key)
            
            if not cached_keys:
                logger.info("Cache is already empty")
                return
            
            logger.info(f"Clearing {len(cached_keys)} cached questions...")
            
            # Delete all cache entries
            for cache_key in cached_keys:
                self.redis.delete(cache_key)
            
            # Clear index
            self.redis.delete(self.index_key)
            
            logger.info("✓ Cache cleared")
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            raise
    
    def invalidate_question(self, question_id: str):
        """
        Remove a specific question from cache
        
        Args:
            question_id: Question ID to invalidate
        """
        try:
            cache_key = self._generate_cache_key(question_id)
            
            # Delete from Redis
            self.redis.delete(cache_key)
            
            # Remove from index
            self.redis.srem(self.index_key, cache_key)
            
            logger.info(f"✓ Invalidated cache for: {question_id}")
            
        except Exception as e:
            logger.error(f"Failed to invalidate cache for {question_id}: {e}")


def main():
    """
    Test cache manager
    """
    import argparse
    from scripts.initialization import PipelineInitializer
    from scripts.question_loader import QuestionLoader
    
    parser = argparse.ArgumentParser(description="Test Cache Manager")
    parser.add_argument("--action", choices=["check", "save", "stats", "clear"], required=True)
    parser.add_argument("--question-json", help="Path to question JSON file")
    parser.add_argument("--result-json", help="Path to result JSON file (for save action)")
    
    args = parser.parse_args()
    
    # Initialize
    print("\n" + "="*80)
    print("INITIALIZING")
    print("="*80 + "\n")
    
    initializer = PipelineInitializer()
    components = initializer.initialize_all()
    
    cache_manager = CacheManager(
        redis_client=components['clients']['redis'],
        embedding_model=components['embedding_model'],
        configs=components['configs']
    )
    
    if args.action == "stats":
        stats = cache_manager.get_cache_stats()
        print("\n" + "="*80)
        print("CACHE STATISTICS")
        print("="*80)
        print(json.dumps(stats, indent=2))
    
    elif args.action == "clear":
        confirm = input("Are you sure you want to clear the cache? (yes/no): ")
        if confirm.lower() == "yes":
            cache_manager.clear_cache(confirm=True)
        else:
            print("Cache clear cancelled")
    
    elif args.action == "check":
        if not args.question_json:
            print("Error: --question-json required for check action")
            return
        
        loader = QuestionLoader(os.path.dirname(args.question_json))
        question = loader.load_question(args.question_json)
        
        cached_result = cache_manager.check_cache(question)
        
        if cached_result:
            print("\n✓ CACHE HIT")
            print(json.dumps(cached_result, indent=2)[:500] + "...")
        else:
            print("\n✗ CACHE MISS")
    
    elif args.action == "save":
        if not args.question_json or not args.result_json:
            print("Error: --question-json and --result-json required for save action")
            return
        
        loader = QuestionLoader(os.path.dirname(args.question_json))
        question = loader.load_question(args.question_json)
        
        with open(args.result_json, 'r') as f:
            result = json.load(f)
        
        cache_manager.save_to_cache(question, result)
        print("\n✓ Saved to cache")


if __name__ == "__main__":
    main()