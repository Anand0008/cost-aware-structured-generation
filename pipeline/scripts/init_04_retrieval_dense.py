"""
============================================================================
STAGE 4A: DENSE RETRIEVAL (Qdrant Vector Search)
============================================================================
Purpose: Retrieve top 10 most semantically similar chunks from vector database
Used by: 99_pipeline_runner.py (Stage 4)
Author: GATE AE SOTA Pipeline
============================================================================
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import sys

# Add project root to path
current_file = Path(__file__).resolve()
PROJECT_ROOT = current_file.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Robust imports
try:
    from pipeline.utils.logging_utils import setup_logger, log_stage
    from pipeline.utils.embedding_utils import generate_embedding
except ImportError:
    # Fallback
    sys.path.append(str(current_file.parent.parent))
    from utils.logging_utils import setup_logger, log_stage
    from utils.embedding_utils import generate_embedding

logger = setup_logger("04_retrieval_dense")


class DenseRetriever:
    """
    Retrieve semantically similar chunks using Qdrant vector search
    """
    
    def __init__(self, qdrant_client, embedding_model, configs: Dict):
        """
        Args:
            qdrant_client: Initialized Qdrant client
            embedding_model: BGE-M3 embedding model
            configs: All configuration dictionaries
        """
        self.qdrant = qdrant_client
        self.embedding_model = embedding_model
        self.configs = configs
        
        # Get collection names from config, with safe defaults matching indexing script
        qdrant_config = configs.get('models_config', {}).get('qdrant', {})
        collections_config = qdrant_config.get('collections', {})
        
        self.books_collection = collections_config.get('books', 'qbt_books')
        self.videos_collection = collections_config.get('videos', 'qbt_videos')
        
        # Number of results to retrieve
        self.top_k = 10
        
        logger.info(f"Initialized DenseRetriever with collections: Books='{self.books_collection}', Videos='{self.videos_collection}'")
    
    @log_stage("Stage 4A: Dense Retrieval")
    def retrieve(self, question: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Retrieve top 10 semantically similar chunks
        
        Args:
            question: Question object with question_text
        
        Returns:
            list: Top 10 chunks with metadata and scores
        """
        question_id = question.get('question_id', 'unknown')
        logger.info(f"Dense retrieval for: {question_id}")
        
        # Generate query embedding
        query_embedding = self._generate_query_embedding(question)
        
        # Search books collection
        book_results = self._search_collection(
            collection_name=self.books_collection,
            query_embedding=query_embedding,
            top_k=self.top_k
        )
        
        # Search videos collection
        video_results = self._search_collection(
            collection_name=self.videos_collection,
            query_embedding=query_embedding,
            top_k=self.top_k
        )
        
        # Combine and sort by score
        all_results = book_results + video_results
        all_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Take top 10
        top_results = all_results[:self.top_k]
        
        logger.info(f"  ✓ Retrieved {len(top_results)} chunks")
        logger.info(f"    - Books: {sum(1 for r in top_results if 'book' in r.get('source_type', ''))}")
        logger.info(f"    - Videos: {sum(1 for r in top_results if 'video' in r.get('source_type', ''))}")
        if top_results:
            logger.info(f"    - Avg score: {np.mean([r['score'] for r in top_results]):.3f}")
        
        return top_results
    
    def _generate_query_embedding(self, question: Dict) -> np.ndarray:
        """
        Generate embedding for question
        
        Uses question_text as primary query
        Optionally includes options for context
        """
        query_text = question['question_text']
        
        # For MCQ, include options to improve retrieval
        if question.get('options'):
            # Handle various option formats
            options = question['options']
            if isinstance(options, dict):
                options_text = " ".join([f"{k}: {v}" for k, v in options.items()])
            elif isinstance(options, list):
                options_text = " ".join([str(o) for o in options])
            else:
                options_text = str(options)
            
            query_text = f"{query_text} {options_text}"
        
        logger.info(f"  Generating query embedding (length: {len(query_text)} chars)...")
        
        embedding = generate_embedding(
            text=query_text,
            model=self.embedding_model
        )
        
        return embedding
    
    def _search_collection(
        self,
        collection_name: str,
        query_embedding: np.ndarray,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Search a Qdrant collection
        
        Args:
            collection_name: Name of Qdrant collection
            query_embedding: Query vector
            top_k: Number of results to return
        
        Returns:
            list: Search results with metadata
        """
        try:
            logger.info(f"  Searching {collection_name}...")
            
            # Verify collection exists first
            try:
                collection_info = self.qdrant.get_collection(collection_name)
                logger.debug(f"    Collection '{collection_name}' exists with {collection_info.points_count} vectors")
            except Exception as check_error:
                if "Not found" in str(check_error) or "does not exist" in str(check_error).lower():
                    logger.error(f"    Collection '{collection_name}' does not exist in Qdrant")
                    logger.error(f"    Available collections: Check Qdrant dashboard or use get_collections()")
                    return []
                else:
                    logger.warning(f"    Could not verify collection existence: {check_error}")
            
            # Search Qdrant - Use correct API based on client version
            try:
                # Try query_points (v1.7+ API)
                from qdrant_client.models import Query, Filter
                query_result = self.qdrant.query_points(
                    collection_name=collection_name,
                    query=query_embedding.tolist(),
                    limit=top_k,
                    with_payload=True,
                    score_threshold=0.5  # Lowered from 0.7 to get more results
                )
                search_results = query_result.points
            except (AttributeError, TypeError) as e:
                # Fallback to query() method (older/newer alternative)
                try:
                    from qdrant_client.models import Query, Filter
                    query_result = self.qdrant.query(
                        collection_name=collection_name,
                        query_vector=query_embedding.tolist(),
                        limit=top_k,
                        with_payload=True,
                        score_threshold=0.7
                    )
                    search_results = query_result if hasattr(query_result, '__iter__') else []
                except (AttributeError, TypeError) as e2:
                    # Final fallback - check collection exists first
                    try:
                        collection_info = self.qdrant.get_collection(collection_name)
                        logger.error(f"Qdrant API method not found. Collection exists with {collection_info.points_count} vectors.")
                        logger.error(f"Available methods: {[m for m in dir(self.qdrant) if not m.startswith('_')]}")
                        return []
                    except Exception as e3:
                        if "Not found" in str(e3) or "does not exist" in str(e3).lower():
                            logger.error(f"Collection '{collection_name}' does not exist in Qdrant.")
                            return []
                        raise e3
            
            # Format results
            formatted_results = []
            for result in search_results:
                chunk = {
                    'chunk_id': result.id,
                    'score': result.score,
                    'source_type': result.payload.get('source_type', 'unknown'),
                    'source_name': result.payload.get('source_name') or result.payload.get('book') or result.payload.get('video_source') or 'Unknown',
                    'text': result.payload.get('text', ''),
                    'metadata': self._extract_metadata(result.payload, collection_name)
                }
                formatted_results.append(chunk)
            
            logger.info(f"    Found {len(formatted_results)} results from {collection_name}")
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Failed to search {collection_name}: {e}")
            return []
    
    def _extract_metadata(self, payload: Dict, collection_name: str) -> Dict:
        """
        Extract metadata from Qdrant payload
        """
        # Common metadata
        base_meta = {
            'chunk_id': payload.get('chunk_id'),
            'subject': payload.get('subject'),
            'concept_name': payload.get('concept_name'),
            'prerequisites': payload.get('prerequisites', [])
        }

        if collection_name == self.books_collection or payload.get('source_type') == 'book_note':
            base_meta.update({
                'author': payload.get('author'),
                'book': payload.get('book') or payload.get('book_name'),
                'chapter_number': payload.get('chapter_number'),
                'chapter_title': payload.get('chapter_title'),
                'section': payload.get('section') or payload.get('section_number'),
                'page_range': payload.get('page_range'),
                'complexity': payload.get('complexity', 'intermediate')
            })
            base_meta['source_type'] = 'book'
        
        elif collection_name == self.videos_collection or payload.get('source_type') == 'video_note':
            base_meta.update({
                'professor': payload.get('professor'),
                'video_url': payload.get('video_url') or payload.get('url'),
                'timestamp_start': payload.get('timestamp_start'),
                'timestamp_end': payload.get('timestamp_end'),
                'topic_covered': payload.get('topic_covered') or payload.get('concept_name'),
                'book_reference': payload.get('book_reference')
            })
            base_meta['source_type'] = 'video'
        
        return base_meta
    
    def test_connection(self) -> bool:
        """
        Test Qdrant connection and check collections exist
        
        Returns:
            bool: True if both collections exist and are accessible
        """
        try:
            collections = self.qdrant.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            books_exists = self.books_collection in collection_names
            videos_exists = self.videos_collection in collection_names
            
            if books_exists and videos_exists:
                # Get collection stats
                books_info = self.qdrant.get_collection(self.books_collection)
                videos_info = self.qdrant.get_collection(self.videos_collection)
                
                logger.info(f"✓ Qdrant collections found:")
                logger.info(f"  - {self.books_collection}: {books_info.points_count} vectors")
                logger.info(f"  - {self.videos_collection}: {videos_info.points_count} vectors")
                
                return True
            else:
                missing = []
                if not books_exists:
                    missing.append(self.books_collection)
                if not videos_exists:
                    missing.append(self.videos_collection)
                
                logger.error(f"Missing Qdrant collections: {missing}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            return False
    
    def get_collection_stats(self) -> Dict:
        """
        Get statistics about Qdrant collections
        
        Returns:
            dict: {books: {...}, videos: {...}}
        """
        try:
            stats = {}
            
            for collection_name in [self.books_collection, self.videos_collection]:
                try:
                    info = self.qdrant.get_collection(collection_name)
                    stats[collection_name] = {
                        'points_count': info.points_count,
                        'points_count': info.points_count,
                        'segments_count': info.segments_count,
                        'status': info.status
                    }
                except Exception as e:
                    stats[collection_name] = {'error': str(e)}
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {'error': str(e)}


def main():
    """
    Test dense retrieval
    """
    import argparse
    # Dynamic imports for main execution
    try:
        from pipeline.scripts.init_00_initialization import PipelineInitializer
        from pipeline.scripts.init_01_question_loader import QuestionLoader
    except ImportError:
         # Local testing fallbacks if needed
         pass
    
    parser = argparse.ArgumentParser(description="Test Dense Retrieval")
    parser.add_argument("--question-json", help="Path to question JSON file")
    parser.add_argument("--stats", action="store_true", help="Show collection statistics")
    
    args = parser.parse_args()
    
    # Initialize
    print("\n" + "="*80)
    print("INITIALIZING")
    print("="*80 + "\n")
    
    initializer = PipelineInitializer()
    components = initializer.initialize_all()
    
    retriever = DenseRetriever(
        qdrant_client=components['clients']['qdrant'],
        embedding_model=components['embedding_model'],
        configs=components['configs']
    )
    
    if args.stats:
        stats = retriever.get_collection_stats()
        print("\n" + "="*80)
        print("COLLECTION STATISTICS")
        print("="*80)
        print(json.dumps(stats, indent=2))
        return
    
    if not args.question_json:
        print("Error: --question-json required (or use --stats)")
        return
    
    # Load question
    print("\n" + "="*80)
    print("LOADING QUESTION")
    print("="*80 + "\n")
    
    loader = QuestionLoader(os.path.dirname(args.question_json))
    question = loader.load_question(args.question_json)
    
    # Retrieve
    print("\n" + "="*80)
    print("DENSE RETRIEVAL")
    print("="*80 + "\n")
    
    results = retriever.retrieve(question)
    
    # Print results
    print("\n" + "="*80)
    print(f"RETRIEVED {len(results)} CHUNKS")
    print("="*80 + "\n")
    
    for i, chunk in enumerate(results, 1):
        print(f"\n--- CHUNK {i} (Score: {chunk['score']:.3f}) ---")
        print(f"Source: {chunk['source_type']} - {chunk['source_name']}")
        print(f"Metadata: {json.dumps(chunk['metadata'], indent=2)}")
        print(f"Text: {chunk['text'][:200]}...")


if __name__ == "__main__":
    main()