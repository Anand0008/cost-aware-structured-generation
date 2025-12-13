"""
============================================================================
STAGE 4C: RETRIEVAL MERGER (Reciprocal Rank Fusion)
============================================================================
Purpose: Merge dense (vector) and sparse (BM25) retrieval results into top 6 chunks
Used by: 99_pipeline_runner.py (Stage 4)
Algorithm: Reciprocal Rank Fusion (RRF)
Author: GATE AE SOTA Pipeline
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

logger = setup_logger("06_retrieval_merger")


class RetrievalMerger:
    """
    Merge dense and sparse retrieval results using Reciprocal Rank Fusion
    """
    
    def __init__(self, configs: Dict):
        """
        Args:
            configs: All configuration dictionaries
        """
        self.configs = configs
        
        # RRF parameter (k constant)
        self.rrf_k = 60  # Standard value
        
        # Final number of chunks to return
        self.final_top_k = 6
    
    @log_stage("Stage 4C: Retrieval Merger")
    def merge(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        question: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Merge dense and sparse results using Reciprocal Rank Fusion
        
        Args:
            dense_results: Results from dense retrieval (Qdrant vector search)
            sparse_results: Results from sparse retrieval (BM25)
            question: Question object (for logging)
        
        Returns:
            list: Top 6 merged chunks with hybrid scores
        """
        question_id = question.get('question_id', 'unknown')
        logger.info(f"Merging retrieval results for: {question_id}")
        logger.info(f"  Dense results: {len(dense_results)}")
        logger.info(f"  Sparse results: {len(sparse_results)}")
        
        # Calculate RRF scores
        merged_results = self._reciprocal_rank_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results
        )
        
        # Sort by hybrid score
        merged_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        # Take top 6
        top_chunks = merged_results[:self.final_top_k]
        
        # Add rank and format
        for i, chunk in enumerate(top_chunks, 1):
            chunk['rank'] = i
            chunk = self._add_relevance_metadata(chunk)
        
        # Log statistics
        self._log_merge_stats(top_chunks, dense_results, sparse_results)
        
        logger.info(f"  ✓ Merged to {len(top_chunks)} final chunks")
        
        return top_chunks
    
    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Dict],
        sparse_results: List[Dict]
    ) -> List[Dict]:
        """
        Apply Reciprocal Rank Fusion algorithm
        
        RRF Formula:
            RRF_score(d) = Σ (1 / (k + rank_i(d)))
        
        Where:
            - d: document
            - rank_i(d): rank of document d in retrieval system i
            - k: constant (typically 60)
        
        Args:
            dense_results: Dense retrieval results
            sparse_results: Sparse retrieval results
        
        Returns:
            list: All unique chunks with hybrid scores
        """
        # Create dictionaries for fast lookup by chunk_id
        dense_ranks = {
            result['chunk_id']: rank 
            for rank, result in enumerate(dense_results, 1)
        }
        
        sparse_ranks = {
            result['chunk_id']: rank 
            for rank, result in enumerate(sparse_results, 1)
        }
        
        # Get all unique chunk IDs
        all_chunk_ids = set(dense_ranks.keys()) | set(sparse_ranks.keys())
        
        logger.info(f"  Total unique chunks: {len(all_chunk_ids)}")
        
        # Calculate RRF score for each chunk
        merged = []
        
        for chunk_id in all_chunk_ids:
            # RRF score = 1/(k + dense_rank) + 1/(k + sparse_rank)
            rrf_score = 0.0
            
            # Add dense contribution
            if chunk_id in dense_ranks:
                rrf_score += 1.0 / (self.rrf_k + dense_ranks[chunk_id])
            
            # Add sparse contribution
            if chunk_id in sparse_ranks:
                rrf_score += 1.0 / (self.rrf_k + sparse_ranks[chunk_id])
            
            # Get chunk data (prefer dense if in both)
            if chunk_id in dense_ranks:
                chunk_data = next(r for r in dense_results if r['chunk_id'] == chunk_id)
                source = "dense"
            else:
                chunk_data = next(r for r in sparse_results if r['chunk_id'] == chunk_id)
                source = "sparse"
            
            # Add to merged results
            merged.append({
                'chunk_id': chunk_id,
                'hybrid_score': rrf_score,
                'dense_rank': dense_ranks.get(chunk_id),
                'sparse_rank': sparse_ranks.get(chunk_id),
                'dense_score': chunk_data.get('score') if source == "dense" else None,
                'sparse_score': chunk_data.get('score') if source == "sparse" else None,
                'source_type': chunk_data['source_type'],
                'source_name': chunk_data['source_name'],
                'text': chunk_data['text'],
                'metadata': chunk_data.get('metadata', {}),
                'retrieval_source': self._determine_retrieval_source(
                    chunk_id, dense_ranks, sparse_ranks
                )
            })
        
        return merged
    
    def _determine_retrieval_source(
        self,
        chunk_id: str,
        dense_ranks: Dict,
        sparse_ranks: Dict
    ) -> str:
        """
        Determine if chunk came from dense, sparse, or both
        
        Returns:
            str: "both", "dense_only", "sparse_only"
        """
        in_dense = chunk_id in dense_ranks
        in_sparse = chunk_id in sparse_ranks
        
        if in_dense and in_sparse:
            return "both"
        elif in_dense:
            return "dense_only"
        else:
            return "sparse_only"
    
    def _add_relevance_metadata(self, chunk: Dict) -> Dict:
        """
        Add relevance metadata to chunk
        
        Calculates:
            - relevance_score: Normalized 0-1 score for consistency
            - reference: Formatted reference string
        """
        # Normalize hybrid score to 0-1 range (approximately)
        # RRF scores typically range from 0.01 to 0.03, so multiply by 30
        chunk['relevance_score'] = min(chunk['hybrid_score'] * 30, 1.0)
        
        # Create formatted reference string
        chunk['reference'] = self._format_reference(chunk)
        
        # Add text snippet (first 200 chars)
        if chunk.get('text') and len(chunk['text']) > 200:
            chunk['text_snippet'] = chunk['text'][:200] + "..."
        else:
            chunk['text_snippet'] = chunk.get('text', '')
        
        return chunk
    
    def _format_reference(self, chunk: Dict) -> str:
        """
        Format reference string for chunk
        
        Books: "Author - Book, Chapter X: Title, Section Y, pp. Z"
        Videos: "Professor - Video Title, timestamp X-Y"
        """
        metadata = chunk.get('metadata', {})
        source_type = chunk['source_type']
        
        if 'book' in source_type:
            author = metadata.get('author', 'Unknown')
            book = metadata.get('book', 'Unknown')
            chapter_num = metadata.get('chapter_number', '')
            chapter_title = metadata.get('chapter_title', '')
            section = metadata.get('section', '')
            pages = metadata.get('page_range', '')
            
            ref = f"{author} - {book}"
            
            if chapter_num:
                ref += f", Ch. {chapter_num}"
                if chapter_title:
                    ref += f": {chapter_title}"
            
            if section:
                ref += f", §{section}"
            
            if pages:
                ref += f", pp. {pages}"
            
            return ref
        
        elif 'video' in source_type:
            professor = metadata.get('professor', 'Unknown')
            video_name = chunk['source_name']
            timestamp_start = metadata.get('timestamp_start', '')
            timestamp_end = metadata.get('timestamp_end', '')
            
            ref = f"{professor} - {video_name}"
            
            if timestamp_start:
                ref += f", {timestamp_start}"
                if timestamp_end:
                    ref += f"-{timestamp_end}"
            
            return ref
        
        return chunk['source_name']
    
    def _log_merge_stats(
        self,
        top_chunks: List[Dict],
        dense_results: List[Dict],
        sparse_results: List[Dict]
    ):
        """Log statistics about the merge"""
        
        # Count sources
        both_count = sum(1 for c in top_chunks if c['retrieval_source'] == 'both')
        dense_only = sum(1 for c in top_chunks if c['retrieval_source'] == 'dense_only')
        sparse_only = sum(1 for c in top_chunks if c['retrieval_source'] == 'sparse_only')
        
        # Count source types
        books = sum(1 for c in top_chunks if 'book' in c.get('source_type', ''))
        videos = sum(1 for c in top_chunks if 'video' in c.get('source_type', ''))
        
        logger.info(f"  Merge statistics:")
        logger.info(f"    - From both: {both_count}")
        logger.info(f"    - Dense only: {dense_only}")
        logger.info(f"    - Sparse only: {sparse_only}")
        logger.info(f"    - Books: {books}, Videos: {videos}")
        
        if top_chunks:
            avg_score = sum(c['hybrid_score'] for c in top_chunks) / len(top_chunks)
            logger.info(f"    - Avg hybrid score: {avg_score:.4f}")


def main():
    """
    Test retrieval merger
    """
    import argparse
    from scripts.initialization import PipelineInitializer
    from scripts.question_loader import QuestionLoader
    from scripts.retrieval_dense import DenseRetriever
    from scripts.retrieval_sparse import SparseRetriever
    
    parser = argparse.ArgumentParser(description="Test Retrieval Merger")
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
    
    # Dense retrieval
    print("\n" + "="*80)
    print("DENSE RETRIEVAL")
    print("="*80 + "\n")
    
    dense_retriever = DenseRetriever(
        qdrant_client=components['clients']['qdrant'],
        embedding_model=components['embedding_model'],
        configs=components['configs']
    )
    dense_results = dense_retriever.retrieve(question)
    
    # Sparse retrieval
    print("\n" + "="*80)
    print("SPARSE RETRIEVAL")
    print("="*80 + "\n")
    
    sparse_retriever = SparseRetriever(
        redis_client=components['clients']['redis'],
        configs=components['configs']
    )
    sparse_results = sparse_retriever.retrieve(question)
    
    # Merge
    print("\n" + "="*80)
    print("MERGING RESULTS")
    print("="*80 + "\n")
    
    merger = RetrievalMerger(configs=components['configs'])
    final_chunks = merger.merge(dense_results, sparse_results, question)
    
    # Print results
    print("\n" + "="*80)
    print(f"FINAL {len(final_chunks)} CHUNKS")
    print("="*80 + "\n")
    
    for chunk in final_chunks:
        print(f"\n--- RANK {chunk['rank']} (Hybrid Score: {chunk['hybrid_score']:.4f}) ---")
        print(f"Source: {chunk['retrieval_source']}")
        print(f"  Dense rank: {chunk['dense_rank']}, Sparse rank: {chunk['sparse_rank']}")
        print(f"Type: {chunk['source_type']} - {chunk['source_name']}")
        print(f"Reference: {chunk['reference']}")
        print(f"Relevance: {chunk['relevance_score']:.2f}")
        print(f"Text: {chunk['text'][:150]}...")


if __name__ == "__main__":
    main()
