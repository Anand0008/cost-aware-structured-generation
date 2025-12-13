"""
============================================================================
STAGE 4B: SPARSE RETRIEVAL (BM25 Keyword Search)
============================================================================
Purpose: Retrieve top 10 chunks using keyword-based BM25 ranking
Used by: 99_pipeline_runner.py (Stage 4)
Author: GATE AE SOTA Pipeline
============================================================================
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, Any, List
from collections import Counter
import math
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from utils.logging_utils import setup_logger, log_stage

logger = setup_logger("05_retrieval_sparse")


class SparseRetriever:
    """
    Retrieve chunks using BM25 keyword-based ranking
    
    BM25 (Best Matching 25) is a probabilistic ranking function
    that scores documents based on term frequency and document frequency
    """
    
    def __init__(self, redis_client, configs: Dict):
        """
        Args:
            redis_client: Initialized Redis client
            configs: All configuration dictionaries
        """
        self.redis = redis_client
        self.configs = configs
        
        # BM25 parameters
        self.k1 = 1.5  # Term frequency saturation parameter
        self.b = 0.75  # Length normalization parameter
        
        # Redis keys for BM25 index
        self.index_prefix = "bm25_index:"
        self.doc_lengths_key = "bm25_doc_lengths"
        self.avg_doc_length_key = "bm25_avg_doc_length"
        self.total_docs_key = "bm25_total_docs"
        
        # Number of results to retrieve
        self.top_k = 10
    
    @log_stage("Stage 4B: Sparse Retrieval")
    def retrieve(self, question: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Retrieve top 10 chunks using BM25
        
        Args:
            question: Question object with question_text
        
        Returns:
            list: Top 10 chunks with BM25 scores
        """
        question_id = question.get('question_id', 'unknown')
        logger.info(f"BM25 retrieval for: {question_id}")
        
        # Extract keywords from question
        query_terms = self._extract_keywords(question)
        logger.info(f"  Query terms: {query_terms[:10]}...")
        
        # Calculate BM25 scores
        scored_chunks = self._calculate_bm25_scores(query_terms)
        
        # Sort by score and take top K
        scored_chunks.sort(key=lambda x: x['score'], reverse=True)
        top_results = scored_chunks[:self.top_k]
        
        logger.info(f"  ✓ Retrieved {len(top_results)} chunks")
        if top_results:
            logger.info(f"    - Avg score: {sum(r['score'] for r in top_results)/len(top_results):.3f}")
            logger.info(f"    - Top score: {top_results[0]['score']:.3f}")
        
        return top_results
    
    def _extract_keywords(self, question: Dict) -> List[str]:
        """
        Extract keywords from question text
        
        Uses:
            - Question text
            - Options (if MCQ)
        
        Returns:
            list: Lowercase keywords
        """
        text = question['question_text']
        
        # Add options for MCQ
        if question.get('options'):
            options_text = " ".join(question['options'].values())
            text = f"{text} {options_text}"
        
        # Tokenize
        # Remove special characters, keep only alphanumeric and spaces
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        
        # Split into words
        words = text.split()
        
        # Remove stopwords
        stopwords = self._get_stopwords()
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        
        return keywords
    
    def _get_stopwords(self) -> set:
        """
        Get common English stopwords
        
        Returns:
            set: Stopwords to exclude from query
        """
        return {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'should', 'could', 'may', 'might', 'can', 'this', 'that',
            'these', 'those', 'it', 'its', 'which', 'what', 'when', 'where',
            'who', 'how', 'if', 'than', 'then', 'so', 'just', 'not', 'all',
            'any', 'some', 'such', 'no', 'nor', 'too', 'very', 'only', 'also'
        }
    
    def _calculate_bm25_scores(self, query_terms: List[str]) -> List[Dict]:
        """
        Calculate BM25 scores for all documents
        
        BM25 formula:
        score(D,Q) = Σ IDF(qi) * (f(qi,D) * (k1 + 1)) / (f(qi,D) + k1 * (1 - b + b * |D| / avgdl))
        
        Where:
            - D: document
            - Q: query
            - qi: query term i
            - f(qi,D): frequency of qi in D
            - |D|: length of D
            - avgdl: average document length
            - k1, b: tuning parameters
        
        Args:
            query_terms: List of query keywords
        
        Returns:
            list: Documents with BM25 scores
        """
        try:
            # Get corpus statistics
            total_docs = int(self.redis.get(self.total_docs_key) or 0)
            avg_doc_length = float(self.redis.get(self.avg_doc_length_key) or 1.0)
            
            if total_docs == 0:
                logger.warning("BM25 index is empty - no documents indexed")
                return []
            
            logger.info(f"  Corpus: {total_docs} documents, avg length: {avg_doc_length:.1f} terms")
            
            # Calculate IDF for each query term
            idf_scores = {}
            for term in set(query_terms):  # Unique terms
                # Get document frequency (how many docs contain this term)
                df = self.redis.scard(f"{self.index_prefix}term:{term}")
                
                if df > 0:
                    # IDF formula: log((N - df + 0.5) / (df + 0.5) + 1)
                    idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1)
                    idf_scores[term] = idf
            
            if not idf_scores:
                logger.warning("No query terms found in index")
                return []
            
            # Get all candidate documents (documents containing any query term)
            candidate_docs = set()
            for term in idf_scores.keys():
                docs = self.redis.smembers(f"{self.index_prefix}term:{term}")
                candidate_docs.update(docs)
            
            logger.info(f"  Found {len(candidate_docs)} candidate documents")
            
            # Calculate BM25 score for each candidate document
            scored_docs = []
            
            for doc_id in candidate_docs:
                # Get document data
                doc_data = self.redis.get(f"{self.index_prefix}doc:{doc_id}")
                if not doc_data:
                    continue
                
                doc = json.loads(doc_data)
                doc_length = len(doc.get('terms', []))
                
                # Calculate BM25 score
                score = 0.0
                
                for term in idf_scores.keys():
                    # Term frequency in document
                    tf = doc.get('term_frequencies', {}).get(term, 0)
                    
                    if tf > 0:
                        # BM25 component for this term
                        numerator = tf * (self.k1 + 1)
                        denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / avg_doc_length)
                        
                        score += idf_scores[term] * (numerator / denominator)
                
                if score > 0:
                    scored_docs.append({
                        'chunk_id': doc_id,
                        'score': score,
                        'source_type': doc.get('source_type'),
                        'source_name': doc.get('source_name'),
                        'text': doc.get('text'),
                        'metadata': doc.get('metadata', {})
                    })
            
            return scored_docs
            
        except Exception as e:
            logger.error(f"BM25 scoring failed: {e}")
            return []
    
    def index_document(
        self,
        doc_id: str,
        text: str,
        source_type: str,
        source_name: str,
        metadata: Dict
    ):
        """
        Index a document for BM25 search
        
        This is called during corpus preparation (offline)
        
        Args:
            doc_id: Unique document ID
            text: Document text
            source_type: "book" or "video"
            source_name: Book/video title
            metadata: Additional metadata
        """
        try:
            # Tokenize text
            text_lower = re.sub(r'[^\w\s]', ' ', text.lower())
            terms = text_lower.split()
            
            # Remove stopwords
            stopwords = self._get_stopwords()
            terms = [t for t in terms if t not in stopwords and len(t) > 2]
            
            # Calculate term frequencies
            term_frequencies = Counter(terms)
            
            # Create document entry
            doc_entry = {
                'doc_id': doc_id,
                'source_type': source_type,
                'source_name': source_name,
                'text': text,
                'metadata': metadata,
                'terms': terms,
                'term_frequencies': dict(term_frequencies)
            }
            
            # Save document
            self.redis.set(
                f"{self.index_prefix}doc:{doc_id}",
                json.dumps(doc_entry)
            )
            
            # Update inverted index (term -> doc_ids)
            for term in term_frequencies.keys():
                self.redis.sadd(f"{self.index_prefix}term:{term}", doc_id)
            
            # Update document length
            self.redis.hset(self.doc_lengths_key, doc_id, len(terms))
            
            # Update total documents count
            self.redis.incr(self.total_docs_key)
            
            # Update average document length (will be calculated after all docs indexed)
            
        except Exception as e:
            logger.error(f"Failed to index document {doc_id}: {e}")
    
    def finalize_index(self):
        """
        Calculate average document length after all documents indexed
        
        Call this once after indexing all documents
        """
        try:
            doc_lengths = self.redis.hgetall(self.doc_lengths_key)
            
            if doc_lengths:
                lengths = [int(length) for length in doc_lengths.values()]
                avg_length = sum(lengths) / len(lengths)
                
                self.redis.set(self.avg_doc_length_key, avg_length)
                
                logger.info(f"✓ BM25 index finalized:")
                logger.info(f"  - Total documents: {len(lengths)}")
                logger.info(f"  - Average length: {avg_length:.1f} terms")
            else:
                logger.warning("No documents found in index")
                
        except Exception as e:
            logger.error(f"Failed to finalize index: {e}")
    
    def get_index_stats(self) -> Dict:
        """
        Get BM25 index statistics
        
        Returns:
            dict: Index statistics
        """
        try:
            total_docs = int(self.redis.get(self.total_docs_key) or 0)
            avg_doc_length = float(self.redis.get(self.avg_doc_length_key) or 0)
            
            # Count unique terms
            term_keys = self.redis.keys(f"{self.index_prefix}term:*")
            unique_terms = len(term_keys)
            
            return {
                'total_documents': total_docs,
                'average_document_length': round(avg_doc_length, 1),
                'unique_terms': unique_terms,
                'index_size_mb': self._estimate_index_size()
            }
            
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            return {'error': str(e)}
    
    def _estimate_index_size(self) -> float:
        """Estimate Redis memory usage for BM25 index"""
        try:
            info = self.redis.info('memory')
            used_memory_mb = info.get('used_memory', 0) / (1024 * 1024)
            return round(used_memory_mb, 2)
        except:
            return 0.0


def main():
    """
    Test sparse retrieval
    """
    import argparse
    from scripts.initialization import PipelineInitializer
    from scripts.question_loader import QuestionLoader
    
    parser = argparse.ArgumentParser(description="Test Sparse Retrieval")
    parser.add_argument("--question-json", help="Path to question JSON file")
    parser.add_argument("--stats", action="store_true", help="Show index statistics")
    
    args = parser.parse_args()
    
    # Initialize
    print("\n" + "="*80)
    print("INITIALIZING")
    print("="*80 + "\n")
    
    initializer = PipelineInitializer()
    components = initializer.initialize_all()
    
    retriever = SparseRetriever(
        redis_client=components['clients']['redis'],
        configs=components['configs']
    )
    
    if args.stats:
        stats = retriever.get_index_stats()
        print("\n" + "="*80)
        print("BM25 INDEX STATISTICS")
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
    print("SPARSE RETRIEVAL (BM25)")
    print("="*80 + "\n")
    
    results = retriever.retrieve(question)
    
    # Print results
    print("\n" + "="*80)
    print(f"RETRIEVED {len(results)} CHUNKS")
    print("="*80 + "\n")
    
    for i, chunk in enumerate(results, 1):
        print(f"\n--- CHUNK {i} (BM25 Score: {chunk['score']:.3f}) ---")
        print(f"Source: {chunk['source_type']} - {chunk['source_name']}")
        print(f"Text: {chunk['text'][:200]}...")


if __name__ == "__main__":
    main()
