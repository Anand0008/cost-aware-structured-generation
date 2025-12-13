"""
============================================================================
EMBEDDING UTILITIES
============================================================================
Purpose: Text embedding generation and similarity calculations
Features:
    - Generate embeddings using BGE-M3 model
    - Cosine similarity calculation
    - Batch embedding generation
    - Embedding caching
    - Dimensionality reduction (optional)

Model: BGE-M3 (BAAI/bge-m3)
    - Embedding dimension: 1024
    - Max sequence length: 8192 tokens
    - Multilingual support
    - State-of-the-art performance

Usage:
    from utils.embedding_utils import generate_embedding, cosine_similarity
    
    # Generate embedding
    embedding = generate_embedding("What is propulsion?", model)
    
    # Calculate similarity
    sim = cosine_similarity(embedding1, embedding2)

Author: GATE AE SOTA Pipeline
============================================================================
"""

import numpy as np
from typing import List, Union, Optional
import hashlib
import pickle
from pathlib import Path

from utils.logging_utils import setup_logger

logger = setup_logger("embedding_utils")


# Cache directory for embeddings
CACHE_DIR = Path(__file__).parent.parent / "cache" / "embeddings"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def generate_embedding(
    text: str,
    model,
    normalize: bool = True,
    use_cache: bool = True
) -> np.ndarray:
    """
    Generate embedding for text using BGE-M3 model
    
    Args:
        text: Input text
        model: SentenceTransformer model (BGE-M3)
        normalize: Normalize embedding to unit length
        use_cache: Use cached embedding if available
    
    Returns:
        np.ndarray: Embedding vector (1024 dimensions)
    
    Example:
        embedding = generate_embedding(
            "The fuel-air ratio in combustion...",
            bge_model
        )
        # Returns: array([0.123, -0.456, ...]) shape=(1024,)
    """
    if not text or not text.strip():
        logger.warning("Empty text provided for embedding")
        return np.zeros(1024)
    
    # Check cache
    if use_cache:
        cached = _load_from_cache(text)
        if cached is not None:
            return cached
    
    try:
        # Generate embedding
        embedding = model.encode(
            text,
            normalize_embeddings=normalize,
            show_progress_bar=False
        )
        
        # Convert to numpy array if not already
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)
        
        # Save to cache
        if use_cache:
            _save_to_cache(text, embedding)
        
        return embedding
        
    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}")
        # Return zero vector as fallback
        return np.zeros(1024)


def generate_embeddings_batch(
    texts: List[str],
    model,
    normalize: bool = True,
    batch_size: int = 32,
    show_progress: bool = False
) -> List[np.ndarray]:
    """
    Generate embeddings for multiple texts (batched for efficiency)
    
    Args:
        texts: List of input texts
        model: SentenceTransformer model
        normalize: Normalize embeddings
        batch_size: Batch size for processing
        show_progress: Show progress bar
    
    Returns:
        list: List of embedding vectors
    
    Example:
        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = generate_embeddings_batch(texts, model)
        # Returns: [array([...]), array([...]), array([...])]
    """
    if not texts:
        return []
    
    try:
        # Generate embeddings in batches
        embeddings = model.encode(
            texts,
            normalize_embeddings=normalize,
            batch_size=batch_size,
            show_progress_bar=show_progress
        )
        
        # Convert to list of numpy arrays
        if isinstance(embeddings, np.ndarray):
            embeddings = [embeddings[i] for i in range(len(embeddings))]
        
        return embeddings
        
    except Exception as e:
        logger.error(f"Failed to generate batch embeddings: {e}")
        # Return zero vectors as fallback
        return [np.zeros(1024) for _ in texts]


def cosine_similarity(
    embedding1: np.ndarray,
    embedding2: np.ndarray
) -> float:
    """
    Calculate cosine similarity between two embeddings
    
    Cosine similarity = (A · B) / (||A|| × ||B||)
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
    
    Returns:
        float: Cosine similarity (-1.0 to 1.0, typically 0.0 to 1.0)
    
    Example:
        sim = cosine_similarity(emb1, emb2)
        # Returns: 0.87 (87% similar)
    """
    if embedding1 is None or embedding2 is None:
        return 0.0
    
    if len(embedding1) == 0 or len(embedding2) == 0:
        return 0.0
    
    # Ensure numpy arrays
    if not isinstance(embedding1, np.ndarray):
        embedding1 = np.array(embedding1)
    if not isinstance(embedding2, np.ndarray):
        embedding2 = np.array(embedding2)
    
    # Calculate cosine similarity
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    similarity = dot_product / (norm1 * norm2)
    
    # Clamp to [-1, 1] range (numerical stability)
    similarity = np.clip(similarity, -1.0, 1.0)
    
    return float(similarity)


def euclidean_distance(
    embedding1: np.ndarray,
    embedding2: np.ndarray
) -> float:
    """
    Calculate Euclidean distance between two embeddings
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
    
    Returns:
        float: Euclidean distance (0.0 = identical, higher = more different)
    """
    if embedding1 is None or embedding2 is None:
        return float('inf')
    
    if not isinstance(embedding1, np.ndarray):
        embedding1 = np.array(embedding1)
    if not isinstance(embedding2, np.ndarray):
        embedding2 = np.array(embedding2)
    
    return float(np.linalg.norm(embedding1 - embedding2))


def find_most_similar(
    query_embedding: np.ndarray,
    candidate_embeddings: List[np.ndarray],
    top_k: int = 5
) -> List[tuple]:
    """
    Find top-k most similar embeddings to query
    
    Args:
        query_embedding: Query embedding
        candidate_embeddings: List of candidate embeddings
        top_k: Number of top results to return
    
    Returns:
        list: List of (index, similarity_score) tuples, sorted by similarity
    
    Example:
        results = find_most_similar(query_emb, [emb1, emb2, emb3], top_k=2)
        # Returns: [(1, 0.95), (0, 0.87)]  # indices and scores
    """
    if not candidate_embeddings:
        return []
    
    # Calculate similarities
    similarities = []
    
    for i, candidate in enumerate(candidate_embeddings):
        sim = cosine_similarity(query_embedding, candidate)
        similarities.append((i, sim))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Return top-k
    return similarities[:top_k]


def _get_cache_key(text: str) -> str:
    """
    Generate cache key for text
    
    Uses MD5 hash of text to create unique filename
    
    Args:
        text: Input text
    
    Returns:
        str: Cache key (MD5 hash)
    """
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def _load_from_cache(text: str) -> Optional[np.ndarray]:
    """
    Load embedding from cache
    
    Args:
        text: Input text
    
    Returns:
        np.ndarray: Cached embedding or None if not found
    """
    cache_key = _get_cache_key(text)
    cache_file = CACHE_DIR / f"{cache_key}.pkl"
    
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                embedding = pickle.load(f)
            
            logger.debug(f"Embedding loaded from cache: {cache_key[:8]}...")
            return embedding
            
        except Exception as e:
            logger.warning(f"Failed to load cached embedding: {e}")
            return None
    
    return None


def _save_to_cache(text: str, embedding: np.ndarray):
    """
    Save embedding to cache
    
    Args:
        text: Input text
        embedding: Embedding to cache
    """
    cache_key = _get_cache_key(text)
    cache_file = CACHE_DIR / f"{cache_key}.pkl"
    
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(embedding, f)
        
        logger.debug(f"Embedding saved to cache: {cache_key[:8]}...")
        
    except Exception as e:
        logger.warning(f"Failed to save embedding to cache: {e}")


def clear_cache():
    """Clear all cached embeddings"""
    cache_files = list(CACHE_DIR.glob("*.pkl"))
    
    for cache_file in cache_files:
        cache_file.unlink()
    
    logger.info(f"Cleared {len(cache_files)} cached embeddings")


def get_cache_stats() -> dict:
    """
    Get cache statistics
    
    Returns:
        dict: {
            "total_embeddings": int,
            "cache_size_mb": float
        }
    """
    cache_files = list(CACHE_DIR.glob("*.pkl"))
    
    total_size = sum(f.stat().st_size for f in cache_files)
    size_mb = total_size / (1024 * 1024)
    
    return {
        "total_embeddings": len(cache_files),
        "cache_size_mb": round(size_mb, 2)
    }


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """
    Normalize embedding to unit length
    
    Args:
        embedding: Input embedding
    
    Returns:
        np.ndarray: Normalized embedding
    """
    if not isinstance(embedding, np.ndarray):
        embedding = np.array(embedding)
    
    norm = np.linalg.norm(embedding)
    
    if norm == 0:
        return embedding
    
    return embedding / norm


def average_embeddings(
    embeddings: List[np.ndarray],
    weights: Optional[List[float]] = None
) -> np.ndarray:
    """
    Calculate weighted average of embeddings
    
    Args:
        embeddings: List of embedding vectors
        weights: Optional weights (must sum to 1.0)
    
    Returns:
        np.ndarray: Average embedding
    
    Example:
        avg_emb = average_embeddings([emb1, emb2, emb3], weights=[0.5, 0.3, 0.2])
    """
    if not embeddings:
        return np.zeros(1024)
    
    # Default weights (equal)
    if weights is None:
        weights = [1.0 / len(embeddings)] * len(embeddings)
    
    # Ensure weights sum to 1.0
    total_weight = sum(weights)
    if total_weight > 0:
        weights = [w / total_weight for w in weights]
    
    # Calculate weighted average
    avg_embedding = np.zeros_like(embeddings[0])
    
    for embedding, weight in zip(embeddings, weights):
        avg_embedding += embedding * weight
    
    return avg_embedding


# Example usage
if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer
    
    print("Testing Embedding Utils\n")
    
    # Load BGE-M3 model
    print("Loading BGE-M3 model...")
    model = SentenceTransformer('BAAI/bge-m3', device='cpu')
    print("✓ Model loaded\n")
    
    # Test single embedding
    print("1. Single Embedding:")
    text = "The fuel-air ratio in gas turbine combustion"
    embedding = generate_embedding(text, model)
    print(f"  Text: {text}")
    print(f"  Embedding shape: {embedding.shape}")
    print(f"  Embedding norm: {np.linalg.norm(embedding):.3f}")
    
    # Test batch embeddings
    print("\n2. Batch Embeddings:")
    texts = [
        "Combustion thermochemistry",
        "Propulsion systems",
        "Gas turbine engines"
    ]
    embeddings = generate_embeddings_batch(texts, model)
    print(f"  Generated {len(embeddings)} embeddings")
    
    # Test similarity
    print("\n3. Cosine Similarity:")
    pairs = [
        (0, 1),  # Combustion vs Propulsion
        (0, 2),  # Combustion vs Gas turbine
        (1, 2)   # Propulsion vs Gas turbine
    ]
    
    for i, j in pairs:
        sim = cosine_similarity(embeddings[i], embeddings[j])
        print(f"  '{texts[i]}' vs '{texts[j]}':")
        print(f"    Similarity: {sim:.3f}")
    
    # Test find most similar
    print("\n4. Find Most Similar:")
    query = "Rocket propulsion fundamentals"
    query_emb = generate_embedding(query, model)
    
    results = find_most_similar(query_emb, embeddings, top_k=3)
    print(f"  Query: {query}")
    print(f"  Top results:")
    for idx, sim in results:
        print(f"    - {texts[idx]}: {sim:.3f}")
    
    # Test average embeddings
    print("\n5. Average Embeddings:")
    avg_emb = average_embeddings(
        [embeddings[0], embeddings[1]],
        weights=[0.6, 0.4]
    )
    print(f"  Average embedding shape: {avg_emb.shape}")
    print(f"  Similarity to first: {cosine_similarity(avg_emb, embeddings[0]):.3f}")
    print(f"  Similarity to second: {cosine_similarity(avg_emb, embeddings[1]):.3f}")
    
    # Test cache
    print("\n6. Cache Stats:")
    stats = get_cache_stats()
    print(f"  Cached embeddings: {stats['total_embeddings']}")
    print(f"  Cache size: {stats['cache_size_mb']:.2f} MB")
    
    print("\n✓ All tests complete")