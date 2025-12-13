"""
============================================================================
SIMILARITY UTILITIES
============================================================================
Purpose: Text similarity and merging utilities for consensus building
Features:
    - Semantic similarity calculation
    - Text merging by consensus
    - Array deduplication by similarity
    - String normalization
    - Fuzzy matching

Usage:
    from utils.similarity_utils import (
        semantic_similarity,
        are_semantically_similar,
        merge_text_by_consensus,
        merge_arrays_by_similarity
    )
    
    # Check similarity
    sim = semantic_similarity("propulsion", "rocket engines")
    
    # Merge text from multiple models
    consensus = merge_text_by_consensus([text1, text2, text3])
    
    # Merge arrays with deduplication
    merged = merge_arrays_by_similarity([arr1, arr2, arr3])

Author: GATE AE SOTA Pipeline
============================================================================
"""

from typing import List, Any, Dict, Set
from difflib import SequenceMatcher
import re

from utils.logging_utils import setup_logger

logger = setup_logger("similarity_utils")


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison
    
    - Lowercase
    - Remove extra whitespace
    - Remove special characters (keep alphanumeric and spaces)
    
    Args:
        text: Input text
    
    Returns:
        str: Normalized text
    
    Example:
        normalize_text("  Combustion  Thermochemistry! ") 
        → "combustion thermochemistry"
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Lowercase
    text = text.lower()
    
    # Remove special characters (keep alphanumeric and spaces)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text


def semantic_similarity(text1: str, text2: str) -> float:
    """
    Calculate semantic similarity between two texts
    
    Uses SequenceMatcher for character-level similarity
    For production, could use sentence embeddings (slower but more accurate)
    
    Args:
        text1: First text
        text2: Second text
    
    Returns:
        float: Similarity score (0.0-1.0)
    
    Example:
        semantic_similarity("propulsion", "propulsion systems")
        → 0.73
        
        semantic_similarity("combustion", "thermochemistry")
        → 0.12
    """
    if not text1 or not text2:
        return 0.0
    
    # Normalize texts
    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)
    
    # Calculate similarity using SequenceMatcher
    similarity = SequenceMatcher(None, norm1, norm2).ratio()
    
    return similarity


def are_semantically_similar(
    text1: str,
    text2: str,
    threshold: float = 0.85
) -> bool:
    """
    Check if two texts are semantically similar above threshold
    
    Args:
        text1: First text
        text2: Second text
        threshold: Similarity threshold (0.0-1.0)
    
    Returns:
        bool: True if similar, False otherwise
    
    Example:
        are_semantically_similar("Combustion theory", "Combustion Theory", 0.85)
        → True
        
        are_semantically_similar("Combustion", "Propulsion", 0.85)
        → False
    """
    return semantic_similarity(text1, text2) >= threshold


def merge_text_by_consensus(
    texts: List[str],
    weights: List[float] = None,
    min_agreement: int = 2
) -> str:
    """
    Merge multiple text responses into consensus
    
    Strategy:
    1. Split texts into sentences
    2. Find common sentences (appear in multiple texts)
    3. Include sentences mentioned by min_agreement or more texts
    4. Apply weights if provided
    
    Args:
        texts: List of text responses
        weights: Optional weights for each text
        min_agreement: Minimum number of texts that must mention something
    
    Returns:
        str: Merged consensus text
    
    Example:
        texts = [
            "The image shows a graph. It has two axes.",
            "Graph with X and Y axes is shown.",
            "The diagram displays a graph with axes."
        ]
        merge_text_by_consensus(texts, min_agreement=2)
        → "The image shows a graph with axes."
    """
    if not texts:
        return ""
    
    if len(texts) == 1:
        return texts[0]
    
    # Default weights (equal)
    if weights is None:
        weights = [1.0 / len(texts)] * len(texts)
    
    # Split into sentences
    all_sentences = []
    for text, weight in zip(texts, weights):
        # Simple sentence splitting (split on . ! ?)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        all_sentences.append((sentences, weight))
    
    # Find common content
    sentence_scores = {}
    
    for sentences, weight in all_sentences:
        for sentence in sentences:
            norm_sentence = normalize_text(sentence)
            
            # Check if similar sentence already scored
            found_similar = False
            
            for existing_norm in sentence_scores.keys():
                if are_semantically_similar(norm_sentence, existing_norm, threshold=0.80):
                    # Add weight to existing
                    sentence_scores[existing_norm]['weight'] += weight
                    sentence_scores[existing_norm]['count'] += 1
                    found_similar = True
                    break
            
            if not found_similar:
                # New sentence
                sentence_scores[norm_sentence] = {
                    'original': sentence,
                    'weight': weight,
                    'count': 1
                }
    
    # Filter sentences by min_agreement
    consensus_sentences = []
    
    for norm_sentence, data in sentence_scores.items():
        if data['count'] >= min_agreement:
            consensus_sentences.append({
                'text': data['original'],
                'weight': data['weight'],
                'count': data['count']
            })
    
    # Sort by weight (highest first)
    consensus_sentences.sort(key=lambda x: x['weight'], reverse=True)
    
    # Combine into text
    if not consensus_sentences:
        # No consensus, use first text
        return texts[0]
    
    merged_text = '. '.join([s['text'] for s in consensus_sentences]) + '.'
    
    return merged_text


def merge_arrays_by_similarity(
    arrays: List[List[Any]],
    similarity_threshold: float = 0.85
) -> List[Any]:
    """
    Merge multiple arrays, deduplicating similar items
    
    Args:
        arrays: List of arrays to merge
        similarity_threshold: Threshold for considering items similar
    
    Returns:
        list: Merged array with duplicates removed
    
    Example:
        arrays = [
            ["Combustion theory", "Gas dynamics"],
            ["Combustion Theory", "Thermodynamics"],
            ["Gas Dynamics", "Propulsion"]
        ]
        merge_arrays_by_similarity(arrays)
        → ["Combustion theory", "Gas dynamics", "Thermodynamics", "Propulsion"]
    """
    if not arrays:
        return []
    
    if len(arrays) == 1:
        return arrays[0]
    
    # Flatten all arrays
    all_items = []
    for arr in arrays:
        all_items.extend(arr)
    
    # Deduplicate
    unique_items = []
    
    for item in all_items:
        is_duplicate = False
        
        for existing_item in unique_items:
            # Check similarity
            if isinstance(item, str) and isinstance(existing_item, str):
                if are_semantically_similar(item, existing_item, similarity_threshold):
                    is_duplicate = True
                    break
            elif item == existing_item:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_items.append(item)
    
    return unique_items


def deduplicate_list(
    items: List[Any],
    similarity_threshold: float = 0.85
) -> List[Any]:
    """
    Deduplicate list by removing similar items
    
    Args:
        items: List of items
        similarity_threshold: Similarity threshold for strings
    
    Returns:
        list: Deduplicated list
    """
    unique_items = []
    
    for item in items:
        is_duplicate = False
        
        for existing_item in unique_items:
            if isinstance(item, str) and isinstance(existing_item, str):
                if are_semantically_similar(item, existing_item, similarity_threshold):
                    is_duplicate = True
                    break
            elif item == existing_item:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_items.append(item)
    
    return unique_items


def find_most_common(items: List[str]) -> str:
    """
    Find most common item in list (with similarity consideration)
    
    Args:
        items: List of strings
    
    Returns:
        str: Most common item
    """
    if not items:
        return ""
    
    if len(items) == 1:
        return items[0]
    
    # Count occurrences (with similarity)
    item_counts = {}
    
    for item in items:
        norm_item = normalize_text(item)
        
        # Check if similar item already counted
        found_similar = False
        
        for existing_norm in item_counts.keys():
            if are_semantically_similar(norm_item, existing_norm, threshold=0.85):
                item_counts[existing_norm]['count'] += 1
                found_similar = True
                break
        
        if not found_similar:
            item_counts[norm_item] = {
                'original': item,
                'count': 1
            }
    
    # Find most common
    most_common = max(item_counts.values(), key=lambda x: x['count'])
    
    return most_common['original']


def calculate_consensus_score(
    values: List[Any],
    weights: List[float] = None
) -> float:
    """
    Calculate consensus score for a list of values
    
    Args:
        values: List of values
        weights: Optional weights for each value
    
    Returns:
        float: Consensus score (0.0-1.0)
            1.0 = all agree
            0.0 = no agreement
    
    Example:
        calculate_consensus_score([8, 8, 8, 7])
        → 0.75 (3 out of 4 agree)
    """
    if not values:
        return 0.0
    
    if len(values) == 1:
        return 1.0
    
    # Default weights
    if weights is None:
        weights = [1.0 / len(values)] * len(values)
    
    # Group similar values
    value_groups = {}
    
    for value, weight in zip(values, weights):
        found_group = False
        
        for existing_value in value_groups.keys():
            # Check if similar
            if isinstance(value, str) and isinstance(existing_value, str):
                if are_semantically_similar(value, existing_value, threshold=0.85):
                    value_groups[existing_value] += weight
                    found_group = True
                    break
            elif value == existing_value:
                value_groups[existing_value] += weight
                found_group = True
                break
        
        if not found_group:
            value_groups[value] = weight
    
    # Find max weight
    max_weight = max(value_groups.values()) if value_groups else 0.0
    
    return max_weight


def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    """
    Extract top keywords from text
    
    Simple extraction: most common non-stopword tokens
    
    Args:
        text: Input text
        top_n: Number of keywords to return
    
    Returns:
        list: Top keywords
    """
    # Normalize
    norm_text = normalize_text(text)
    
    # Split into words
    words = norm_text.split()
    
    # Stopwords
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
        'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'should', 'could', 'may', 'might', 'can', 'this', 'that'
    }
    
    # Filter stopwords and short words
    keywords = [w for w in words if w not in stopwords and len(w) > 2]
    
    # Count frequencies
    from collections import Counter
    keyword_counts = Counter(keywords)
    
    # Get top N
    top_keywords = [k for k, v in keyword_counts.most_common(top_n)]
    
    return top_keywords


def jaccard_similarity(set1: Set, set2: Set) -> float:
    """
    Calculate Jaccard similarity between two sets
    
    Jaccard = |A ∩ B| / |A ∪ B|
    
    Args:
        set1: First set
        set2: Second set
    
    Returns:
        float: Jaccard similarity (0.0-1.0)
    """
    if not set1 or not set2:
        return 0.0
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    if union == 0:
        return 0.0
    
    return intersection / union


# Example usage
if __name__ == "__main__":
    print("Testing Similarity Utils\n")
    
    # Test semantic similarity
    print("1. Semantic Similarity:")
    pairs = [
        ("Combustion thermochemistry", "Combustion Theory"),
        ("Propulsion", "Rocket engines"),
        ("Gas dynamics", "Fluid mechanics"),
        ("Aerodynamics", "Thermodynamics")
    ]
    
    for text1, text2 in pairs:
        sim = semantic_similarity(text1, text2)
        similar = are_semantically_similar(text1, text2, 0.70)
        print(f"  '{text1}' vs '{text2}':")
        print(f"    Similarity: {sim:.3f}, Similar: {similar}")
    
    # Test text merging
    print("\n2. Text Merging:")
    texts = [
        "The image shows a graph with X and Y axes. The curve is exponential.",
        "Graph with axes is displayed. Shows exponential growth curve.",
        "Diagram shows graph. X axis is horizontal, Y axis is vertical. Exponential curve."
    ]
    
    consensus = merge_text_by_consensus(texts, min_agreement=2)
    print(f"  Consensus: {consensus}")
    
    # Test array merging
    print("\n3. Array Merging:")
    arrays = [
        ["Combustion theory", "Gas dynamics", "Propulsion"],
        ["Combustion Theory", "Thermodynamics", "Fluid Mechanics"],
        ["Gas Dynamics", "Propulsion systems", "Aerodynamics"]
    ]
    
    merged = merge_arrays_by_similarity(arrays, similarity_threshold=0.80)
    print(f"  Merged array ({len(merged)} items):")
    for item in merged:
        print(f"    - {item}")
    
    # Test consensus score
    print("\n4. Consensus Score:")
    test_values = [
        [8, 8, 8, 7],
        ["A", "A", "B", "B"],
        ["Combustion", "Combustion theory", "Thermochemistry"]
    ]
    
    for values in test_values:
        score = calculate_consensus_score(values)
        print(f"  {values} → {score:.2%} consensus")
    
    # Test keyword extraction
    print("\n5. Keyword Extraction:")
    text = """
    The combustion process in gas turbine engines involves complex thermochemical 
    reactions. The fuel-air mixture undergoes rapid oxidation, releasing thermal 
    energy that drives the turbine. Understanding combustion thermochemistry is 
    essential for optimizing engine performance.
    """
    
    keywords = extract_keywords(text, top_n=5)
    print(f"  Keywords: {keywords}")