"""Utility functions for text processing."""

from typing import List
import numpy as np


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        a (List[float]): First vector
        b (List[float]): Second vector
        
    Returns:
        float: Cosine similarity score
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)) 