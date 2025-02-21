"""Tests for the embedding generator."""

import unittest
from src.utils.text_processing import cosine_similarity


class TestEmbedding(unittest.TestCase):
    """Test cases for embedding functionality."""

    def test_cosine_similarity(self):
        """Test the cosine similarity calculation."""
        a = [1, 0, 0]
        b = [1, 0, 0]
        self.assertEqual(cosine_similarity(a, b), 1.0)

        a = [1, 0, 0]
        b = [0, 1, 0]
        self.assertEqual(cosine_similarity(a, b), 0.0)


if __name__ == '__main__':
    unittest.main() 
    