"""
Module for generating and comparing text embeddings using OpenAI's API.
"""

import os
from typing import List, Tuple
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.create_embedding_response import CreateEmbeddingResponse
from .config import Config
from .utils.text_processing import cosine_similarity


class EmbeddingGenerator:
    """Class to handle text embedding operations."""
    
    def __init__(self, config: Config = Config()):
        """
        Initialize the embedding generator.
        
        Args:
            config (Config): Configuration object
        """
        load_dotenv()
        self.client = OpenAI()
        self.config = config

    def generate_embeddings(self, texts: List[str]) -> CreateEmbeddingResponse:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts (List[str]): List of texts to generate embeddings for
            
        Returns:
            CreateEmbeddingResponse: Response containing embeddings
            
        Raises:
            openai.RateLimitError: If API quota is exceeded
            openai.AuthenticationError: If API key is invalid
        """
        try:
            response = self.client.embeddings.create(
                input=texts,
                model=self.config.model
            )
            return response
        except Exception as e:
            print(f"Error generating embeddings: {str(e)}")
            raise


def main():
    """Main function to demonstrate embedding generation and comparison."""
    
    # Example texts (Russian and English)
    texts = [
        "Кот сидит на окне",
        "Собака лежит на диване",
        "The cat is sitting on the window"
    ]
    
    try:
        # Initialize generator
        generator = EmbeddingGenerator()
        
        # Generate embeddings
        response = generator.generate_embeddings(texts)
        
        # Extract embeddings
        embeddings = [data.embedding for data in response.data]
        
        # Print sample of embeddings
        for i, embedding in enumerate(embeddings, 1):
            print(f"Embedding {i}:", embedding[:5], "...")
        
        # Calculate similarities
        sim_1_2 = cosine_similarity(embeddings[0], embeddings[1])
        sim_1_3 = cosine_similarity(embeddings[0], embeddings[2])
        
        # Print results
        print(f"\nSimilarity between Russian sentences: {sim_1_2:.4f}")
        print(f"Similarity between Russian and English versions: {sim_1_3:.4f}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main() 