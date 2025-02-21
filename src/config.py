"""Configuration settings for the embedding generator."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Configuration class for embedding generation."""
    
    model: str = "text-embedding-ada-002"
    max_retries: int = 3
    timeout: int = 30 