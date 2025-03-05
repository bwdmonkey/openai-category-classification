"""
Configuration settings for the classification system.
"""

import os
import openai
from typing import Dict, List

class Config:
    """Configuration settings for the OpenAI classification system."""

    # OpenAI API settings
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    VISION_MODEL = "gpt-4o-mini"
    CLASSIFICATION_MODEL = "gpt-4o-mini"

    # Category definitions - TO BE EXTENDED
    CATEGORIES = {
        "Electronics": ["Laptop", "Smartphone", "Tablet", "Camera", "Headphones"],
        "Furniture": ["Chair", "Table", "Couch", "Desk", "Shelf", "Cabinet"],
        "Clothing": ["Shirt", "Pants", "Jacket", "Dress", "Shoes", "Hat"],
        "Books": ["Fiction", "Non-fiction", "Textbook", "Magazine"],
        "Kitchen": ["Utensil", "Appliance", "Cookware", "Dish"]
    }

    # Output settings
    OUTPUT_DIR = "output"
    SAVE_RESULTS = False

    # Models that support the beta structured output parse API
    SUPPORTED_MODELS = [
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-2024-08-06"
    ]

    @classmethod
    def get_openai_client(cls):
        if not cls.OPENAI_API_KEY:
            raise ValueError("OpenAI API key not found in environment variables")
        return openai.OpenAI(api_key=cls.OPENAI_API_KEY)

    @classmethod
    def check_model_compatibility(cls, model_name: str) -> bool:
        """Check if the model is compatible with the beta structured output parse API"""
        return any(model_name.startswith(supported) for supported in cls.SUPPORTED_MODELS)
