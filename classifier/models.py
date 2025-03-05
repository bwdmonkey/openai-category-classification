"""
Data models for the classification system.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any

from .config import Config

class Item:
    """Represents an analyzed item with its properties."""

    def __init__(self,
                 file_path: str,
                 text: str = "",
                 category: str = "Unknown",
                 parent_category: str = "Unknown",
                 properties: Dict[str, Any] = None):
        self.file_path = file_path
        self.file_name = os.path.basename(file_path)
        self.text = text
        self.category = category
        self.parent_category = parent_category
        self.properties = properties or {}
        self.timestamp = datetime.now().isoformat()
        self.confidence_score = self.properties.get("confidence", 0.0)

    def to_dict(self) -> Dict[str, Any]:
        """Convert item to dictionary representation"""
        return {
            "file_path": self.file_path,
            "file_name": self.file_name,
            "text": self.text,
            "category": self.category,
            "parent_category": self.parent_category,
            "properties": self.properties,
            "timestamp": self.timestamp,
            "confidence_score": self.confidence_score
        }

    def to_json(self) -> str:
        """Convert item to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    def save(self, output_dir: str = None) -> str:
        """Save item analysis to file"""
        output_dir = output_dir or Config.OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{os.path.splitext(self.file_name)[0]}_{timestamp}.json"
        output_path = os.path.join(output_dir, filename)

        with open(output_path, 'w') as f:
            f.write(self.to_json())

        return output_path