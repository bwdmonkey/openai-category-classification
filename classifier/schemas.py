"""
Pydantic schemas for structured outputs from OpenAI API.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, RootModel

class ItemProperties(BaseModel):
    """Properties of an identified item"""
    condition: Optional[str] = Field(None, description="The condition of this specific item")
    quantity: Optional[int] = Field(description="The quantity of the specific item")
    confidence: Optional[float] = Field(None, description="Confidence in the classification")

class VisionItem(BaseModel):
    """A single item identified in an image"""
    category: str = Field(description="The specific item category")
    description: str = Field(description="Detailed description of the specific item")
    properties: Optional[ItemProperties] = None

class VisionAnalysis(BaseModel):
    """Complete analysis of an image with multiple items"""
    items: List[VisionItem] = Field(description="Array of items identified in the image")
    total_items: int = Field(description="Total number of distinct items detected")

class Classification(BaseModel):
    """Classification of an item into predefined categories"""
    category: str = Field(description="The most specific subcategory that matches the item")
    parent_category: str = Field(description="The parent category of the item")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")

class RowClassification(BaseModel):
    """Classification for a single row in a dataset"""
    category: str
    parent_category: str
    confidence: float
    properties: Dict[str, Any] = Field(default_factory=dict)

# Use RootModel instead of __root__ for Pydantic v2 compatibility
class MultiRowClassification(RootModel):
    """Classification results for multiple rows"""
    root: Dict[str, RowClassification]