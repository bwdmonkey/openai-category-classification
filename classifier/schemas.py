"""
Pydantic schemas for structured outputs from OpenAI API.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, RootModel

class ItemProperties(BaseModel):
    """Core properties of an identified item"""
    quantity: Optional[int] = Field(None, description="The quantity of the specific item")
    condition: Optional[float] = Field(None, description="The condition of this specific item (0.0-1.0 scale)")
    confidence: Optional[float] = Field(None, description="Confidence in the classification (0.0-1.0 scale)")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional item metadata")

class ClassifiedItem(BaseModel):
    """A single classified item with unified schema for both image and tabular data"""
    category: str = Field(description="The specific item category")
    parent_category: str = Field(description="The parent category of the item")
    description: str = Field(description="Detailed description of the specific item")
    properties: Optional[ItemProperties] = Field(default_factory=ItemProperties, description="Item properties")
    source: Optional[str] = Field(None, description="Source of the item (file path, etc.)")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")

class ClassificationResult(BaseModel):
    """Complete analysis result with multiple items"""
    items: List[ClassifiedItem] = Field(description="Array of classified items")
    total_items: int = Field(description="Total number of items")
    source_file: Optional[str] = Field(None, description="Source file path")
    processed_date: Optional[str] = Field(None, description="Processing timestamp")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")

# Specific schemas for API interactions

class VisionAnalysis(BaseModel):
    """Schema for vision API response"""
    items: List[ClassifiedItem] = Field(description="Array of items identified in the image")
    total_items: int = Field(description="Total number of distinct items detected")

class BatchRequest(BaseModel):
    """Batch request format"""
    items: List[ClassifiedItem] = Field(description="Array of items to process")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Request metadata")