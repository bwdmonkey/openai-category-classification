"""
Utility functions for the classification system.
"""

import base64
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("classifier")

def encode_image(image_path: str) -> str:
    """Encodes an image file to base64 format for OpenAI API."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        logger.error(f"Error encoding image {image_path}: {str(e)}")
        raise

def print_results(item: Any) -> None:
    """Print formatted results"""
    print("\n" + "="*80)
    print(f"📄 File: {item.file_name}")
    print(f"📂 Category: {item.category} (Parent: {item.parent_category})")
    print(f"🎯 Confidence: {item.confidence_score:.2f}")
    print("-"*80)

    # Print properties
    if item.properties:
        print("📊 Properties:")
        for key, value in item.properties.items():
            if key != "confidence":  # Already printed above
                print(f"  - {key.capitalize()}: {value}")

    # Print text preview (shortened)
    if len(item.text) > 200:
        print(f"\n📝 Description: {item.text[:200]}...")
    else:
        print(f"\n📝 Description: {item.text}")

    print("="*80)

def print_summary(items: List[Any]) -> None:
    """Print a summary of processed items to console"""
    if not items:
        print("No items processed.")
        return

    # Count files and items
    file_paths = set(item.file_path for item in items)

    # Count categories
    category_counts = {}
    error_count = 0
    for item in items:
        if item.category == "Error":
            error_count += 1
        else:
            parent = item.parent_category
            if parent not in category_counts:
                category_counts[parent] = {}

            if item.category not in category_counts[parent]:
                category_counts[parent][item.category] = 1
            else:
                category_counts[parent][item.category] += 1

    # Print summary
    print("\n" + "="*80)
    print(f"📊 PROCESSING SUMMARY")
    print(f"📁 Total files processed: {len(file_paths)}")
    print(f"🔍 Total items found: {len(items)}")
    print(f"✅ Successfully categorized: {len(items) - error_count}")
    print(f"❌ Errors: {error_count}")
    print(f"📊 Average items per file: {len(items)/len(file_paths):.2f}")
    print("-"*80)

    # Print categories
    if category_counts:
        print("📋 Categories:")
        for parent, children in category_counts.items():
            print(f"  - {parent}:")
            for category, count in children.items():
                print(f"    - {category}: {count} item(s)")

    print("="*80)

def generate_summary_report(items: List[Any], output_path: str = None) -> str:
    """Generate a summary report of all processed items"""
    if not items:
        return "No items processed."

    summary = {
        "total_items": len(items),
        "categories": {},
        "processing_time": datetime.now().isoformat(),
        "success_rate": 0,
        "items": []
    }

    success_count = 0
    for item in items:
        # Add to category counts
        if item.category != "Error":
            success_count += 1
            if item.parent_category not in summary["categories"]:
                summary["categories"][item.parent_category] = {}

            if item.category not in summary["categories"][item.parent_category]:
                summary["categories"][item.parent_category][item.category] = 1
            else:
                summary["categories"][item.parent_category][item.category] += 1

        # Add item summary
        summary["items"].append({
            "file_name": item.file_name,
            "category": item.category,
            "parent_category": item.parent_category,
            "confidence": item.confidence_score
        })

    # Calculate success rate
    if items:
        summary["success_rate"] = success_count / len(items)

    # Save to file if requested
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)

    return json.dumps(summary, indent=2)