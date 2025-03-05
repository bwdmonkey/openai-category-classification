"""
Core classification functionality.
"""

import os
import glob
from typing import Dict, List, Union, Any

import pandas as pd

from .config import Config
from .models import Item
from .schemas import VisionAnalysis, Classification
from .utils import encode_image, logger

def extract_text_from_image(image_path: str) -> Dict[str, Any]:
    """
    Extract structured information from an image using OpenAI's vision model.

    Returns a dictionary with extracted properties and raw text.
    """
    client = Config.get_openai_client()
    base64_image = encode_image(image_path)

    system_prompt = """
    Analyze the image and identify each distinct item as a separate entity.
    For EACH item you identify, provide a complete separate analysis with category, description, and properties.
    Remember to include a quantity value in properties for each item.
    """

    try:
        if not Config.check_model_compatibility(Config.VISION_MODEL):
            raise ValueError(f"Model {Config.VISION_MODEL} is not compatible with the beta parse API. Use one of: {', '.join(Config.SUPPORTED_MODELS)}")

        completion = client.beta.chat.completions.parse(
            model=Config.VISION_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": "Please identify each distinct object in this image as a separate item with its own complete description."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                ]},
            ],
            response_format=VisionAnalysis,
        )

        # Extract the parsed result as a dictionary
        result = completion.choices[0].message.parsed.model_dump()

        # Initialize properties if needed
        for item in result.get('items', []):
            if 'properties' not in item or item['properties'] is None:
                item['properties'] = {}

        return result

    except Exception as e:
        logger.error(f"Error analyzing image {image_path}: {str(e)}")
        return {
            "items": [
                {
                    "error": str(e),
                    "category": "Error",
                    "description": f"Failed to analyze: {str(e)}",
                    "properties": {}
                }
            ],
            "total_items": 1
        }

def extract_text_from_csv(file_path: str) -> str:
    """Extract text from CSV."""
    try:
        df = pd.read_csv(file_path)
        return df.to_string()
    except Exception as e:
        logger.error(f"Error processing CSV {file_path}: {str(e)}")
        raise

def extract_text_from_xls(file_path: str) -> str:
    """Extract text from XLS/XLSX."""
    try:
        df = pd.read_excel(file_path)
        return df.to_string()
    except Exception as e:
        logger.error(f"Error processing Excel file {file_path}: {str(e)}")
        raise

def classify_category(extracted_data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Classify extracted data into predefined categories.

    Returns a dictionary with category and parent_category.
    """
    client = Config.get_openai_client()

    # Prepare text for classification
    if isinstance(extracted_data, dict):
        if "raw_text" in extracted_data:
            text_to_classify = extracted_data["raw_text"]
        else:
            # Use the extracted category and description
            category = extracted_data.get("category", "Unknown")
            description = extracted_data.get("description", "")
            text_to_classify = f"Category: {category}\nDescription: {description}"

            # If the category is already in our system, just return it
            for parent, children in Config.CATEGORIES.items():
                if category == parent:
                    return {"category": category, "parent_category": parent, "confidence": 1.0}
                if category in children:
                    return {"category": category, "parent_category": parent, "confidence": 1.0}
    else:
        text_to_classify = extracted_data

    # Prepare the prompt with category information
    categories_prompt = "Available categories:\n"
    for parent, children in Config.CATEGORIES.items():
        categories_prompt += f"- {parent}: {', '.join(children)}\n"

    system_prompt = f"""
    {categories_prompt}

    Classify the provided text into one of the predefined categories above.
    """

    try:
        if not Config.check_model_compatibility(Config.CLASSIFICATION_MODEL):
            raise ValueError(f"Model {Config.CLASSIFICATION_MODEL} is not compatible with the beta parse API. Use one of: {', '.join(Config.SUPPORTED_MODELS)}")

        completion = client.beta.chat.completions.parse(
            model=Config.CLASSIFICATION_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text_to_classify},
            ],
            response_format=Classification,
        )

        # Extract the parsed result as a dictionary
        result = completion.choices[0].message.parsed.model_dump()
        return result

    except Exception as e:
        logger.error(f"Error in classification: {str(e)}")
        return {"category": "Error", "parent_category": "Error", "confidence": 0.0}

def process_file(file_path: str) -> List[Item]:
    """
    Determine file type and process accordingly.
    Returns a list of Item objects with structured information.
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return [Item(file_path, text=f"Error: File not found", category="Error")]

    try:
        ext = os.path.splitext(file_path)[1].lower()
        items = []

        # Process based on file type
        if ext in [".png", ".jpeg", ".jpg", ".webp", ".gif", ".bmp"]:
            logger.info(f"Processing image file: {file_path}")
            extracted_data = extract_text_from_image(file_path)

            # Process each item found in the image
            for item_data in extracted_data.get("items", []):
                text = item_data.get("description", "")

                # If the extraction already provided a category, use it
                if item_data.get("category") not in ["Unknown", "Error"]:
                    classification = classify_category(item_data)
                else:
                    classification = classify_category(text)

                properties = item_data.get("properties", {})
                properties["confidence"] = classification.get("confidence", 0.0)

                item = Item(
                    file_path=file_path,
                    text=text,
                    category=classification["category"],
                    parent_category=classification["parent_category"],
                    properties=properties
                )
                items.append(item)

            # If no items were found, create a default item
            if not items:
                items.append(Item(
                    file_path=file_path,
                    text="No items detected",
                    category="Unknown",
                    parent_category="Unknown"
                ))

        elif ext == ".csv":
            logger.info(f"Processing CSV file: {file_path}")
            try:
                # Read the CSV file once
                df = pd.read_csv(file_path)
                text = df.to_string()

                # Process the entire CSV as one item
                classification = classify_category(text)

                item = Item(
                    file_path=file_path,
                    text=text,
                    category=classification["category"],
                    parent_category=classification["parent_category"],
                    properties={
                        "confidence": classification.get("confidence", 0.0),
                        "rows": len(df),
                        "columns": len(df.columns)
                    }
                )
                items.append(item)

            except Exception as e:
                logger.error(f"Failed to process CSV file {file_path}: {str(e)}")
                items.append(Item(
                    file_path=file_path,
                    text=f"Error processing CSV: {str(e)}",
                    category="Error",
                    parent_category="Error"
                ))

        elif ext in [".xls", ".xlsx"]:
            logger.info(f"Processing Excel file: {file_path}")
            try:
                # Get a list of all sheets
                xls = pd.ExcelFile(file_path)
                sheet_names = xls.sheet_names

                # If there's only one sheet, process the file as a single item
                if len(sheet_names) == 1:
                    df = pd.read_excel(file_path, sheet_name=sheet_names[0])
                    text = df.to_string()
                    classification = classify_category(text)

                    item = Item(
                        file_path=file_path,
                        text=text,
                        category=classification["category"],
                        parent_category=classification["parent_category"],
                        properties={
                            "confidence": classification.get("confidence", 0.0),
                            "sheet_name": sheet_names[0],
                            "rows": len(df),
                            "columns": len(df.columns)
                        }
                    )
                    items.append(item)

                # If there are multiple sheets, process the file as a whole
                else:
                    # Generate a single text representation of all sheets
                    all_sheets_text = []
                    for sheet_name in sheet_names:
                        df = pd.read_excel(file_path, sheet_name=sheet_name)
                        all_sheets_text.append(f"Sheet: {sheet_name}\n{df.to_string()}")

                    combined_text = "\n\n".join(all_sheets_text)
                    classification = classify_category(combined_text)

                    item = Item(
                        file_path=file_path,
                        text=combined_text,
                        category=classification["category"],
                        parent_category=classification["parent_category"],
                        properties={
                            "confidence": classification.get("confidence", 0.0),
                            "sheets": sheet_names,
                            "sheet_count": len(sheet_names)
                        }
                    )
                    items.append(item)

            except Exception as e:
                logger.error(f"Failed to process Excel file {file_path}: {str(e)}")
                items.append(Item(
                    file_path=file_path,
                    text=f"Error processing Excel file: {str(e)}",
                    category="Error",
                    parent_category="Error"
                ))

        else:
            logger.warning(f"Unsupported file format: {ext}")
            items.append(Item(file_path, text="Unsupported file format", category="Error"))

        # Save results if configured
        if Config.SAVE_RESULTS:
            for i, item in enumerate(items):
                if len(items) > 1:
                    # Modify file_name to indicate it's one of multiple items
                    base_name = os.path.splitext(item.file_name)[0]
                    ext = os.path.splitext(item.file_name)[1]
                    item.file_name = f"{base_name}_item{i+1}{ext}"

                output_path = item.save()
                logger.info(f"Results saved to: {output_path}")

        return items

    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        return [Item(file_path, text=f"Error: {str(e)}", category="Error")]

def process_files(file_paths: List[str]) -> List[Item]:
    """Process a list of files and return Item objects for each"""
    results = []
    total_files = len(file_paths)

    for index, file_path in enumerate(file_paths):
        try:
            logger.info(f"Processing file {index + 1}/{total_files}: {file_path}")
            items = process_file(file_path)
            results.extend(items)

            # Log if multiple items were found
            if len(items) > 1:
                logger.info(f"Found {len(items)} items in file: {file_path}")

        except Exception as e:
            logger.error(f"Failed to process {file_path}: {str(e)}")
            # Create an error item
            results.append(Item(
                file_path=file_path,
                text=f"Error processing file: {str(e)}",
                category="Error",
                parent_category="Error"
            ))

    return results

def classify_image(image_path: str, categories: List[str] = None) -> Dict[str, Any]:
    """Classify a single image and return the results."""
    items = process_file(image_path)
    result = {
        "file_path": image_path,
        "items": [item.to_dict() for item in items]
    }
    return result

def batch_classify(directory: str, recursive: bool = False,
                  pattern: str = None, output_dir: str = None) -> List[Item]:
    """Classify all images in a directory."""
    # Collect all files from the directory
    all_files = []

    if recursive:
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                if pattern:
                    if glob.fnmatch.fnmatch(file_path, pattern):
                        all_files.append(file_path)
                else:
                    # Check if it's a supported file type
                    ext = os.path.splitext(file_path)[1].lower()
                    supported_extensions = [
                        ".png", ".jpeg", ".jpg", ".webp", ".gif", ".bmp",
                        ".csv", ".xls", ".xlsx"
                    ]
                    if ext in supported_extensions:
                        all_files.append(file_path)
    else:
        # Non-recursive - just check files in the top directory
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                if pattern:
                    if glob.fnmatch.fnmatch(file_path, pattern):
                        all_files.append(file_path)
                else:
                    # Check if it's a supported file type
                    ext = os.path.splitext(file_path)[1].lower()
                    supported_extensions = [
                        ".png", ".jpeg", ".jpg", ".webp", ".gif", ".bmp",
                        ".csv", ".xls", ".xlsx"
                    ]
                    if ext in supported_extensions:
                        all_files.append(file_path)

    # Remove duplicates
    all_files = list(set(all_files))

    if not all_files:
        logger.warning("No files found to process!")
        return []

    logger.info(f"Found {len(all_files)} files to process")

    # Process all files
    if output_dir:
        Config.OUTPUT_DIR = output_dir

    return process_files(all_files)