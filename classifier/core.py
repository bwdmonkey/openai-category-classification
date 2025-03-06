"""
Core classification functionality.
"""

import os
import glob
import json
from typing import Dict, List, Union, Any, Tuple, Optional, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from .config import Config
from .models import Item
from .schemas import VisionAnalysis
from .utils import encode_image, logger

def extract_text_from_image(image_path: str) -> Dict[str, Any]:
    """
    Extract structured information from an image using OpenAI's vision model.
    Now includes category classification in the same API call.

    Returns a dictionary with extracted properties and category classifications.
    """
    client = Config.get_openai_client()
    base64_image = encode_image(image_path)

    # Include category information directly in the system prompt
    categories_info = json.dumps(Config.CATEGORIES, indent=2)

    # Define example JSON separately to avoid f-string formatting issues
    json_example = """
    {
        "items": [
            {
                "category": "T-Shirt", // The specific category of the item
                "parent_category": "Clothing", // The parent category of the item
                "description": "Red cotton t-shirt with logo on front", // A detailed description of the item
                "properties": {
                    "quantity": 1, // The quantity of the item
                    "condition": 0.8, // The condition of the item (0.0-1.0 scale)
                    "confidence": 0.9 // The confidence in the classification (0.0-1.0 scale)
                }
            },
            // Add more items as needed
        ],
        "total_items": 1
    }
    """

    system_prompt = f"""
    Analyze the image and identify each distinct item as a separate entity.
    For EACH item you identify, provide a complete separate analysis with:

    1. CATEGORY: Classify the item into one of these categories:
    {categories_info}

    2. DESCRIPTION: Detailed description of the item

    3. PROPERTIES: Include at minimum:
       - quantity: Number of this item (integer)
       - condition: Value between 0.0-1.0 where:
         - 1.0 = New/Perfect
         - 0.8 = Very Good
         - 0.6 = Good
         - 0.5 = Fair
         - 0.3 = Poor
         - 0.1 = Broken
         - 0.0 = Unusable
       - confidence: Value between 0.0-1.0

    For EACH item, determine both the specific category and parent category.

    Format your response in JSON like this example:
    ```
    {json_example}
    ```

    IMPORTANT: Your response MUST always include these two required fields:
    1. "items": An array of identified items, even if empty
    2. "total_items": The count of items found, even if 0

    These fields are required for processing and must always be present in your response.
    """

    try:
        if not Config.check_model_compatibility(Config.VISION_MODEL):
            raise ValueError(f"Model {Config.VISION_MODEL} is not compatible with the beta parse API. Use one of: {', '.join(Config.SUPPORTED_MODELS)}")

        logger.info(f"!!!!!! CALLING VISION MODEL !!!!!")
        # Use the standard chat completions API with JSON response format
        completion = client.chat.completions.create(
            model=Config.VISION_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": "Please identify and classify each distinct object in this image with its category and detailed description."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                ]},
            ],
            response_format={"type": "json_object"},
        )

        # Parse the JSON response
        try:
            response_text = completion.choices[0].message.content
            result = json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {str(e)}\nResponse: {response_text}")
            raise ValueError(f"Invalid JSON response from API: {str(e)}")

        # Ensure the result has the expected structure
        if "items" not in result or "total_items" not in result:
            logger.warning(f"Response missing required fields: {response_text}")
            result = {
                "items": result.get("items", []),
                "total_items": result.get("total_items", 0)
            }

        # Ensure properties exist and include parent_category for each item
        for item in result.get('items', []):
            if 'properties' not in item or item['properties'] is None:
                item['properties'] = {}

            # Set confidence to 1.0 since we're getting the category directly
            if 'confidence' not in item['properties']:
                item['properties']['confidence'] = 1.0

            # Determine parent_category based on the category
            category = item.get('category', 'Unknown')
            parent_found = False

            if 'parent_category' not in item or not item['parent_category']:
                for parent, children in Config.CATEGORIES.items():
                    if category == parent:
                        item['parent_category'] = parent
                        parent_found = True
                        break
                    if category in children:
                        item['parent_category'] = parent
                        parent_found = True
                        break
                # If we couldn't find the parent, set it to Unknown
                if not parent_found:
                    item['parent_category'] = 'Unknown'

        return result

    except Exception as e:
        logger.error(f"Error analyzing image {image_path}: {str(e)}")
        return {
            "items": [
                {
                    "error": str(e),
                    "category": "Error",
                    "parent_category": "Error",
                    "description": f"Failed to analyze: {str(e)}",
                    "properties": {"confidence": 0.0}
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

def batch_items(items: List[Any], batch_size: int) -> List[List[Any]]:
    """Split a list of items into batches of the specified size."""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

def process_batch(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process a batch of items with OpenAI API.

    Args:
        batch: List of dictionaries with row data to process

    Returns:
        List of processed items with classifications
    """
    client = Config.get_openai_client()
    system_prompt = """
    Analyze each row of data provided and identify the item it represents.
    For EACH row, provide a complete analysis with category, description, and properties.
    Include quantity, condition, and any other relevant properties.
    """

    try:
        if not Config.check_model_compatibility(Config.CLASSIFICATION_MODEL):
            raise ValueError(f"Model {Config.CLASSIFICATION_MODEL} is not compatible with the beta parse API.")

        # Prepare the request with all rows in the batch
        formatted_rows = []
        for row_data in batch:
            row_str = ", ".join([f"{k}: {v}" for k, v in row_data.get("data", {}).items()])
            formatted_rows.append(f"Row {row_data.get('row_index', 0)}: {row_str}")

        rows_text = "\n\n".join(formatted_rows)

        # We need to adjust the schema for batch processing
        # For now, let's use a custom prompt and parse the results manually
        completion = client.chat.completions.create(
            model=Config.CLASSIFICATION_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Please analyze each of these data rows and provide category, description, and properties for each:\n\n{rows_text}"}
            ],
            temperature=0.2,
        )

        # Here we'd process the response and match it back to the original rows
        # This is a simplified version - in a real implementation, we'd need a more
        # robust parser for the API response
        response_text = completion.choices[0].message.content

        # Process the response to extract classifications per row
        # For now, returning a placeholder with row data copied over
        results = []
        for row_data in batch:
            # We'd extract the actual classification from the response
            # This is a simplified placeholder
            results.append({
                "category": "Placeholder",
                "description": f"Row {row_data.get('row_index', 0)} analysis",
                "properties": {
                    "quantity": 1,
                    "condition": "Unknown"
                },
                "row_data": row_data.get("data", {}),
                "row_index": row_data.get("row_index", 0)
            })

        return results

    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        # Return error data for each row in the batch
        return [{
            "category": "Error",
            "description": f"Failed to analyze: {str(e)}",
            "properties": {},
            "row_data": row.get("data", {}),
            "row_index": row.get("row_index", 0)
        } for row in batch]

def process_tabular_file(file_path: str, batch_size: int = 250) -> List[Item]:
    """
    Process a CSV or Excel file row by row with batching.

    Args:
        file_path: Path to the CSV or Excel file
        batch_size: Number of rows to process in each batch

    Returns:
        List of Item objects, one per row
    """
    items = []
    ext = os.path.splitext(file_path)[1].lower()

    try:
        # Load the data
        if ext == ".csv":
            df = pd.read_csv(file_path)
        elif ext in [".xls", ".xlsx"]:
            # Process all sheets
            all_dfs = []
            xls = pd.ExcelFile(file_path)
            for sheet_name in xls.sheet_names:
                sheet_df = pd.read_excel(file_path, sheet_name=sheet_name)
                # Add sheet name as a column
                sheet_df['_sheet_name'] = sheet_name
                all_dfs.append(sheet_df)
            df = pd.concat(all_dfs, ignore_index=True)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        # Skip empty dataframes
        if df.empty:
            return [Item(
                file_path=file_path,
                text="Empty file - no data found",
                category="Empty",
                parent_category="Error"
            )]

        # Convert rows to a list of dictionaries
        row_data = []
        for idx, row in df.iterrows():
            # Convert row to dictionary and handle NaN values
            row_dict = row.to_dict()
            # Remove NaN values
            row_dict = {k: (str(v) if pd.notna(v) else "") for k, v in row_dict.items()}
            row_data.append({
                "row_index": int(idx),
                "data": row_dict
            })

        # Split into batches
        batches = batch_items(row_data, batch_size)
        logger.info(f"Processing {len(row_data)} rows in {len(batches)} batches")

        # Process batches with ThreadPoolExecutor for concurrent processing
        with ThreadPoolExecutor(max_workers=min(len(batches), 5)) as executor:
            future_to_batch = {executor.submit(classify_tabular_data_batch, batch): batch for batch in batches}

            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                try:
                    batch_results = future.result()

                    # Ensure we have a result for each row in the batch
                    if len(batch_results) != len(batch):
                        logger.warning(f"Batch size mismatch: expected {len(batch)} results, got {len(batch_results)}")
                        # Adjust the results to match the input batch
                        if len(batch_results) < len(batch):
                            # Add error entries for missing results
                            for i in range(len(batch_results), len(batch)):
                                batch_results.append({
                                    "category": "Error",
                                    "parent_category": "Error",
                                    "description": "Missing result from batch processing",
                                    "properties": {},
                                    "row_index": batch[i]["row_index"]
                                })
                        else:
                            # Truncate extra results
                            batch_results = batch_results[:len(batch)]

                    # Create Item objects from the batch results
                    for i, result in enumerate(batch_results):
                        row_idx = result.get("row_index", batch[i]["row_index"])
                        row_data = batch[i]["data"]

                        # Build a descriptive text from the row data
                        text = result.get("description", f"Row {row_idx}")

                        # Extract category information
                        category = result.get("category", "Unknown")
                        parent_category = result.get("parent_category", "Unknown")

                        # Ensure properties has at least the basic fields
                        properties = result.get("properties", {})
                        if not isinstance(properties, dict):
                            properties = {}

                        # Add row data to properties
                        properties["row_data"] = row_data
                        properties["row_index"] = row_idx

                        item = Item(
                            file_path=file_path,
                            text=text,
                            category=category,
                            parent_category=parent_category,
                            properties=properties
                        )
                        items.append(item)

                except Exception as e:
                    logger.error(f"Error processing batch: {str(e)}")
                    # Add error items for this batch
                    for row in batch:
                        items.append(Item(
                            file_path=file_path,
                            text=f"Error processing row {row['row_index']}: {str(e)}",
                            category="Error",
                            parent_category="Error",
                            properties={"row_index": row["row_index"], "row_data": row["data"]}
                        ))

        return items

    except Exception as e:
        logger.error(f"Error processing tabular file {file_path}: {str(e)}")
        return [Item(
            file_path=file_path,
            text=f"Error processing file: {str(e)}",
            category="Error",
            parent_category="Error"
        )]

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
            # Image processing with combined extraction and classification
            logger.info(f"Processing image file: {file_path}")
            extracted_data = extract_text_from_image(file_path)

            # Process each item found in the image
            for item_data in extracted_data.get("items", []):
                text = item_data.get("description", "")
                category = item_data.get("category", "Unknown")
                parent_category = item_data.get("parent_category", "Unknown")

                properties = item_data.get("properties", {})

                item = Item(
                    file_path=file_path,
                    text=text,
                    category=category,
                    parent_category=parent_category,
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

        elif ext in [".csv", ".xls", ".xlsx"]:
            # Tabular data processing
            logger.info(f"Processing tabular file: {file_path}")
            items = process_tabular_file(file_path, batch_size=250)

        else:
            logger.warning(f"Unsupported file format: {ext}")
            items = [Item(file_path, text="Unsupported file format", category="Error")]

        # Save all results to a single consolidated file
        if Config.SAVE_RESULTS and items:
            output_path = save_batch_results(items, file_path)
            logger.info(f"Consolidated results saved to: {output_path}")

        return items

    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        return [Item(file_path, text=f"Error: {str(e)}", category="Error")]

def process_files(file_paths: List[str], max_workers: int = 5) -> List[Item]:
    """
    Process a list of files concurrently using threading.

    Args:
        file_paths: List of paths to files to process
        max_workers: Maximum number of concurrent threads to use

    Returns:
        List of Item objects with structured information
    """
    results = []
    total_files = len(file_paths)

    logger.info(f"Processing {total_files} files using up to {max_workers} concurrent threads")

    # Process files concurrently using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(max_workers, total_files)) as executor:
        future_to_file = {executor.submit(process_file, file_path): file_path for file_path in file_paths}

        # As each file completes, add its results to our list
        for i, future in enumerate(as_completed(future_to_file)):
            file_path = future_to_file[future]
            try:
                logger.info(f"Completed file {i + 1}/{total_files}: {file_path}")
                items = future.result()
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

    logger.info(f"Completed processing {total_files} files with {len(results)} total items")
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
                  pattern: str = None, output_dir: str = None,
                  max_workers: int = 5) -> List[Item]:
    """
    Classify all images in a directory.

    Args:
        directory: Path to directory containing files to process
        recursive: Whether to recursively process subdirectories
        pattern: File pattern to match
        output_dir: Directory to save output files
        max_workers: Maximum number of concurrent threads to use

    Returns:
        List of Item objects with classification results
    """
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

    if output_dir:
        Config.OUTPUT_DIR = output_dir

    return process_files(all_files, max_workers=max_workers)

def classify_tabular_data_batch(batch_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Classify a batch of tabular data rows using the OpenAI API.

    Args:
        batch_data: List of dictionaries containing row data

    Returns:
        List of classification results for each row
    """
    client = Config.get_openai_client()

    # Create a structured prompt for the batch
    rows_json = json.dumps(batch_data, indent=0)

    system_prompt = f"""
    You are processing a batch of rows from a tabular dataset.
    For each row, you need to:
    1. Identify the category of the item described in the row
    2. Write a brief description of the item
    3. Extract properties including quantity and condition if present

    Available categories:
    {json.dumps(Config.CATEGORIES, indent=0)}

    IMPORTANT: For condition, provide a numerical value between 0.0 and 1.0, where:
    - 1.0 represents "New" or "Perfect" condition
    - 0.8 represents "Very Good" condition
    - 0.6 represents "Good" condition
    - 0.5 represents "Fair" condition
    - 0.3 represents "Poor" condition
    - 0.1 represents "Very Poor" or "Broken" condition
    - 0.0 represents "Unusable" condition

    Return your analysis as a JSON list with one object per input row, wrapped in an "analysis" array.
    For each row, include:
    - category: The specific category of the item
    - parent_category: The parent category the item belongs to
    - description: A brief description of the item
    - properties: An object with quantity, condition (0.0-1.0), and any other relevant properties
    - row_index: The original index of the row
    """

    try:
        # For simpler batch processing, we'll use the chat completion API directly
        # rather than the beta.parse API, and then parse the JSON response ourselves
        completion = client.chat.completions.create(
            model=Config.CLASSIFICATION_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze these rows: {rows_json}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )

        response_text = completion.choices[0].message.content
        try:
            results = json.loads(response_text)
            # Check for different possible response formats
            if "analysis" in results and isinstance(results["analysis"], list):
                return results["analysis"]
            elif "results" in results and isinstance(results["results"], list):
                return results["results"]
            elif isinstance(results, list):
                return results
            else:
                # Try to extract any list we can find in the response
                for key, value in results.items():
                    if isinstance(value, list) and len(value) > 0:
                        # Check if the first item has expected keys
                        if isinstance(value[0], dict) and "category" in value[0]:
                            logger.info(f"Found results in key: {key}")
                            return value

                # If we get here, we couldn't find a valid results list
                logger.warning(f"Unexpected response format: {response_text}")
                return [{
                    "category": "Error",
                    "parent_category": "Error",
                    "description": "Failed to parse API response",
                    "properties": {},
                    "row_index": row.get("row_index", 0)
                } for row in batch_data]
        except json.JSONDecodeError:
            logger.error(f"Could not parse JSON response: {response_text}")
            return [{
                "category": "Error",
                "parent_category": "Error",
                "description": "Invalid JSON response from API",
                "properties": {},
                "row_index": row.get("row_index", 0)
            } for row in batch_data]

    except Exception as e:
        logger.error(f"Error in batch classification: {str(e)}")
        return [{
            "category": "Error",
            "parent_category": "Error",
            "description": f"API error: {str(e)}",
            "properties": {},
            "row_index": row.get("row_index", 0)
        } for row in batch_data]

def save_batch_results(items: List[Item], file_path: str) -> str:
    """
    Save all items using the unified schema format.

    Args:
        items: List of Item objects to save
        file_path: Original file path of the source data

    Returns:
        Path to the saved JSON file
    """
    import datetime
    import json

    # Create output directory if it doesn't exist
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    # Create output filename based on input filename
    file_name = os.path.basename(file_path)
    base_name = os.path.splitext(file_name)[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(Config.OUTPUT_DIR, f"{timestamp}_{base_name}.json")

    # Format the items using the unified schema
    formatted_items = [format_item_for_json(item) for item in items]

    # Create the output following ClassificationResult schema
    output_data = {
        "items": formatted_items,
        "total_items": len(items),
        "source_file": file_path,
        "processed_date": datetime.datetime.now().isoformat(),
        "metadata": {
            "file_type": os.path.splitext(file_path)[1].lower(),
            "classifier_version": "1.0"  # Add version info
        }
    }

    # Write the consolidated JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    return output_file

def format_item_for_json(item: Item) -> Dict[str, Any]:
    """
    Format an Item object for JSON serialization using the unified schema.

    Args:
        item: The Item object to format

    Returns:
        A dictionary representation of the item suitable for JSON serialization
    """
    # Create a clean dictionary following the ClassifiedItem schema
    result = {
        "category": item.category,
        "parent_category": item.parent_category,
        "description": item.text,
        "source": item.file_path,
        "properties": {},
        "metadata": {}
    }

    # Extract properties
    if hasattr(item, "properties") and item.properties:
        properties = dict(item.properties)

        # Core properties go in the properties object
        core_properties = ["quantity", "condition", "confidence"]
        result["properties"] = {
            k: properties.pop(k) for k in core_properties
            if k in properties
        }

        # Move row_data and row_index to metadata
        if "row_data" in properties:
            result["metadata"]["row_data"] = properties.pop("row_data")
        if "row_index" in properties:
            result["metadata"]["row_index"] = properties.pop("row_index")

        # Add remaining properties to metadata
        result["metadata"].update(properties)

    return result

def normalize_condition(condition_text: str) -> Optional[float]:
    """
    Convert a textual condition description to a numerical value between 0.0 and 1.0.

    Args:
        condition_text: A string describing the condition (e.g., "Excellent", "Good", "Fair")

    Returns:
        A float between 0.0 and 1.0 representing the condition, or None if uncertain
    """
    # Dictionary mapping common condition descriptions to numerical values
    condition_map = {
        # Common condition descriptions with exact matches
        "new": 1.0,
        "excellent": 0.9,
        "very good": 0.8,
        "good": 0.7,
        "fair": 0.5,
        "poor": 0.3,
        "very poor": 0.2,
        "broken": 0.1,
        "unusable": 0.0,

        # Single-letter conditions often used in databases
        "n": 1.0,  # New
        "e": 0.9,  # Excellent
        "v": 0.8,  # Very Good
        "g": 0.7,  # Good
        "f": 0.5,  # Fair
        "p": 0.3,  # Poor
        "b": 0.1,  # Broken
    }

    # If it's already a float, just ensure it's in the right range
    if isinstance(condition_text, (float, int)):
        return max(0.0, min(float(condition_text), 1.0))

    # If it's None or empty, return None
    if not condition_text:
        return None  # We don't know the condition

    # Convert to lowercase and strip for better matching
    normalized_text = condition_text.lower().strip()

    # Check for exact match
    if normalized_text in condition_map:
        return condition_map[normalized_text]

    # Handle percentage format (e.g. "75%")
    if normalized_text.endswith("%"):
        try:
            percentage = float(normalized_text.rstrip("%"))
            return percentage / 100.0
        except ValueError:
            pass

    # Try to find partial matches
    for key, value in condition_map.items():
        if key in normalized_text:
            return value

    # If no match found, return None to indicate uncertainty
    return None