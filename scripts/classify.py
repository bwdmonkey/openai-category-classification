#!/usr/bin/env python3
"""
OpenAI vision-based category classification command-line interface.
"""

import argparse
import os
import glob
import time

from classifier import (
    process_file,
    process_files,
    Config
)
from classifier.utils import print_results, print_summary, generate_summary_report

def main():
    """Command-line interface for the classification system."""
    start_time = time.time()

    parser = argparse.ArgumentParser(
        description="Classify files using OpenAI Vision and Language Models")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Single image classification
    single_parser = subparsers.add_parser("single", help="Classify a single image")
    single_parser.add_argument("image_path", help="Path to the image file")
    single_parser.add_argument("--categories", nargs="+", help="List of categories to classify against")
    single_parser.add_argument("--output", help="Output directory for results")
    single_parser.add_argument("--save", action="store_true", help="Save results to output directory")
    single_parser.add_argument("--threads", type=int, default=5,
                              help="Maximum number of concurrent threads")

    # Batch classification
    batch_parser = subparsers.add_parser("batch", help="Classify all images in a directory")
    batch_parser.add_argument("directory", help="Directory containing images")
    batch_parser.add_argument("--recursive", "-r", action="store_true",
                             help="Recursively process directories")
    batch_parser.add_argument("--pattern", help="File pattern to match (e.g. '*.jpg')")
    batch_parser.add_argument("--output", help="Output directory for results")
    batch_parser.add_argument("--save", action="store_true", help="Save results to output directory")
    batch_parser.add_argument("--summary", action="store_true", help="Generate a summary report")
    batch_parser.add_argument("--summary-file", help="Path to save summary report")
    batch_parser.add_argument("--threads", type=int, default=5,
                             help="Maximum number of concurrent threads")

    # Advanced processing
    advanced_parser = subparsers.add_parser("advanced",
                                           help="Process files with advanced options")
    advanced_parser.add_argument("paths", nargs='+',
                               help="Paths to files or directories to process")
    advanced_parser.add_argument("--recursive", "-r", action="store_true",
                               help="Recursively process directories")
    advanced_parser.add_argument("--pattern", help="File pattern to match (e.g. '*.jpg')")
    advanced_parser.add_argument("--output", help="Output directory for results")
    advanced_parser.add_argument("--save", action="store_true", help="Save results to output directory")
    advanced_parser.add_argument("--summary", action="store_true",
                               help="Generate a summary report")
    advanced_parser.add_argument("--summary-file", help="Path to save summary report")
    advanced_parser.add_argument("--threads", type=int, default=5,
                               help="Maximum number of concurrent threads")

    args = parser.parse_args()

    # Set output directory if specified
    if hasattr(args, 'output') and args.output:
        Config.OUTPUT_DIR = args.output

    # Set save results flag
    if hasattr(args, 'save') and args.save:
        Config.SAVE_RESULTS = True
    else:
        Config.SAVE_RESULTS = False

    # Set categories if specified
    if hasattr(args, 'categories') and args.categories:
        # Update categories in config if provided
        custom_categories = {}
        for category in args.categories:
            custom_categories[category] = [category.lower()]
        Config.CATEGORIES = custom_categories

    # Process the appropriate command
    if args.command == "single":
        # Process a single image
        results = process_file(args.image_path)
        for item in results:
            print_results(item)

    elif args.command == "batch":
        # Collect all files from the directory
        all_files = []
        if args.recursive:
            for root, _, files in os.walk(args.directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    if args.pattern:
                        if glob.fnmatch.fnmatch(file_path, args.pattern):
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
            for filename in os.listdir(args.directory):
                file_path = os.path.join(args.directory, filename)
                if os.path.isfile(file_path):
                    if args.pattern:
                        if glob.fnmatch.fnmatch(file_path, args.pattern):
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
            print("No files found to process!")
            return

        print(f"Found {len(all_files)} files to process")

        # Process all files
        results = process_files(all_files, max_workers=args.threads)

        # Print results
        for item in results:
            print_results(item)

        # Generate summary if requested
        if args.summary or args.summary_file:
            print_summary(results)

            if args.summary_file:
                generate_summary_report(results, args.summary_file)
                print(f"Summary report saved to: {args.summary_file}")

    elif args.command == "advanced":
        # Collect all files to process
        all_files = []

        for path in args.paths:
            if os.path.isdir(path):
                print(f"Collecting files from directory: {path}")

                if args.recursive:
                    # Recursive directory traversal
                    for root, _, files in os.walk(path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            if args.pattern:
                                if glob.fnmatch.fnmatch(file_path, args.pattern):
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
                    # Just check files in the top directory
                    for filename in os.listdir(path):
                        file_path = os.path.join(path, filename)
                        if os.path.isfile(file_path):
                            if args.pattern:
                                if glob.fnmatch.fnmatch(file_path, args.pattern):
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
                # It's a file, add it directly
                all_files.append(path)

        # Remove duplicates
        all_files = list(set(all_files))

        if not all_files:
            print("No files found to process!")
            return

        print(f"Found {len(all_files)} files to process")

        # Process all files
        results = process_files(all_files, max_workers=args.threads)

        # Print individual results
        for item in results:
            print_results(item)

        # Generate summary if requested
        if args.summary or args.summary_file:
            print_summary(results)

            if args.summary_file:
                generate_summary_report(results, args.summary_file)
                print(f"Summary report saved to: {args.summary_file}")
    else:
        parser.print_help()

    # Print elapsed time at the end
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    hours, minutes = divmod(minutes, 60)

    if hours > 0:
        print(f"\nTotal processing time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    elif minutes > 0:
        print(f"\nTotal processing time: {int(minutes)}m {seconds:.2f}s")
    else:
        print(f"\nTotal processing time: {seconds:.2f}s")

if __name__ == "__main__":
    main()