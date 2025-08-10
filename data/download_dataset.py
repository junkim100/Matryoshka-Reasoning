#!/usr/bin/env python3
"""
Dataset download script for processing HuggingFace datasets with tokenization and filtering.

Usage:
    python data/download_dataset.py --dataset GAIR/LIMO --max_token_length 4096
    python data/download_dataset.py --dataset GAIR/LIMO --max_num 1000 --shuffle True
"""

import os
import json
import random
from typing import Optional, Union
import fire
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import jsonlines


def download_and_process_dataset(
    dataset: str = "GAIR/LIMO",
    max_token_length: int = 4096,
    min_token_length: int = 0,
    tokenizer: str = "meta-llama/Llama-3.1-8B-Instruct",
    shuffle: bool = True,
    max_num: Optional[int] = None,
    output_file: Optional[str] = None,
    split: str = "train",
    create_train_val_split: bool = True,
    train_ratio: float = 0.9,
):
    """
    Download and process a HuggingFace dataset with tokenization and filtering.

    Args:
        dataset: HuggingFace dataset name (default: "GAIR/LIMO")
        max_token_length: Maximum token length for filtering (default: 4096)
        min_token_length: Minimum token length for filtering (default: 0)
        tokenizer: Tokenizer model name (default: "meta-llama/Llama-3.1-8B-Instruct")
        shuffle: Whether to shuffle the dataset (default: True)
        max_num: Maximum number of samples to keep (default: None for all)
        output_file: Output file path (default: auto-generated)
        split: Dataset split to use (default: "train")
        create_train_val_split: Whether to create train/val split (default: True)
        train_ratio: Ratio for train split when creating train/val split (default: 0.9)
    """

    print(f"Loading dataset: {dataset}")
    print(f"Split: {split}")
    print(f"Tokenizer: {tokenizer}")
    print(f"Token length range: [{min_token_length}, {max_token_length}]")
    print(f"Shuffle: {shuffle}")
    print(f"Max samples: {max_num if max_num else 'all'}")
    print(f"Create train/val split: {create_train_val_split}")
    if create_train_val_split:
        print(f"Train ratio: {train_ratio} (val ratio: {1-train_ratio})")

    # Load the dataset
    try:
        ds = load_dataset(dataset, split=split)
        print(f"Loaded {len(ds)} samples from {dataset}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Load tokenizer
    try:
        tok = AutoTokenizer.from_pretrained(tokenizer)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        print(f"Loaded tokenizer: {tokenizer}")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # Convert to list for processing
    data_list = list(ds)

    # Shuffle if requested
    if shuffle:
        random.shuffle(data_list)
        print("Dataset shuffled")

    # Filter by token length
    filtered_data = []
    print("Filtering by token length...")

    for item in tqdm(data_list, desc="Processing samples"):
        # Determine text content to tokenize
        # This assumes the dataset has common text fields
        text_content = ""

        # Try common field names for text content
        if isinstance(item, dict):
            # Common text fields in datasets
            text_fields = [
                "text",
                "content",
                "input",
                "prompt",
                "question",
                "instruction",
            ]

            # If it's a conversation/chat format
            if "messages" in item:
                # Handle chat format
                messages = item["messages"]
                if isinstance(messages, list):
                    text_content = " ".join(
                        [
                            msg.get("content", "")
                            for msg in messages
                            if isinstance(msg, dict)
                        ]
                    )
                else:
                    text_content = str(messages)
            elif "conversations" in item:
                # Handle conversation format
                conversations = item["conversations"]
                if isinstance(conversations, list):
                    text_content = " ".join(
                        [
                            conv.get("value", "")
                            for conv in conversations
                            if isinstance(conv, dict)
                        ]
                    )
                else:
                    text_content = str(conversations)
            else:
                # Try to find text fields
                for field in text_fields:
                    if field in item and item[field]:
                        text_content += str(item[field]) + " "

                # If no common fields found, concatenate all string values
                if not text_content.strip():
                    text_content = " ".join(
                        [str(v) for v in item.values() if isinstance(v, str)]
                    )
        else:
            text_content = str(item)

        # Tokenize and check length
        if text_content.strip():
            try:
                tokens = tok.encode(text_content, add_special_tokens=True)
                token_length = len(tokens)

                if min_token_length <= token_length <= max_token_length:
                    # Add token length info to the item
                    if isinstance(item, dict):
                        item["token_length"] = token_length
                    filtered_data.append(item)
            except Exception as e:
                print(f"Error tokenizing sample: {e}")
                continue

    print(f"Filtered to {len(filtered_data)} samples (from {len(data_list)})")

    # Apply max_num limit
    if max_num is not None and len(filtered_data) > max_num:
        if shuffle:
            # Already shuffled, just take first max_num
            filtered_data = filtered_data[:max_num]
        else:
            # Sort by token length (shortest first) and take max_num
            filtered_data.sort(key=lambda x: x.get("token_length", 0))
            filtered_data = filtered_data[:max_num]
        print(f"Limited to {max_num} samples")

    # Split into train and validation sets if requested
    if create_train_val_split:
        # Ensure data is shuffled for proper split
        if not shuffle:
            random.shuffle(filtered_data)
            print("Data shuffled for train/val split")

        # Calculate split point
        train_size = int(len(filtered_data) * train_ratio)
        train_data = filtered_data[:train_size]
        val_data = filtered_data[train_size:]

        print(
            f"Split into {len(train_data)} train and {len(val_data)} validation samples"
        )

        # Generate output filenames if not provided
        if output_file is None:
            dataset_name = dataset.replace("/", "_").replace("-", "_")
            suffix = f"_max{max_token_length}_min{min_token_length}"
            if max_num:
                suffix += f"_n{max_num}"
            if shuffle:
                suffix += "_shuffled"
            train_file = f"data/{dataset_name}{suffix}_train.jsonl"
            val_file = f"data/{dataset_name}{suffix}_val.jsonl"
        else:
            # Use provided output_file as base for train/val files
            base_name = output_file.rsplit(".", 1)[0]
            extension = output_file.rsplit(".", 1)[1] if "." in output_file else "jsonl"
            train_file = f"{base_name}_train.{extension}"
            val_file = f"{base_name}_val.{extension}"

        # Save train and validation sets
        for data_split, filename, split_name in [
            (train_data, train_file, "train"),
            (val_data, val_file, "validation"),
        ]:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            print(f"Saving {split_name} set to: {filename}")
            with jsonlines.open(filename, "w") as writer:
                for item in data_split:
                    writer.write(item)

            print(
                f"Successfully saved {len(data_split)} {split_name} samples to {filename}"
            )

            # Print statistics for this split
            if (
                data_split
                and isinstance(data_split[0], dict)
                and "token_length" in data_split[0]
            ):
                token_lengths = [item["token_length"] for item in data_split]
                print(f"{split_name.capitalize()} token length statistics:")
                print(f"  Min: {min(token_lengths)}")
                print(f"  Max: {max(token_lengths)}")
                print(f"  Mean: {sum(token_lengths) / len(token_lengths):.1f}")

        return  # Exit early since we've saved train/val splits

    # Generate output filename if not provided (for single file output)
    if output_file is None:
        dataset_name = dataset.replace("/", "_").replace("-", "_")
        suffix = f"_max{max_token_length}_min{min_token_length}"
        if max_num:
            suffix += f"_n{max_num}"
        if shuffle:
            suffix += "_shuffled"
        output_file = f"data/{dataset_name}{suffix}.jsonl"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save the processed dataset
    print(f"Saving to: {output_file}")
    with jsonlines.open(output_file, "w") as writer:
        for item in filtered_data:
            writer.write(item)

    print(f"Successfully saved {len(filtered_data)} samples to {output_file}")

    # Print some statistics
    if (
        filtered_data
        and isinstance(filtered_data[0], dict)
        and "token_length" in filtered_data[0]
    ):
        token_lengths = [item["token_length"] for item in filtered_data]
        print(f"Token length statistics:")
        print(f"  Min: {min(token_lengths)}")
        print(f"  Max: {max(token_lengths)}")
        print(f"  Mean: {sum(token_lengths) / len(token_lengths):.1f}")


if __name__ == "__main__":
    fire.Fire(download_and_process_dataset)
