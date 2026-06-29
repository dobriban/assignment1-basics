#!/usr/bin/env python3
"""
Encode TinyStories datasets into sequences of integer token IDs.
Saves as NumPy arrays with uint16 datatype for efficient storage.
"""

import sys
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tokenizer import Tokenizer

def encode_tinystories_dataset(tokenizer, input_path, output_path):
    """
    Encode a TinyStories dataset efficiently.
    
    Args:
        tokenizer: Tokenizer instance
        input_path: Path to input text file
        output_path: Path to save encoded NumPy array
    """
    print(f"Encoding {input_path.name} -> {output_path.name}")
    
    file_size = os.path.getsize(input_path)
    print(f"File size: {file_size / (1024*1024):.1f} MB")
    
    all_token_ids = []
    chars_processed = 0
    
    with open(input_path, 'r', encoding='utf-8') as f:
        with tqdm(total=file_size, unit='B', unit_scale=True, desc="Encoding") as pbar:
            
            # Process in chunks for memory efficiency
            chunk_size = 1024 * 1024  # 1MB chunks
            buffer = ""
            
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                
                buffer += chunk
                chars_processed += len(chunk)
                pbar.update(len(chunk.encode('utf-8')))
                
                # Process complete lines to avoid cutting in middle of sentences
                lines = buffer.split('\n')
                buffer = lines[-1]  # Keep incomplete line
                
                for line in lines[:-1]:
                    if line.strip():  # Skip empty lines
                        tokens = tokenizer.encode(line + '\n')
                        all_token_ids.extend(tokens)
            
            # Process remaining buffer
            if buffer.strip():
                tokens = tokenizer.encode(buffer)
                all_token_ids.extend(tokens)
    
    print(f"Tokenized {chars_processed:,} characters into {len(all_token_ids):,} tokens")
    
    # Check if tokens fit in uint16
    max_token_id = max(all_token_ids) if all_token_ids else 0
    max_uint16 = 2**16 - 1  # 65535
    
    if max_token_id > max_uint16:
        print(f"WARNING: Max token ID ({max_token_id}) exceeds uint16 range!")
        print("Using uint32 instead...")
        dtype = np.uint32
        output_path = str(output_path).replace('uint16', 'uint32')
    else:
        print(f"Max token ID: {max_token_id} (fits in uint16)")
        dtype = np.uint16
    
    # Convert to NumPy array
    print("Converting to NumPy array...")
    token_array = np.array(all_token_ids, dtype=dtype)
    
    # Save array
    np.save(output_path, token_array)
    
    # Statistics
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    compression_ratio = chars_processed / len(token_array) if token_array.size > 0 else 0
    
    print(f"Saved to {output_path}")
    print(f"Array shape: {token_array.shape}")
    print(f"Array dtype: {token_array.dtype}")
    print(f"Output file size: {file_size_mb:.1f} MB")
    print(f"Compression ratio: {compression_ratio:.2f} chars/token")
    
    return token_array

def main():
    """Encode TinyStories datasets only"""
    print("=" * 60)
    print("ENCODING TINYSTORIES DATASETS")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path("../encoded_datasets")
    output_dir.mkdir(exist_ok=True)
    
    # Load TinyStories tokenizer
    vocab_path = Path("../results/tinystories_bpe_output/vocab.json")
    merges_path = Path("../results/tinystories_bpe_output/merges.json")
    
    if not (vocab_path.exists() and merges_path.exists()):
        print("TinyStories tokenizer not found!")
        print(f"Expected: {vocab_path}")
        print(f"Expected: {merges_path}")
        return
    
    print("Loading TinyStories tokenizer...")
    tokenizer = Tokenizer.from_files(str(vocab_path), str(merges_path))
    
    # Check vocabulary size
    max_token_id = max(tokenizer.vocab.keys()) if tokenizer.vocab else 0
    vocab_size = len(tokenizer.vocab)
    print(f"Vocabulary size: {vocab_size:,}")
    print(f"Max token ID: {max_token_id:,}")
    print(f"Fits in uint16: {max_token_id <= 65535}")
    
    # TinyStories datasets to encode
    datasets = [
        {
            "name": "TinyStories Validation",
            "input": Path("../data/TinyStoriesV2-GPT4-valid.txt"),
            "output": output_dir / "tinystories_valid_uint16.npy"
        },
        {
            "name": "TinyStories Training",
            "input": Path("../data/TinyStoriesV2-GPT4-train.txt"), 
            "output": output_dir / "tinystories_train_uint16.npy"
        }
    ]
    
    for dataset in datasets:
        print(f"\n" + "=" * 40)
        print(f"Processing: {dataset['name']}")
        print("=" * 40)
        
        if not dataset["input"].exists():
            print(f"Dataset not found: {dataset['input']}")
            continue
        
        if dataset["output"].exists():
            print(f"Already encoded: {dataset['output'].name}")
            # Show existing file info
            try:
                existing_array = np.load(dataset["output"], mmap_mode='r')
                size_mb = os.path.getsize(dataset["output"]) / (1024*1024)
                print(f"  {len(existing_array):,} tokens, {size_mb:.1f} MB, dtype: {existing_array.dtype}")
            except Exception as e:
                print(f"  Error reading existing file: {e}")
            continue
        
        try:
            encode_tinystories_dataset(tokenizer, dataset["input"], dataset["output"])
        except Exception as e:
            print(f"Error encoding {dataset['name']}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n" + "=" * 60)
    print("TINYSTORIES ENCODING COMPLETE")
    print("=" * 60)
    
    # List all TinyStories encoded files
    encoded_files = list(output_dir.glob("tinystories_*.npy"))
    if encoded_files:
        print("\nEncoded TinyStories datasets:")
        total_tokens = 0
        for file_path in sorted(encoded_files):
            try:
                array = np.load(file_path, mmap_mode='r')
                size_mb = os.path.getsize(file_path) / (1024*1024)
                print(f"  {file_path.name}")
                print(f"    {len(array):,} tokens, {size_mb:.1f} MB, dtype: {array.dtype}")
                total_tokens += len(array)
            except Exception as e:
                print(f"  {file_path.name}: Error reading file - {e}")
        print(f"\nTotal TinyStories tokens: {total_tokens:,}")
        
        # Calculate storage efficiency
        if total_tokens > 0:
            total_size_mb = sum(os.path.getsize(f) / (1024*1024) for f in encoded_files)
            bytes_per_token = (total_size_mb * 1024 * 1024) / total_tokens
            print(f"Storage efficiency: {bytes_per_token:.2f} bytes/token")
    else:
        print("\nNo TinyStories datasets were encoded.")

if __name__ == "__main__":
    main()