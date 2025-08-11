#!/usr/bin/env python3
"""
Fast encoding for TinyStories datasets with better progress tracking.
"""

import sys
import os
import numpy as np
from pathlib import Path
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tokenizer import Tokenizer

def fast_encode_dataset(tokenizer, input_path, output_path, chunk_size_kb=100):
    """
    Fast encoding with smaller chunks and progress updates.
    """
    print(f"Fast encoding: {input_path.name}")
    
    file_size = os.path.getsize(input_path)
    print(f"File size: {file_size / (1024*1024):.1f} MB")
    
    all_tokens = []
    bytes_processed = 0
    chunk_size = chunk_size_kb * 1024  # Convert KB to bytes
    
    start_time = time.time()
    last_update = start_time
    
    with open(input_path, 'r', encoding='utf-8') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            
            # Tokenize chunk directly
            chunk_tokens = tokenizer.encode(chunk)
            all_tokens.extend(chunk_tokens)
            
            bytes_processed += len(chunk.encode('utf-8'))
            
            # Progress update every 5 seconds
            current_time = time.time()
            if current_time - last_update > 5:
                progress = (bytes_processed / file_size) * 100
                elapsed = current_time - start_time
                rate = bytes_processed / (1024 * 1024) / elapsed  # MB/s
                
                print(f"Progress: {progress:.1f}% ({bytes_processed/(1024*1024):.1f}/{file_size/(1024*1024):.1f} MB) "
                      f"Rate: {rate:.1f} MB/s, Tokens: {len(all_tokens):,}")
                last_update = current_time
    
    print(f"Tokenization complete: {len(all_tokens):,} tokens")
    
    # Convert to numpy array
    print("Converting to uint16 array...")
    token_array = np.array(all_tokens, dtype=np.uint16)
    
    # Save
    print(f"Saving to {output_path.name}...")
    np.save(output_path, token_array)
    
    # Final stats
    total_time = time.time() - start_time
    output_size = os.path.getsize(output_path)
    
    print(f"Complete! {len(token_array):,} tokens saved")
    print(f"Output size: {output_size/(1024*1024):.1f} MB")
    print(f"Time taken: {total_time:.1f} seconds")
    print(f"Compression: {bytes_processed / len(all_tokens):.2f} bytes/token")
    
    return token_array

def main():
    print("FAST TINYSTORIES ENCODING")
    print("=" * 40)
    
    # Output dir
    output_dir = Path("../encoded_datasets")
    output_dir.mkdir(exist_ok=True)
    
    # Load tokenizer
    vocab_path = Path("../results/tinystories_bpe_output/vocab.json")
    merges_path = Path("../results/tinystories_bpe_output/merges.json")
    
    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_files(str(vocab_path), str(merges_path))
    print(f"Vocab size: {len(tokenizer.vocab)}")
    
    # Start with validation (smaller)
    datasets = [
        ("validation", "../data/TinyStoriesV2-GPT4-valid.txt", "tinystories_valid_uint16.npy"),
        ("training", "../data/TinyStoriesV2-GPT4-train.txt", "tinystories_train_uint16.npy"),
    ]
    
    for name, input_file, output_file in datasets:
        print(f"\n{name.upper()}:")
        print("-" * 20)
        
        input_path = Path(input_file)
        output_path = output_dir / output_file
        
        if not input_path.exists():
            print(f"File not found: {input_path}")
            continue
            
        if output_path.exists():
            print(f"Already encoded: {output_path.name}")
            arr = np.load(output_path)
            print(f"  {len(arr):,} tokens, {os.path.getsize(output_path)/(1024*1024):.1f} MB")
            continue
        
        try:
            fast_encode_dataset(tokenizer, input_path, output_path)
        except KeyboardInterrupt:
            print("Interrupted by user")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

if __name__ == "__main__":
    main()