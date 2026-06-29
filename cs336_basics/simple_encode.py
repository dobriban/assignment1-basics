#!/usr/bin/env python3
"""
Simple, direct encoding approach.
"""

import sys
import os
sys.path.insert(0, '..')
import numpy as np
from pathlib import Path

# Import our tokenizer
from cs336_basics.tokenizer import Tokenizer

def simple_encode(input_file, output_file, max_mb=None):
    """Simple encoding with optional size limit"""
    
    print(f"Loading tokenizer...")
    tokenizer = Tokenizer.from_files(
        '../results/tinystories_bpe_output/vocab.json', 
        '../results/tinystories_bpe_output/merges.json'
    )
    
    print(f"Reading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        if max_mb:
            # Read limited amount for testing
            text = f.read(max_mb * 1024 * 1024)
            print(f"Read first {max_mb} MB")
        else:
            text = f.read()
            print(f"Read {len(text)} characters")
    
    print("Encoding...")
    tokens = tokenizer.encode(text)
    print(f"Generated {len(tokens)} tokens")
    
    print("Converting to numpy array...")
    arr = np.array(tokens, dtype=np.uint16)
    
    print(f"Saving to {output_file}...")
    np.save(output_file, arr)
    
    size_mb = os.path.getsize(output_file) / (1024*1024)
    print(f"Saved: {len(arr)} tokens, {size_mb:.1f} MB")
    
    return arr

if __name__ == "__main__":
    output_dir = Path('../encoded_datasets')
    output_dir.mkdir(exist_ok=True)
    
    print("SIMPLE ENCODING")
    print("=" * 30)
    
    # Start with validation (smaller file)
    try:
        print("\nValidation dataset:")
        simple_encode(
            '../data/TinyStoriesV2-GPT4-valid.txt',
            '../encoded_datasets/tinystories_valid_uint16.npy'
        )
        
        print("\nTraining dataset:")
        simple_encode(
            '../data/TinyStoriesV2-GPT4-train.txt', 
            '../encoded_datasets/tinystories_train_uint16.npy'
        )
        
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()