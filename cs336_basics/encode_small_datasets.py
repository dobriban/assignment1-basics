#!/usr/bin/env python3
"""
Encode smaller datasets first (validation sets) to test the process.
"""

import sys
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tokenizer import Tokenizer

def encode_small_dataset(tokenizer, input_path, output_path, max_chars=None):
    """
    Encode a smaller dataset efficiently.
    
    Args:
        tokenizer: Tokenizer instance
        input_path: Path to input text file
        output_path: Path to save encoded NumPy array
        max_chars: Optional limit on characters to process (for testing)
    """
    print(f"Encoding {input_path.name} -> {output_path.name}")
    
    # Read file in chunks to handle encoding
    all_token_ids = []
    total_chars = 0
    chars_processed = 0
    
    file_size = os.path.getsize(input_path)
    print(f"File size: {file_size / (1024*1024):.1f} MB")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        with tqdm(total=file_size, unit='B', unit_scale=True, desc="Reading") as pbar:
            
            # Process in chunks for memory efficiency
            chunk_size = 1024 * 1024  # 1MB chunks
            buffer = ""
            
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                
                if max_chars and chars_processed + len(chunk) > max_chars:
                    # Truncate chunk if we're near the limit
                    remaining = max_chars - chars_processed
                    chunk = chunk[:remaining]
                
                buffer += chunk
                chars_processed += len(chunk)
                pbar.update(len(chunk.encode('utf-8')))
                
                # Process complete lines to avoid cutting in middle of sentences
                lines = buffer.split('\\n')
                buffer = lines[-1]  # Keep incomplete line
                
                for line in lines[:-1]:
                    if line.strip():  # Skip empty lines
                        tokens = tokenizer.encode(line + '\\n')
                        all_token_ids.extend(tokens)
                
                if max_chars and chars_processed >= max_chars:
                    print(f"Reached character limit: {max_chars:,}")
                    break
            
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
        output_path = str(output_path).replace('.npy', '_uint32.npy')
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
    """Encode validation datasets first (they're smaller)"""
    print("=" * 60)
    print("ENCODING VALIDATION DATASETS (SMALL FILES)")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path("../encoded_datasets")
    output_dir.mkdir(exist_ok=True)
    
    # Load TinyStories tokenizer
    vocab_path = Path("../results/tinystories_bpe_output/vocab.json")
    merges_path = Path("../results/tinystories_bpe_output/merges.json")
    
    if not (vocab_path.exists() and merges_path.exists()):
        print("TinyStories tokenizer not found!")
        print("Please train the tokenizer first.")
        return
    
    print("Loading TinyStories tokenizer...")
    tokenizer = Tokenizer.from_files(str(vocab_path), str(merges_path))
    
    # Check vocabulary size
    max_token_id = max(tokenizer.vocab.keys()) if tokenizer.vocab else 0
    vocab_size = len(tokenizer.vocab)
    print(f"Vocabulary size: {vocab_size:,}")
    print(f"Max token ID: {max_token_id:,}")
    
    # Encode validation datasets (smaller files)
    datasets_to_encode = [
        {
            "name": "tinystories_valid",
            "input": Path("../data/TinyStoriesV2-GPT4-valid.txt"),
            "output": output_dir / "tinystories_valid_uint16.npy"
        },
        {
            "name": "openwebtext_valid", 
            "input": Path("../data/owt_valid.txt"),
            "output": output_dir / "openwebtext_valid_uint16.npy"
        }
    ]
    
    for dataset in datasets_to_encode:
        print(f"\\n" + "-" * 40)
        
        if not dataset["input"].exists():
            print(f"Dataset not found: {dataset['input']}")
            continue
        
        if dataset["output"].exists():
            print(f"Already encoded: {dataset['output'].name}")
            # Show existing file info
            existing_array = np.load(dataset["output"], mmap_mode='r')
            size_mb = os.path.getsize(dataset["output"]) / (1024*1024)
            print(f"  {len(existing_array):,} tokens, {size_mb:.1f} MB")
            continue
        
        try:
            encode_small_dataset(tokenizer, dataset["input"], dataset["output"])
        except Exception as e:
            print(f"Error encoding {dataset['name']}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\\n" + "=" * 60)
    print("VALIDATION ENCODING COMPLETE")
    print("=" * 60)
    
    # List encoded files
    encoded_files = list(output_dir.glob("*valid*.npy"))
    if encoded_files:
        print("\\nEncoded validation datasets:")
        total_tokens = 0
        for file_path in sorted(encoded_files):
            array = np.load(file_path, mmap_mode='r')
            size_mb = os.path.getsize(file_path) / (1024*1024)
            print(f"  {file_path.name}: {len(array):,} tokens ({size_mb:.1f} MB)")
            total_tokens += len(array)
        print(f"\\nTotal validation tokens: {total_tokens:,}")
    
    print(f"\\nNext: Run script for training datasets (much larger files)")

if __name__ == "__main__":
    main()