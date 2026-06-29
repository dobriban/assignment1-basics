#!/usr/bin/env python3
"""
Train BPE tokenizer on a smaller sample of TinyStories dataset for testing.
"""
import time
import pickle
import json
import os
from pathlib import Path

from tests.adapters import run_train_bpe

def main():
    # Use smaller sample file for faster testing
    input_path = "tests/fixtures/tinystories_sample.txt"
    vocab_size = 1000  # Smaller vocab for testing
    special_tokens = ["<|endoftext|>"]
    
    print(f"Training BPE tokenizer on {input_path} (sample)")
    print(f"Vocabulary size: {vocab_size:,}")
    print(f"Special tokens: {special_tokens}")
    print("-" * 50)
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found!")
        return
    
    # Get file size for reference
    file_size_mb = os.path.getsize(input_path) / (1024 * 1024)
    print(f"Input file size: {file_size_mb:.3f} MB")
    
    # Train BPE tokenizer with timing
    start_time = time.time()
    print("Starting BPE training on sample...")
    
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens
    )
    
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"Final vocabulary size: {len(vocab):,}")
    print(f"Number of merges: {len(merges):,}")
    
    # Save vocabulary and merges to disk
    output_dir = Path("tinystories_bpe_sample_output")
    output_dir.mkdir(exist_ok=True)
    
    # Save as pickle files
    vocab_path = output_dir / "vocab.pkl"
    merges_path = output_dir / "merges.pkl"
    
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print(f"Vocabulary saved to {vocab_path}")
    
    with open(merges_path, 'wb') as f:
        pickle.dump(merges, f)
    print(f"Merges saved to {merges_path}")
    
    # Also save in human-readable JSON format for inspection
    vocab_json = {}
    for token_id, token_bytes in vocab.items():
        try:
            # Try to decode as UTF-8 for readability
            vocab_json[str(token_id)] = token_bytes.decode('utf-8', errors='replace')
        except:
            # If decoding fails, use repr
            vocab_json[str(token_id)] = repr(token_bytes)
    
    with open(output_dir / "vocab.json", 'w', encoding='utf-8') as f:
        json.dump(vocab_json, f, indent=2, ensure_ascii=False)
    
    merges_json = []
    for merge in merges:
        try:
            merge_readable = (
                merge[0].decode('utf-8', errors='replace'),
                merge[1].decode('utf-8', errors='replace')
            )
        except:
            merge_readable = (repr(merge[0]), repr(merge[1]))
        merges_json.append(merge_readable)
    
    with open(output_dir / "merges.json", 'w', encoding='utf-8') as f:
        json.dump(merges_json, f, indent=2, ensure_ascii=False)
    
    print(f"Human-readable files saved to {output_dir}/")
    
    # Find the longest token
    longest_token = max(vocab.values(), key=len)
    longest_length = len(longest_token)
    longest_token_readable = longest_token.decode('utf-8', errors='replace')
    
    print(f"\nLongest token:")
    print(f"  Length: {longest_length} bytes")
    print(f"  Content: {repr(longest_token_readable)}")
    
    # Find all tokens of maximum length
    max_length_tokens = [(tid, token) for tid, token in vocab.items() if len(token) == longest_length]
    if len(max_length_tokens) > 1:
        print(f"  Note: {len(max_length_tokens)} tokens share this maximum length")
    
    print(f"\nSample training summary:")
    print(f"  Time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"  Input size: {file_size_mb:.3f} MB")
    print(f"  Vocabulary size: {len(vocab):,}")
    print(f"  Longest token: {longest_length} bytes")

if __name__ == "__main__":
    main()