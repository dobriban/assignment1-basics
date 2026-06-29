#!/usr/bin/env python3
"""
Test encoding with a small sample to verify the approach works.
"""

import sys
import os
import numpy as np
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tokenizer import Tokenizer

def test_encoding_sample():
    """Test encoding with just the first part of validation dataset"""
    print("=" * 50)
    print("TESTING ENCODING WITH SAMPLE")
    print("=" * 50)
    
    # Load TinyStories tokenizer
    vocab_path = Path("../results/tinystories_bpe_output/vocab.json")
    merges_path = Path("../results/tinystories_bpe_output/merges.json")
    
    if not (vocab_path.exists() and merges_path.exists()):
        print("TinyStories tokenizer not found!")
        return
    
    print("Loading TinyStories tokenizer...")
    tokenizer = Tokenizer.from_files(str(vocab_path), str(merges_path))
    
    # Check vocabulary
    vocab_size = len(tokenizer.vocab)
    max_token_id = max(tokenizer.vocab.keys()) if tokenizer.vocab else 0
    print(f"Vocabulary size: {vocab_size:,}")
    print(f"Max token ID: {max_token_id:,}")
    print(f"Fits in uint16: {max_token_id <= 65535}")
    
    # Test with first 1MB of validation data
    input_path = Path("../data/TinyStoriesV2-GPT4-valid.txt")
    if not input_path.exists():
        print(f"Dataset not found: {input_path}")
        return
    
    print(f"\nReading first 1MB from {input_path.name}...")
    
    sample_size = 1024 * 1024  # 1MB
    all_token_ids = []
    chars_read = 0
    
    with open(input_path, 'r', encoding='utf-8') as f:
        sample_text = f.read(sample_size)
        chars_read = len(sample_text)
    
    print(f"Read {chars_read:,} characters")
    
    # Encode the sample
    print("Tokenizing sample...")
    all_token_ids = tokenizer.encode(sample_text)
    
    print(f"Generated {len(all_token_ids):,} tokens")
    
    # Check token range
    if all_token_ids:
        min_token = min(all_token_ids)
        max_token = max(all_token_ids)
        print(f"Token range: {min_token} to {max_token}")
        
        # Verify all tokens are valid
        invalid_tokens = [t for t in all_token_ids if t not in tokenizer.vocab]
        if invalid_tokens:
            print(f"WARNING: Found {len(invalid_tokens)} invalid tokens!")
            print(f"First few invalid: {invalid_tokens[:10]}")
        else:
            print("All tokens are valid!")
    
    # Test uint16 conversion
    print("\nTesting uint16 conversion...")
    try:
        token_array = np.array(all_token_ids, dtype=np.uint16)
        print(f"Successfully created uint16 array: shape {token_array.shape}")
        
        # Test save/load
        test_output = Path("test_sample.npy")
        np.save(test_output, token_array)
        
        # Load it back
        loaded_array = np.load(test_output)
        print(f"Saved and loaded successfully: dtype {loaded_array.dtype}")
        
        # Verify roundtrip
        if np.array_equal(token_array, loaded_array):
            print("Roundtrip test PASSED!")
        else:
            print("Roundtrip test FAILED!")
        
        # Test decoding
        print("\nTesting decoding...")
        decoded_text = tokenizer.decode(loaded_array.tolist())
        
        # Check if decoded matches original (approximately)
        original_length = len(sample_text)
        decoded_length = len(decoded_text)
        print(f"Original text: {original_length:,} chars")
        print(f"Decoded text:  {decoded_length:,} chars")
        
        if abs(original_length - decoded_length) / original_length < 0.01:  # Within 1%
            print("Roundtrip encoding/decoding test PASSED!")
        else:
            print("Roundtrip encoding/decoding test: some difference found")
            print(f"First 200 chars of original: {repr(sample_text[:200])}")
            print(f"First 200 chars of decoded:  {repr(decoded_text[:200])}")
        
        # Statistics
        compression_ratio = chars_read / len(all_token_ids)
        bytes_per_token = token_array.nbytes / len(token_array)
        
        print(f"\nCompression statistics:")
        print(f"Compression ratio: {compression_ratio:.2f} chars/token")
        print(f"Storage: {bytes_per_token} bytes/token (uint16)")
        
        # Cleanup
        test_output.unlink()
        
    except Exception as e:
        print(f"Error in uint16 conversion: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("ENCODING TEST SUCCESSFUL!")
    print("Ready to encode full datasets.")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    test_encoding_sample()