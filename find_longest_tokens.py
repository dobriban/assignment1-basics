#!/usr/bin/env python3
"""
Find the longest tokens in the BPE vocabulary.
"""
import pickle
from pathlib import Path

def main():
    # Load the vocabulary
    vocab_path = Path("tinystories_bpe_output/vocab.pkl")
    
    if not vocab_path.exists():
        print(f"Vocabulary file {vocab_path} not found!")
        return
    
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    print(f"Loaded vocabulary with {len(vocab):,} tokens")
    
    # Find the maximum token length
    max_length = max(len(token_bytes) for token_bytes in vocab.values())
    print(f"Maximum token length: {max_length} bytes")
    
    # Find all tokens with maximum length
    max_length_tokens = []
    for token_id, token_bytes in vocab.items():
        if len(token_bytes) == max_length:
            try:
                # Try to decode as UTF-8
                token_text = token_bytes.decode('utf-8', errors='replace')
                max_length_tokens.append((token_id, token_bytes, token_text))
            except Exception as e:
                max_length_tokens.append((token_id, token_bytes, f"<decode error: {e}>"))
    
    print(f"\nFound {len(max_length_tokens)} tokens with maximum length ({max_length} bytes):")
    print("-" * 60)
    
    for i, (token_id, token_bytes, token_text) in enumerate(sorted(max_length_tokens, key=lambda x: x[2]), 1):
        print(f"{i}. Token ID {token_id}: {repr(token_text)}")
        print(f"   Raw bytes: {token_bytes}")
        print(f"   Length: {len(token_bytes)} bytes")
        print()

if __name__ == "__main__":
    main()