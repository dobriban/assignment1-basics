#!/usr/bin/env python3
"""
Train a byte-level BPE tokenizer on the OpenWebText dataset.
"""
import json
import pickle
from pathlib import Path
from .train_bpe import train_bpe

def main():
    # Configuration
    vocab_size = 32000
    special_tokens = ["<|endoftext|>"]  # Common special token for OpenWebText
    
    # Input path for OpenWebText dataset
    input_path = "data/owt_train.txt"  # Training data from OpenWebText sample
    
    # Output directory
    output_dir = Path("owt_bpe_output")
    output_dir.mkdir(exist_ok=True)
    
    print(f"Training BPE tokenizer on OpenWebText dataset...")
    print(f"Vocabulary size: {vocab_size:,}")
    print(f"Special tokens: {special_tokens}")
    print(f"Input file: {input_path}")
    print(f"Output directory: {output_dir}")
    
    # Check if input file exists
    if not Path(input_path).exists():
        print(f"Error: Input file {input_path} not found!")
        print("Please ensure the OpenWebText dataset is available at this path.")
        return
    
    # Train the BPE tokenizer
    print("\nStarting BPE training...")
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    
    print(f"Training complete! Generated {len(vocab):,} tokens and {len(merges):,} merges.")
    
    # Save vocabulary as JSON (human readable)
    vocab_json_path = output_dir / "vocab.json"
    vocab_for_json = {str(k): v.decode('utf-8', errors='replace') for k, v in vocab.items()}
    with open(vocab_json_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_for_json, f, indent=2, ensure_ascii=False)
    print(f"Saved vocabulary to {vocab_json_path}")
    
    # Save vocabulary as pickle (for exact byte preservation)
    vocab_pkl_path = output_dir / "vocab.pkl"
    with open(vocab_pkl_path, 'wb') as f:
        pickle.dump(vocab, f)
    print(f"Saved vocabulary (pickle) to {vocab_pkl_path}")
    
    # Save merges as JSON (human readable)
    merges_json_path = output_dir / "merges.json"
    merges_for_json = [[m[0].decode('utf-8', errors='replace'), m[1].decode('utf-8', errors='replace')] for m in merges]
    with open(merges_json_path, 'w', encoding='utf-8') as f:
        json.dump(merges_for_json, f, indent=2, ensure_ascii=False)
    print(f"Saved merges to {merges_json_path}")
    
    # Save merges as pickle (for exact byte preservation)
    merges_pkl_path = output_dir / "merges.pkl"
    with open(merges_pkl_path, 'wb') as f:
        pickle.dump(merges, f)
    print(f"Saved merges (pickle) to {merges_pkl_path}")
    
    # Find and display the longest token
    print("\n" + "="*60)
    print("ANALYZING LONGEST TOKENS")
    print("="*60)
    
    max_length = max(len(token_bytes) for token_bytes in vocab.values())
    print(f"Maximum token length: {max_length} bytes")
    
    longest_tokens = []
    for token_id, token_bytes in vocab.items():
        if len(token_bytes) == max_length:
            try:
                token_text = token_bytes.decode('utf-8', errors='replace')
                longest_tokens.append((token_id, token_bytes, token_text))
            except Exception as e:
                longest_tokens.append((token_id, token_bytes, f"<decode error: {e}>"))
    
    print(f"\nFound {len(longest_tokens)} tokens with maximum length ({max_length} bytes):")
    print("-" * 60)
    
    for i, (token_id, token_bytes, token_text) in enumerate(sorted(longest_tokens, key=lambda x: x[2]), 1):
        print(f"{i}. Token ID {token_id}: {repr(token_text)}")
        print(f"   Raw bytes: {token_bytes}")
        print(f"   Length: {len(token_bytes)} bytes")
        print()
    
    # Analysis of whether the longest token makes sense
    print("ANALYSIS:")
    print("-" * 40)
    if longest_tokens:
        _, _, first_token = longest_tokens[0]
        print(f"The longest token is: {repr(first_token)}")
        print(f"Length: {max_length} bytes")
        
        # Basic analysis
        if max_length > 20:
            print("This token is quite long, which suggests it represents a frequently occurring")
            print("multi-word phrase or compound word that appears often in the OpenWebText dataset.")
        
        if any(char.isspace() for char in first_token if isinstance(first_token, str)):
            print("The token contains whitespace, indicating it captures phrase-level patterns.")
        
        if first_token.startswith(' '):
            print("The token starts with a space, following BPE's space-prefix convention.")
        
        print("\nWhether this makes sense depends on:")
        print("1. If it's a common phrase/pattern in web text")
        print("2. If it helps with compression/tokenization efficiency")
        print("3. If it represents meaningful linguistic units")

if __name__ == "__main__":
    main()