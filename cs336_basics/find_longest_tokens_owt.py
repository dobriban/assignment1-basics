#!/usr/bin/env python3
"""
Find the longest tokens in the OpenWebText BPE vocabulary.
"""
import pickle
from pathlib import Path

def main():
    # Load the vocabulary from OpenWebText training
    vocab_path = Path("owt_bpe_output/vocab.pkl")
    
    if not vocab_path.exists():
        print(f"Vocabulary file {vocab_path} not found!")
        print("Please run train_bpe_owt.py first to train the tokenizer.")
        return
    
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    print(f"Loaded OpenWebText BPE vocabulary with {len(vocab):,} tokens")
    
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
    print("=" * 80)
    
    for i, (token_id, token_bytes, token_text) in enumerate(sorted(max_length_tokens, key=lambda x: x[2]), 1):
        print(f"{i}. Token ID {token_id}: {repr(token_text)}")
        print(f"   Raw bytes: {token_bytes}")
        print(f"   Length: {len(token_bytes)} bytes")
        print()
    
    # Provide analysis
    print("ANALYSIS OF THE LONGEST TOKEN:")
    print("=" * 80)
    if max_length_tokens:
        _, token_bytes, token_text = max_length_tokens[0]
        
        print(f"Longest token: {repr(token_text)}")
        print(f"Byte length: {max_length}")
        
        # Character analysis
        if isinstance(token_text, str):
            char_count = len(token_text)
            print(f"Character count: {char_count}")
            
            # Check for common patterns
            has_spaces = ' ' in token_text
            starts_with_space = token_text.startswith(' ')
            is_alphanumeric = token_text.replace(' ', '').isalnum()
            has_punctuation = any(not c.isalnum() and not c.isspace() for c in token_text)
            
            print(f"Contains spaces: {has_spaces}")
            print(f"Starts with space: {starts_with_space}")
            print(f"Alphanumeric (ignoring spaces): {is_alphanumeric}")
            print(f"Contains punctuation: {has_punctuation}")
            
            # Frequency analysis context
            print(f"\nThis token likely represents a frequently occurring sequence in")
            print(f"the OpenWebText dataset. Long tokens in BPE typically emerge when:")
            print(f"1. A phrase or word appears very frequently")
            print(f"2. The token improves compression efficiency")
            print(f"3. It represents a meaningful linguistic or domain-specific unit")
            
            if starts_with_space:
                print(f"\nThe leading space follows BPE's convention for word boundaries.")
            
            if has_spaces and char_count > 10:
                print(f"This appears to be a multi-word phrase that occurs frequently")
                print(f"enough in web text to warrant its own token.")
            
            print(f"\nWhether this makes sense depends on:")
            print(f"- Frequency in the training corpus")
            print(f"- Semantic coherence of the phrase")
            print(f"- Tokenization efficiency benefits")

if __name__ == "__main__":
    main()