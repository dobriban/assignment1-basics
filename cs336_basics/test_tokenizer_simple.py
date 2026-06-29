#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from tokenizer import Tokenizer

def test_simple_tokenizer():
    # Create a simple test vocab and merges
    vocab = {
        # Basic ASCII bytes
        **{i: bytes([i]) for i in range(256)},
        # Some merged tokens
        256: b'he',
        257: b'll',  
        258: b'o ',
        259: b'the'
    }
    
    merges = [
        (b'h', b'e'),    # creates 'he'
        (b'l', b'l'),    # creates 'll'
        (b'o', b' '),    # creates 'o '
        (b't', b'he'),   # creates 'the'
    ]
    
    tokenizer = Tokenizer(vocab, merges)
    
    # Test empty string
    result = tokenizer.encode('')
    assert result == [], f"Expected empty list, got {result}"
    
    result = tokenizer.decode([])
    assert result == '', f"Expected empty string, got '{result}'"
    
    # Test single character
    result = tokenizer.encode('a')
    expected = [ord('a')]  # Should be the byte value for 'a'
    assert result == expected, f"Expected {expected}, got {result}"
    
    result = tokenizer.decode([ord('a')])
    assert result == 'a', f"Expected 'a', got '{result}'"
    
    # Test simple word that should use merges
    result = tokenizer.encode('hello')
    decoded = tokenizer.decode(result)
    assert decoded == 'hello', f"Roundtrip failed: 'hello' -> {result} -> '{decoded}'"
    
    print("All simple tests passed!")
    
    # Test special tokens
    tokenizer_with_special = Tokenizer(vocab, merges, ['<START>', '<END>'])
    
    result = tokenizer_with_special.encode('hello <START> world <END>')
    decoded = tokenizer_with_special.decode(result)
    assert decoded == 'hello <START> world <END>', f"Special token test failed: got '{decoded}'"
    
    print("Special token tests passed!")
    
    # Test from_files method with dummy files
    print("Creating test files...")
    import json
    
    # Create test vocab file
    vocab_dict = {str(k): list(v) for k, v in vocab.items()}
    with open('test_vocab.json', 'w') as f:
        json.dump(vocab_dict, f)
    
    # Create test merges file  
    merges_list = [[list(m[0]), list(m[1])] for m in merges]
    with open('test_merges.json', 'w') as f:
        json.dump(merges_list, f)
    
    # Test from_files
    tokenizer_from_files = Tokenizer.from_files('test_vocab.json', 'test_merges.json')
    
    result = tokenizer_from_files.encode('hello')
    decoded = tokenizer_from_files.decode(result)
    assert decoded == 'hello', f"from_files test failed: got '{decoded}'"
    
    print("from_files test passed!")
    
    # Test encode_iterable
    test_lines = ['hello\n', 'world\n', 'test']
    all_ids = list(tokenizer.encode_iterable(test_lines))
    full_text = 'hello\nworld\ntest'
    expected_ids = tokenizer.encode(full_text)
    
    decoded_from_iterable = tokenizer.decode(all_ids)
    assert decoded_from_iterable == full_text, f"encode_iterable test failed: got '{decoded_from_iterable}'"
    
    print("encode_iterable test passed!")
    
    # Cleanup
    os.remove('test_vocab.json')
    os.remove('test_merges.json')
    
    print("All tests passed successfully!")

if __name__ == '__main__':
    test_simple_tokenizer()