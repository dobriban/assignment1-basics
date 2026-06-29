#!/usr/bin/env python3
"""
Quick tokenizer test suite - core functionality tests only
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import tiktoken
from tests.common import FIXTURES_PATH, gpt2_bytes_to_unicode
from tests.adapters import get_tokenizer

VOCAB_PATH = FIXTURES_PATH / "gpt2_vocab.json"
MERGES_PATH = FIXTURES_PATH / "gpt2_merges.txt"

def get_tokenizer_from_vocab_merges_path(
    vocab_path, merges_path, special_tokens=None
):
    """Helper function equivalent to the one in test_tokenizer.py"""
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(vocab_path, encoding='utf-8') as vocab_f:
        gpt2_vocab = json.load(vocab_f)
    gpt2_bpe_merges = []
    with open(merges_path, encoding='utf-8') as f:
        for line in f:
            cleaned_line = line.rstrip()
            if cleaned_line and len(cleaned_line.split(" ")) == 2:
                gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
    
    vocab = {
        gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
        for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
    }
    
    if special_tokens:
        for special_token in special_tokens:
            byte_encoded_special_token = special_token.encode("utf-8")
            if byte_encoded_special_token not in set(vocab.values()):
                vocab[len(vocab)] = byte_encoded_special_token

    merges = [
        (
            bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
            bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
        )
        for merge_token_1, merge_token_2 in gpt2_bpe_merges
    ]
    return get_tokenizer(vocab, merges, special_tokens)

def test_roundtrip_empty():
    print("test_roundtrip_empty... ", end="")
    tokenizer = get_tokenizer_from_vocab_merges_path(VOCAB_PATH, MERGES_PATH)
    test_string = ""
    encoded_ids = tokenizer.encode(test_string)
    decoded_string = tokenizer.decode(encoded_ids)
    assert test_string == decoded_string
    print("PASS")

def test_empty_matches_tiktoken():
    print("test_empty_matches_tiktoken... ", end="")
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(VOCAB_PATH, MERGES_PATH)
    test_string = ""
    reference_ids = reference_tokenizer.encode(test_string)
    ids = tokenizer.encode(test_string)
    assert ids == reference_ids
    assert tokenizer.decode(ids) == test_string
    print("PASS")

def test_roundtrip_single_character():
    print("test_roundtrip_single_character... ", end="")
    tokenizer = get_tokenizer_from_vocab_merges_path(VOCAB_PATH, MERGES_PATH)
    test_string = "s"
    encoded_ids = tokenizer.encode(test_string)
    decoded_string = tokenizer.decode(encoded_ids)
    assert test_string == decoded_string
    print("PASS")

def test_single_character_matches_tiktoken():
    print("test_single_character_matches_tiktoken... ", end="")
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(VOCAB_PATH, MERGES_PATH)
    test_string = "s"
    reference_ids = reference_tokenizer.encode(test_string)
    ids = tokenizer.encode(test_string)
    assert ids == reference_ids
    assert tokenizer.decode(ids) == test_string
    print("PASS")

def test_roundtrip_ascii_string():
    print("test_roundtrip_ascii_string... ", end="")
    tokenizer = get_tokenizer_from_vocab_merges_path(VOCAB_PATH, MERGES_PATH)
    test_string = "Hello, how are you?"
    encoded_ids = tokenizer.encode(test_string)
    decoded_string = tokenizer.decode(encoded_ids)
    assert test_string == decoded_string
    print("PASS")

def test_ascii_string_matches_tiktoken():
    print("test_ascii_string_matches_tiktoken... ", end="")
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        VOCAB_PATH, MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    test_string = "Hello, how are you?"
    reference_ids = reference_tokenizer.encode(test_string)
    ids = tokenizer.encode(test_string)
    tokenized_string = [tokenizer.decode([x]) for x in ids]
    assert tokenized_string == ["Hello", ",", " how", " are", " you", "?"]
    assert tokenizer.decode(ids) == test_string
    print("PASS")

def test_roundtrip_unicode_string():
    print("test_roundtrip_unicode_string... ", end="")
    tokenizer = get_tokenizer_from_vocab_merges_path(VOCAB_PATH, MERGES_PATH)
    test_string = "Héllò hôw are ü? 🙃"
    encoded_ids = tokenizer.encode(test_string)
    decoded_string = tokenizer.decode(encoded_ids)
    assert test_string == decoded_string
    print("PASS")

def test_unicode_string_matches_tiktoken():
    print("test_unicode_string_matches_tiktoken... ", end="")
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        VOCAB_PATH, MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    test_string = "Héllò hôw are ü? 🙃"
    reference_ids = reference_tokenizer.encode(test_string)
    ids = tokenizer.encode(test_string)
    assert ids == reference_ids
    assert tokenizer.decode(ids) == test_string
    print("PASS")

def test_special_tokens():
    print("test_special_tokens... ", end="")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        VOCAB_PATH, MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    test_string = "Hello <|endoftext|> world"
    encoded_ids = tokenizer.encode(test_string)
    decoded_string = tokenizer.decode(encoded_ids)
    assert test_string == decoded_string
    
    # Check that special token is preserved as single token
    tokenized_string = [tokenizer.decode([x]) for x in encoded_ids]
    assert "<|endoftext|>" in tokenized_string
    print("PASS")

def test_encode_iterable_basic():
    print("test_encode_iterable_basic... ", end="")
    tokenizer = get_tokenizer_from_vocab_merges_path(VOCAB_PATH, MERGES_PATH)
    
    # Test with simple iterable
    test_lines = ["Hello\n", "world\n", "test"]
    all_ids = list(tokenizer.encode_iterable(test_lines))
    full_text = "Hello\nworld\ntest"
    expected_ids = tokenizer.encode(full_text)
    
    decoded_from_iterable = tokenizer.decode(all_ids)
    assert decoded_from_iterable == full_text
    print("PASS")

def run_quick_tests():
    """Run core tokenizer tests"""
    tests = [
        test_roundtrip_empty,
        test_empty_matches_tiktoken,
        test_roundtrip_single_character,
        test_single_character_matches_tiktoken,
        test_roundtrip_ascii_string,
        test_ascii_string_matches_tiktoken,
        test_roundtrip_unicode_string,
        test_unicode_string_matches_tiktoken,
        test_special_tokens,
        test_encode_iterable_basic,
    ]
    
    print(f"Running {len(tests)} core tokenizer tests...")
    print("=" * 50)
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("=" * 50)
    print(f"Tests passed: {passed}")
    print(f"Tests failed: {failed}")
    print(f"Total tests:  {len(tests)}")
    
    if failed == 0:
        print("\nAll core tests PASSED! Tokenizer implementation is working correctly.")
        return True
    else:
        print(f"\n{failed} tests FAILED!")
        return False

if __name__ == "__main__":
    run_quick_tests()