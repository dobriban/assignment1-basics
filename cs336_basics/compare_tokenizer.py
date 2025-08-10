import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import tiktoken

# Compare with tiktoken reference
reference_tokenizer = tiktoken.get_encoding('gpt2')

from tests.common import FIXTURES_PATH, gpt2_bytes_to_unicode
from tests.adapters import get_tokenizer
import json

def get_tokenizer_from_vocab_merges_path(vocab_path, merges_path, special_tokens=None):
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(vocab_path, encoding='utf-8') as vocab_f:
        gpt2_vocab = json.load(vocab_f)
    gpt2_bpe_merges = []
    with open(merges_path, encoding='utf-8') as f:
        for line in f:
            cleaned_line = line.rstrip()
            if cleaned_line and len(cleaned_line.split(' ')) == 2:
                gpt2_bpe_merges.append(tuple(cleaned_line.split(' ')))
    
    vocab = {
        gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
        for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
    }
    
    if special_tokens:
        for special_token in special_tokens:
            byte_encoded_special_token = special_token.encode('utf-8')
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

VOCAB_PATH = FIXTURES_PATH / 'gpt2_vocab.json'
MERGES_PATH = FIXTURES_PATH / 'gpt2_merges.txt'

tokenizer = get_tokenizer_from_vocab_merges_path(
    vocab_path=VOCAB_PATH,
    merges_path=MERGES_PATH,
)

test_string = 'Hello, how are you?'
print(f'Test string: {test_string}')

reference_ids = reference_tokenizer.encode(test_string)
my_ids = tokenizer.encode(test_string)

print(f'Reference IDs: {reference_ids}')
print(f'My IDs:        {my_ids}')
print(f'Match: {reference_ids == my_ids}')

if reference_ids != my_ids:
    print('Tokens differ:')
    ref_tokens = [reference_tokenizer.decode([id]) for id in reference_ids]
    my_tokens = [tokenizer.decode([id]) for id in my_ids]
    print(f'Reference tokens: {ref_tokens}')
    print(f'My tokens:        {my_tokens}')