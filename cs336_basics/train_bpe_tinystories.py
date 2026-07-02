# Train a byte-level BPE tokenizer on the TinyStories dataset, 
# using a maximum vocabulary
# size of 10,000. 

# Make sure to add the TinyStories <|endoftext|> special token to the
# vocabulary. 

# Serialize the resulting vocabulary and merges to disk for further inspection. 
# How much time and memory did training take? 
# What is the longest token in the vocabulary? 
# Does it make sense?

# Hint You should be able to get under 2 minutes for 
# BPE training using multiprocessing
# during pre-tokenization and the following two facts:
# (a) The <|endoftext|> token delimits documents in the data files.
# (b) The <|endoftext|> token is handled as a special case before the BPE merges are applied.

#train_bpe_tinystories

# tasks:
# Download the data. 
# Figure out loading and Processing Pipeline 
# train tokenizer on the TinyStories validation set instead, which is 22K docs

import json

from train_bpe import train_bpe

# path  = "cs336_basics/TinyStories-valid.txt"

def main():
    input_path = "cs336_basics/test.txt"

    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )
    
    output_path = "cs336_basics/test_res.json"
    result = {
        "vocab": {str(token_id): token.hex() for token_id, token in vocab.items()},
        "merges": [[left.hex(), right.hex()] for left, right in merges],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
