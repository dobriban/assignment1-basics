import regex as re
from collections import Counter, defaultdict

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    # 1. Read input text
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 2. Strip special tokens and split
    special_pattern = "|".join(re.escape(tok) for tok in special_tokens)
    chunks = re.split(special_pattern, text)

    # 3. Pre-tokenization and count pre-token frequencies
    token_counter = Counter()
    for chunk in chunks:
        for match in re.finditer(PAT, chunk):
            b = match.group(0).encode("utf-8")
            token_counter[tuple(b)] += 1

    # 4. Initialize vocab and merges
    vocab = {i: bytes([i]) for i in range(256)}
    next_token_id = 256
    for token in special_tokens:
        vocab[next_token_id] = token.encode("utf-8")
        next_token_id += 1

    merges = []

    # 5. Main BPE loop
    while len(vocab) < vocab_size:
        # Count pairs
        pair_counts = Counter()
        for token, freq in token_counter.items():
            for i in range(len(token) - 1):
                pair = (token[i:i+1], token[i+1:i+2])
                pair_counts[pair] += freq

        if not pair_counts:
            break

        # Pick most frequent pair (lexicographic tie-break)
        best_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))[0]
        merges.append(best_pair)

        # Build new tokens
        new_token = b''.join(best_pair)
        new_token_id = next_token_id
        vocab[new_token_id] = new_token
        next_token_id += 1

        # Replace pairs in token_counter
        new_counter = Counter()
        for token, freq in token_counter.items():
            new_token_tuple = []
            i = 0
            while i < len(token):
                if i < len(token) - 1 and (token[i:i+1], token[i+1:i+2]) == best_pair:
                    new_token_tuple.append(new_token)
                    i += 2
                else:
                    new_token_tuple.append(token[i:i+1])
                    i += 1
            new_counter[tuple(new_token_tuple)] += freq
        token_counter = new_counter

    return vocab, merges
