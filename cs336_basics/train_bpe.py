import regex as re
from collections import Counter
import multiprocessing as mp

def process_text_chunk(chunk_and_pat):
    """Process a text chunk (not file chunk) and return token counts"""
    chunk, pat = chunk_and_pat
    token_counter = Counter()
    for match in re.finditer(pat, chunk):
        b = match.group(0).encode("utf-8")
        token_counter[tuple(b)] += 1
    return token_counter

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    # 1. Read input text and split by special tokens (same as original)
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 2. Strip special tokens and split (same as original)
    special_pattern = "|".join(re.escape(tok) for tok in special_tokens)
    chunks = re.split(special_pattern, text)

    # 3. Pre-tokenization with parallel processing of text chunks
    if len(chunks) > 1 and sum(len(chunk) for chunk in chunks) > 100000:  # Parallel for large text
        # Use multiprocessing to process text chunks in parallel
        chunk_args = [(chunk, PAT) for chunk in chunks if chunk.strip()]  # Skip empty chunks
        
        with mp.Pool(processes=min(mp.cpu_count(), len(chunk_args))) as pool:
            chunk_counters = pool.map(process_text_chunk, chunk_args)
        
        # Merge all counters
        token_counter = Counter()
        for counter in chunk_counters:
            token_counter.update(counter)
    else:
        # Serial processing for small files
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
                pair = (token[i], token[i+1])
                pair_counts[pair] += freq

        if not pair_counts:
            break

        # Pick most frequent pair (lexicographic tie-break)
        best_pair = max(pair_counts.items(), key=lambda x: (x[1], (vocab[x[0][0]], vocab[x[0][1]])))[0]
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))

        # Build new tokens
        new_token = vocab[best_pair[0]] + vocab[best_pair[1]]
        new_token_id = next_token_id
        vocab[new_token_id] = new_token
        next_token_id += 1

        # Replace pairs in token_counter
        new_counter = Counter()
        for token, freq in token_counter.items():
            new_token_tuple = []
            i = 0
            while i < len(token):
                if i < len(token) - 1 and (token[i], token[i+1]) == best_pair:
                    new_token_tuple.append(new_token_id)
                    i += 2
                else:
                    new_token_tuple.append(token[i])
                    i += 1
            new_counter[tuple(new_token_tuple)] += freq
        token_counter = new_counter

    return vocab, merges
