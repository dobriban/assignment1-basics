import regex as re
from collections import Counter
import multiprocessing as mp
import os
from typing import BinaryIO

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def _count_pretokens_in_chunk(args: tuple[str, int, int, list[str]]) -> Counter[str]:
    """Worker function: read one byte range and count regex pre-tokens in it."""
    input_path, start, end, special_tokens = args

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    counts: Counter[str] = Counter()

    if special_tokens:
        special_token_pattern = "|".join(
            re.escape(token) for token in sorted(special_tokens, key=len, reverse=True)
        )
        text_parts = re.split(special_token_pattern, chunk)
    else:
        text_parts = [chunk]

    for text_part in text_parts:
        for match in re.finditer(PAT, text_part):
            counts[match.group(0)] += 1

    return counts


def count_pretokens_parallel(
    input_path: str,
    special_tokens: list[str],
    num_processes: int = 4,
) -> Counter[str]:
    """Split the file into chunks and count pre-tokens in parallel."""
    if num_processes < 1:
        raise ValueError("num_processes must be at least 1")

    split_token = special_tokens[0].encode("utf-8") if special_tokens else b"<|endoftext|>"

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(file=f, desired_num_chunks = num_processes, 
                                           split_special_token=split_token)
    
    chunk_args = [
        (input_path, start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    
    total_counts: Counter[str] = Counter()
    if num_processes == 1 or len(chunk_args) <= 1:
        for args in chunk_args:
            total_counts.update(_count_pretokens_in_chunk(args))
    else:
        with mp.Pool(processes=num_processes) as pool:
            chunk_counts = pool.map(_count_pretokens_in_chunk, chunk_args)

        for counts in chunk_counts:
            total_counts.update(counts)

    return total_counts


def _merge_word(
    word: tuple[bytes, ...],
    pair_to_merge: tuple[bytes, bytes],
    merged_token: bytes,
) -> tuple[bytes, ...]:
    merged_word: list[bytes] = []
    i = 0

    while i < len(word):
        if i + 1 < len(word) and (word[i], word[i + 1]) == pair_to_merge:
            merged_word.append(merged_token)
            i += 2
        else:
            merged_word.append(word[i])
            i += 1

    return tuple(merged_word)


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]
              ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    pretoken_counts = count_pretokens_parallel(
        input_path=input_path,
        special_tokens=special_tokens,
        num_processes=4,
    )

    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_token_id = 256

    #special tokens are added to the vocab, They have been separated out and never merged. 
    for special_token in special_tokens:
        if next_token_id >= vocab_size:
            break
        vocab[next_token_id] = special_token.encode("utf-8")
        next_token_id += 1

    merges: list[tuple[bytes, bytes]] = []

    # Represent each pre-token as a tuple of single-byte tokens.
    # Keep its frequency so pair counts are weighted by how often it appears.
    tokenized_pretokens: dict[tuple[bytes, ...], int] = {
        tuple(bytes([byte]) for byte in pretoken.encode("utf-8")): count
        for pretoken, count in pretoken_counts.items()
    }

    while next_token_id < vocab_size:
        pair_counts: Counter[tuple[bytes, bytes]] = Counter()

        for word, count in tokenized_pretokens.items():
            for pair in zip(word, word[1:]):
                pair_counts[pair] += count

        if not pair_counts:
            break

        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        merged_token = best_pair[0] + best_pair[1]

        merges.append(best_pair)
        vocab[next_token_id] = merged_token
        next_token_id += 1

        tokenized_pretokens = {
            _merge_word(word, best_pair, merged_token): count
            for word, count in tokenized_pretokens.items()
        }

    return vocab, merges
