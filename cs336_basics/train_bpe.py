import regex as re
from collections import Counter
from collections.abc import Iterable
import heapq
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

    return count_pretokens_in_texts([chunk], special_tokens)


def count_pretokens_in_texts(
    texts: Iterable[str],
    special_tokens: list[str],
) -> Counter[str]:
    """Count regex pre-tokens in already-decoded text strings."""
    counts: Counter[str] = Counter()

    if special_tokens:
        special_token_pattern = "|".join(
            re.escape(token) for token in sorted(special_tokens, key=len, reverse=True)
        )
    else:
        special_token_pattern = ""

    for text in texts:
        if special_token_pattern:
            text_parts = re.split(special_token_pattern, text)
        else:
            text_parts = [text]

        for text_part in text_parts:
            for match in re.finditer(PAT, text_part):
                counts[match.group(0)] += 1

    return counts


def _desired_num_chunks(file_size: int, num_processes: int) -> int:
    if file_size >= 64 * 1024 * 1024:
        return num_processes * 16
    return num_processes


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
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        f.seek(0)
        boundaries = find_chunk_boundaries(
            file=f,
            desired_num_chunks=_desired_num_chunks(file_size, num_processes),
            split_special_token=split_token,
        )

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


class _ReversePairKey:
    __slots__ = ("key",)

    def __init__(self, pair_bytes: tuple[bytes, bytes]) -> None:
        self.key = pair_bytes

    def __lt__(self, other: "_ReversePairKey") -> bool:
        return self.key > other.key


def _pair_counts_for_word(word: tuple[int, ...]) -> dict[tuple[int, int], int]:
    pair_counts: dict[tuple[int, int], int] = {}
    for i in range(len(word) - 1):
        pair = (word[i], word[i + 1])
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
    return pair_counts


def _push_pair(
    heap: list[tuple[int, _ReversePairKey, tuple[int, int]]],
    pair: tuple[int, int],
    count: int,
    vocab: dict[int, bytes],
) -> None:
    heapq.heappush(heap, (-count, _ReversePairKey((vocab[pair[0]], vocab[pair[1]])), pair))


def _merge_word(
    word: tuple[int, ...],
    pair_to_merge: tuple[int, int],
    merged_token_id: int,
) -> tuple[int, ...]:
    first, second = pair_to_merge
    merged_word: list[int] | None = None
    i = 0
    n = len(word)

    while i < n:
        if i + 1 < n and word[i] == first and word[i + 1] == second:
            if merged_word is None:
                merged_word = list(word[:i])
            merged_word.append(merged_token_id)
            i += 2
        else:
            if merged_word is not None:
                merged_word.append(word[i])
            i += 1

    if merged_word is None:
        return word

    return tuple(merged_word)


def _print_progress(completed_merges: int, total_merges: int, next_percent: int) -> int:
    if total_merges <= 0:
        return next_percent

    percent_complete = min(100, completed_merges * 100 // total_merges)
    while next_percent <= percent_complete:
        print(f"BPE training progress: {next_percent}% ({completed_merges}/{total_merges} merges)")
        next_percent += 1

    return next_percent


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    show_progress: bool = False,
    num_processes: int = 4,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    pretoken_counts = count_pretokens_parallel(
        input_path=input_path,
        special_tokens=special_tokens,
        num_processes=num_processes,
    )

    return train_bpe_from_pretoken_counts(
        pretoken_counts=pretoken_counts,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=show_progress,
    )


def train_bpe_from_pretoken_counts(
    pretoken_counts: Counter[str],
    vocab_size: int,
    special_tokens: list[str],
    show_progress: bool = False,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train byte-level BPE from precomputed pre-token counts."""

    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_token_id = 256

    # Special tokens are added to the vocab but never merged with other tokens.
    for special_token in special_tokens:
        if next_token_id >= vocab_size:
            break
        vocab[next_token_id] = special_token.encode("utf-8")
        next_token_id += 1

    merges: list[tuple[bytes, bytes]] = []
    total_merges = max(vocab_size - next_token_id, 0)
    next_progress_percent = 1

    # Represent each pre-token as a tuple of token IDs, initially raw byte IDs.
    # Keep its frequency so pair counts are weighted by how often it appears.
    tokenized_pretokens = {
        tuple(pretoken.encode("utf-8")): count
        for pretoken, count in pretoken_counts.items()
    }

    words = list(tokenized_pretokens.keys())
    word_counts = list(tokenized_pretokens.values())
    pair_counts: dict[tuple[int, int], int] = {}
    pair_to_word_ids: dict[tuple[int, int], set[int]] = {}

    for word_id, word in enumerate(words):
        word_pair_counts = _pair_counts_for_word(word)
        for pair, occurrences in word_pair_counts.items():
            pair_counts[pair] = pair_counts.get(pair, 0) + occurrences * word_counts[word_id]
            pair_to_word_ids.setdefault(pair, set()).add(word_id)

    pair_heap: list[tuple[int, _ReversePairKey, tuple[int, int]]] = []
    for pair, count in pair_counts.items():
        _push_pair(pair_heap, pair, count, vocab)

    while next_token_id < vocab_size:
        best_pair: tuple[int, int] | None = None
        while pair_heap:
            neg_count, _, pair = heapq.heappop(pair_heap)
            if pair_counts.get(pair, 0) == -neg_count:
                best_pair = pair
                break

        if best_pair is None:
            break

        left_token, right_token = vocab[best_pair[0]], vocab[best_pair[1]]
        merged_token = left_token + right_token

        merges.append((left_token, right_token))
        vocab[next_token_id] = merged_token

        affected_word_ids = list(pair_to_word_ids.pop(best_pair, set()))
        pair_count_deltas: dict[tuple[int, int], int] = {}

        for word_id in affected_word_ids:
            old_word = words[word_id]
            new_word = _merge_word(old_word, best_pair, next_token_id)
            if new_word == old_word:
                continue

            word_count = word_counts[word_id]
            old_pair_counts = _pair_counts_for_word(old_word)
            new_pair_counts = _pair_counts_for_word(new_word)
            words[word_id] = new_word

            for pair, occurrences in old_pair_counts.items():
                word_ids = pair_to_word_ids.get(pair)
                if word_ids is not None:
                    word_ids.discard(word_id)
                    if not word_ids:
                        del pair_to_word_ids[pair]
                pair_count_deltas[pair] = pair_count_deltas.get(pair, 0) - occurrences * word_count

            for pair, occurrences in new_pair_counts.items():
                pair_to_word_ids.setdefault(pair, set()).add(word_id)
                pair_count_deltas[pair] = pair_count_deltas.get(pair, 0) + occurrences * word_count

        for pair, delta in pair_count_deltas.items():
            new_count = pair_counts.get(pair, 0) + delta
            if new_count <= 0:
                pair_counts.pop(pair, None)
            else:
                pair_counts[pair] = new_count
                _push_pair(pair_heap, pair, new_count, vocab)

        next_token_id += 1
        if show_progress:
            next_progress_percent = _print_progress(
                completed_merges=len(merges),
                total_merges=total_merges,
                next_percent=next_progress_percent,
            )

    return vocab, merges
