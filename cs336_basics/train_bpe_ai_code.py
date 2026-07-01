
def _merge_word(
    word: tuple[bytes, ...],
    pair_to_merge: tuple[bytes, bytes],
    merged_token: bytes,
) -> tuple[bytes, ...]:
    merged_word: list[bytes] = []
    i = 0

    while i < len(word):
        if i < len(word) - 1 and (word[i], word[i + 1]) == pair_to_merge:
            merged_word.append(merged_token)
            i += 2
        else:
            merged_word.append(word[i])
            i += 1

    return tuple(merged_word)


def train_bpe(input_path: str, vocab_size: int, 
              special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    pretoken_counts = count_pretokens_parallel(
        input_path=input_path,
        special_tokens=special_tokens,
        num_processes=4,
    )

    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_token_id = 256

    for special_token in special_tokens:
        if next_token_id >= vocab_size:
            break
        vocab[next_token_id] = special_token.encode("utf-8")
        next_token_id += 1

    merges: list[tuple[bytes, bytes]] = []

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
