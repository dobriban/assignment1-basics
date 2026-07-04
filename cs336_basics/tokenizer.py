from __future__ import annotations

from collections.abc import Iterable, Iterator
import json
import os

import regex as re

from cs336_basics.train_bpe import PAT


def _gpt2_bytes_to_unicode() -> dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(161, 173))
        + list(range(174, 256))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(n) for n in cs]))


def _is_hex_string(value: str) -> bool:
    if len(value) % 2 != 0:
        return False
    try:
        bytes.fromhex(value)
    except ValueError:
        return False
    return True


def _load_vocab(vocab_filepath: str | os.PathLike) -> dict[int, bytes]:
    with open(vocab_filepath, encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and "vocab" in payload:
        payload = payload["vocab"]

    if not isinstance(payload, dict):
        raise ValueError("vocab file must contain a JSON object")

    # Native assignment serialization used by the local training scripts:
    # {"0": "00", "1": "01", ...}
    if all(str(key).lstrip("-").isdigit() for key in payload):
        vocab: dict[int, bytes] = {}
        for key, value in payload.items():
            token_id = int(key)
            if isinstance(value, str):
                vocab[token_id] = bytes.fromhex(value)
            elif isinstance(value, list):
                vocab[token_id] = bytes(value)
            else:
                raise ValueError(f"unsupported vocab value for token {token_id}: {value!r}")
        return vocab

    # GPT-2-style JSON maps printable byte strings to token IDs.
    byte_decoder = {v: k for k, v in _gpt2_bytes_to_unicode().items()}
    return {
        int(token_id): bytes(byte_decoder[char] for char in token)
        for token, token_id in payload.items()
    }


def _decode_merge_piece(piece: str, byte_decoder: dict[str, int], allow_hex: bool) -> bytes:
    if allow_hex and _is_hex_string(piece):
        return bytes.fromhex(piece)
    return bytes(byte_decoder[char] for char in piece)


def _load_merges(merges_filepath: str | os.PathLike) -> list[tuple[bytes, bytes]]:
    with open(merges_filepath, encoding="utf-8") as f:
        raw = f.read()

    byte_decoder = {v: k for k, v in _gpt2_bytes_to_unicode().items()}

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        merges: list[tuple[bytes, bytes]] = []
        for line in raw.splitlines():
            parts = line.rstrip().split(" ")
            if len(parts) != 2:
                continue
            left, right = parts
            merges.append(
                (
                    _decode_merge_piece(left, byte_decoder, allow_hex=False),
                    _decode_merge_piece(right, byte_decoder, allow_hex=False),
                )
            )
        return merges

    if isinstance(payload, dict) and "merges" in payload:
        payload = payload["merges"]

    if not isinstance(payload, list):
        raise ValueError("merges file must contain a JSON list or a two-column text file")

    allow_hex = all(
        isinstance(item, (list, tuple))
        and len(item) == 2
        and all(isinstance(piece, str) and _is_hex_string(piece) for piece in item)
        for item in payload
    )

    merges = []
    for item in payload:
        if isinstance(item, str):
            parts = item.split(" ")
        else:
            parts = item
        if len(parts) != 2:
            raise ValueError(f"merge entries must have length 2: {item!r}")
        left, right = parts
        merges.append(
            (
                _decode_merge_piece(left, byte_decoder, allow_hex=allow_hex),
                _decode_merge_piece(right, byte_decoder, allow_hex=allow_hex),
            )
        )
    return merges


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(special_tokens or [])

        self.token_to_id: dict[bytes, int] = {}
        for token_id, token in self.vocab.items():
            self.token_to_id.setdefault(token, token_id)

        next_token_id = max(self.vocab.keys(), default=-1) + 1
        for special_token in self.special_tokens:
            token_bytes = special_token.encode("utf-8")
            if token_bytes not in self.token_to_id:
                self.vocab[next_token_id] = token_bytes
                self.token_to_id[token_bytes] = next_token_id
                next_token_id += 1

        self.special_token_to_id = {
            special_token: self.token_to_id[special_token.encode("utf-8")]
            for special_token in self.special_tokens
        }

        self.merge_ranks: dict[tuple[int, int], tuple[int, int]] = {}
        for rank, (left, right) in enumerate(self.merges):
            left_id = self.token_to_id.get(left)
            right_id = self.token_to_id.get(right)
            merged_id = self.token_to_id.get(left + right)
            if left_id is None or right_id is None or merged_id is None:
                continue
            self.merge_ranks[(left_id, right_id)] = (rank, merged_id)

        self.max_special_token_length = max((len(token) for token in self.special_tokens), default=0)
        if self.special_tokens:
            pattern = "|".join(re.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True))
            self.special_token_pattern = re.compile(pattern)
        else:
            self.special_token_pattern = None

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        return cls(
            vocab=_load_vocab(vocab_filepath),
            merges=_load_merges(merges_filepath),
            special_tokens=special_tokens,
        )

    def _iter_text_and_special_spans(self, text: str) -> Iterator[tuple[bool, int, int, str]]:
        if self.special_token_pattern is None:
            if text:
                yield False, 0, len(text), text
            return

        cursor = 0
        for match in self.special_token_pattern.finditer(text):
            if match.start() > cursor:
                yield False, cursor, match.start(), text[cursor : match.start()]
            yield True, match.start(), match.end(), match.group(0)
            cursor = match.end()

        if cursor < len(text):
            yield False, cursor, len(text), text[cursor:]

    def _iter_text_and_special_tokens(self, text: str) -> Iterator[tuple[bool, str]]:
        for is_special, _start, _end, piece in self._iter_text_and_special_spans(text):
            yield is_special, piece

    def _encode_pretoken(self, pretoken: str) -> Iterator[int]:
        token_ids = [self.token_to_id[bytes([byte])] for byte in pretoken.encode("utf-8")]

        while len(token_ids) > 1:
            best_rank: int | None = None
            best_pair: tuple[int, int] | None = None
            best_merged_id: int | None = None

            for left, right in zip(token_ids, token_ids[1:]):
                merge = self.merge_ranks.get((left, right))
                if merge is None:
                    continue
                rank, merged_id = merge
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_pair = (left, right)
                    best_merged_id = merged_id

            if best_pair is None or best_merged_id is None:
                break

            merged_token_ids: list[int] = []
            i = 0
            while i < len(token_ids):
                if i + 1 < len(token_ids) and (token_ids[i], token_ids[i + 1]) == best_pair:
                    merged_token_ids.append(best_merged_id)
                    i += 2
                else:
                    merged_token_ids.append(token_ids[i])
                    i += 1
            token_ids = merged_token_ids

        yield from token_ids

    def _encode_text_piece(self, text: str) -> Iterator[int]:
        for match in re.finditer(PAT, text):
            yield from self._encode_pretoken(match.group(0))

    def _iter_encoded_ids(self, text: str) -> Iterator[int]:
        for is_special, piece in self._iter_text_and_special_tokens(text):
            if is_special:
                yield self.special_token_to_id[piece]
            else:
                yield from self._encode_text_piece(piece)

    def encode(self, text: str) -> list[int]:
        return list(self._iter_encoded_ids(text))

    def _stable_prefix_length(self, text: str) -> int:
        if not text:
            return 0

        special_stable_limit = len(text)
        if self.max_special_token_length > 1:
            special_stable_limit = max(0, len(text) - self.max_special_token_length + 1)

        stable_end = 0
        for is_special, piece_start, piece_end, piece in self._iter_text_and_special_spans(text):
            if is_special:
                if piece_end <= special_stable_limit:
                    stable_end = piece_end
                continue

            for match in re.finditer(PAT, piece):
                token_end = piece_start + match.end()
                if token_end < len(text) and token_end <= special_stable_limit:
                    stable_end = token_end

        return stable_end

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        buffer = ""
        for chunk in iterable:
            buffer += chunk
            prefix_length = self._stable_prefix_length(buffer)
            if prefix_length == 0:
                continue

            yield from self._iter_encoded_ids(buffer[:prefix_length])
            buffer = buffer[prefix_length:]

        if buffer:
            yield from self._iter_encoded_ids(buffer)

    def decode(self, ids: list[int]) -> str:
        token_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        return token_bytes.decode("utf-8", errors="replace")
