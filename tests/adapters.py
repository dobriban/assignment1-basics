from __future__ import annotations

import os

from cs336_basics.train_bpe import train_bpe


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Run the student's BPE training implementation for the test suite."""
    return train_bpe(
        input_path=os.fspath(input_path),
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )
