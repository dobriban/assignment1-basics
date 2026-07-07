from __future__ import annotations

import argparse
import random
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cs336_basics.tokenizer import Tokenizer


DOC_DELIMITER = "<|endoftext|>"
SPECIAL_TOKENS = [DOC_DELIMITER]
DEFAULT_NUM_DOCS = 10
DEFAULT_SEED = 0

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TINYSTORIES_DATA_CANDIDATES = (
    PROJECT_ROOT / "data" / "TinyStoriesV2-GPT4-valid.txt",
    PROJECT_ROOT / "cs336_basics" / "TinyStories-valid.txt",
    PROJECT_ROOT / "data" / "TinyStoriesV2-GPT4-train.txt",
)
OPENWEBTEXT_DATA_CANDIDATES = (
    PROJECT_ROOT / "data" / "owt_valid.txt",
    PROJECT_ROOT / "data" / "owt_train.txt",
)
TINYSTORIES_TOKENIZER_CANDIDATES = (
    PROJECT_ROOT / "cs336_basics" / "TinyStories-train_res.json",
    PROJECT_ROOT / "cs336_basics" / "TinyStories-valid_res.json",
)
OPENWEBTEXT_TOKENIZER_CANDIDATES = (
    PROJECT_ROOT / "cs336_basics" / "openwebtext_bpe" / "openwebtext_32000_res.json",
)


@dataclass(frozen=True)
class CompressionResult:
    dataset_name: str
    tokenizer_name: str
    num_documents_seen: int
    num_documents_sampled: int
    total_bytes: int
    total_tokens: int

    @property
    def bytes_per_token(self) -> float:
        return self.total_bytes / self.total_tokens


def first_existing_path(candidates: tuple[Path, ...], label: str) -> Path:
    for path in candidates:
        if path.exists():
            return path

    candidate_list = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(f"Could not find {label}. Tried:\n{candidate_list}")


def iter_documents(
    path: Path,
    delimiter: str = DOC_DELIMITER,
    chunk_size: int = 1024 * 1024,
) -> Iterator[str]:
    buffer = ""
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break

            buffer += chunk
            parts = buffer.split(delimiter)
            for raw_document in parts[:-1]:
                document = raw_document.strip()
                if document:
                    yield document
            buffer = parts[-1]

    final_document = buffer.strip()
    if final_document:
        yield final_document


def sample_documents(path: Path, num_documents: int, seed: int) -> tuple[list[str], int]:
    rng = random.Random(seed)
    sample: list[str] = []
    documents_seen = 0

    for document in iter_documents(path):
        documents_seen += 1
        if len(sample) < num_documents:
            sample.append(document)
            continue

        replacement_index = rng.randrange(documents_seen)
        if replacement_index < num_documents:
            sample[replacement_index] = document

    if len(sample) < num_documents:
        raise ValueError(f"Only found {len(sample)} documents in {path}, expected at least {num_documents}.")

    return sample, documents_seen


def load_tokenizer(result_path: Path) -> Tokenizer:
    return Tokenizer.from_files(
        vocab_filepath=result_path,
        merges_filepath=result_path,
        special_tokens=SPECIAL_TOKENS,
    )


def compute_compression_ratio(
    dataset_name: str,
    tokenizer_name: str,
    documents: list[str],
    documents_seen: int,
    tokenizer: Tokenizer,
) -> CompressionResult:
    total_bytes = 0
    total_tokens = 0

    for document in documents:
        total_bytes += len(document.encode("utf-8"))
        total_tokens += len(tokenizer.encode(document))

    if total_tokens == 0:
        raise ValueError(f"{dataset_name} sample produced zero tokens.")

    return CompressionResult(
        dataset_name=dataset_name,
        tokenizer_name=tokenizer_name,
        num_documents_seen=documents_seen,
        num_documents_sampled=len(documents),
        total_bytes=total_bytes,
        total_tokens=total_tokens,
    )


def print_result_table(results: list[CompressionResult]) -> None:
    print("dataset      tokenizer            docs seen  sampled     bytes    tokens  bytes/token")
    print("-----------  -------------------  ---------  -------  --------  --------  -----------")
    for result in results:
        print(
            f"{result.dataset_name:<11}  "
            f"{result.tokenizer_name:<19}  "
            f"{result.num_documents_seen:>9}  "
            f"{result.num_documents_sampled:>7}  "
            f"{result.total_bytes:>8}  "
            f"{result.total_tokens:>8}  "
            f"{result.bytes_per_token:>11.3f}"
        )


def print_deliverable(results: list[CompressionResult], seed: int) -> None:
    by_dataset = {result.dataset_name: result for result in results}
    tinystories = by_dataset["TinyStories"]
    openwebtext = by_dataset["OpenWebText"]
    print()
    print("Deliverable:")
    print(
        f"Using a fixed random seed of {seed} and {tinystories.num_documents_sampled} sampled documents "
        f"from each validation set, "
        f"the TinyStories 10K tokenizer achieves {tinystories.bytes_per_token:.3f} bytes/token on "
        f"TinyStories, while the OpenWebText 32K tokenizer achieves "
        f"{openwebtext.bytes_per_token:.3f} bytes/token on OpenWebText."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure bytes/token compression for trained BPE tokenizers.")
    parser.add_argument("--num-docs", type=int, default=DEFAULT_NUM_DOCS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--tinystories-data", type=Path, default=None)
    parser.add_argument("--openwebtext-data", type=Path, default=None)
    parser.add_argument("--tinystories-tokenizer", type=Path, default=None)
    parser.add_argument("--openwebtext-tokenizer", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tinystories_data = args.tinystories_data or first_existing_path(
        TINYSTORIES_DATA_CANDIDATES,
        "TinyStories data",
    )
    openwebtext_data = args.openwebtext_data or first_existing_path(
        OPENWEBTEXT_DATA_CANDIDATES,
        "OpenWebText data",
    )
    tinystories_tokenizer_path = args.tinystories_tokenizer or first_existing_path(
        TINYSTORIES_TOKENIZER_CANDIDATES,
        "TinyStories tokenizer result JSON",
    )
    openwebtext_tokenizer_path = args.openwebtext_tokenizer or first_existing_path(
        OPENWEBTEXT_TOKENIZER_CANDIDATES,
        "OpenWebText tokenizer result JSON",
    )

    tinystories_sample, tinystories_seen = sample_documents(
        path=tinystories_data,
        num_documents=args.num_docs,
        seed=args.seed,
    )
    openwebtext_sample, openwebtext_seen = sample_documents(
        path=openwebtext_data,
        num_documents=args.num_docs,
        seed=args.seed,
    )

    tinystories_tokenizer = load_tokenizer(tinystories_tokenizer_path)
    openwebtext_tokenizer = load_tokenizer(openwebtext_tokenizer_path)

    results = [
        compute_compression_ratio(
            dataset_name="TinyStories",
            tokenizer_name="TinyStories 10K",
            documents=tinystories_sample,
            documents_seen=tinystories_seen,
            tokenizer=tinystories_tokenizer,
        ),
        compute_compression_ratio(
            dataset_name="OpenWebText",
            tokenizer_name="OpenWebText 32K",
            documents=openwebtext_sample,
            documents_seen=openwebtext_seen,
            tokenizer=openwebtext_tokenizer,
        ),
    ]

    print(f"TinyStories data: {tinystories_data}")
    print(f"OpenWebText data: {openwebtext_data}")
    print(f"TinyStories tokenizer: {tinystories_tokenizer_path}")
    print(f"OpenWebText tokenizer: {openwebtext_tokenizer_path}")
    print()
    print_result_table(results)
    print_deliverable(results, args.seed)


if __name__ == "__main__":
    main()
