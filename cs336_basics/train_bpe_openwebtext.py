"""Train a byte-level BPE tokenizer on OpenWebText with resumable checkpoints."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterator
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import threading
import time

from huggingface_hub import HfApi, hf_hub_download
import psutil
import pyarrow.parquet as pq

from train_bpe import count_pretokens_in_texts, train_bpe_from_pretoken_counts

REPO_ID = "Skylion007/openwebtext"
REPO_TYPE = "dataset"
DATASET_SUBDIR = "plain_text"
TEXT_COLUMN = "text"
SPECIAL_TOKENS = ["<|endoftext|>"]

DEFAULT_DATA_DIR = Path("cs336_basics/openwebtext_data")
DEFAULT_WORK_DIR = Path("cs336_basics/openwebtext_work")
DEFAULT_OUTPUT_PATH = Path("cs336_basics/openwebtext_bpe/openwebtext_32000_res.json")
DEFAULT_VOCAB_SIZE = 32_000
DEFAULT_BATCH_SIZE = 2_048
DEFAULT_NUM_PROCESSES = min(os.cpu_count() or 4, 16)
DEFAULT_HF_TOKEN_ENV = "HF_TOKEN"


def _total_rss_bytes(process: psutil.Process) -> int:
    total = 0
    for proc in [process, *process.children(recursive=True)]:
        try:
            total += proc.memory_info().rss
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
    return total


def _sample_peak_memory(stop_event: threading.Event, peak_memory: list[int]) -> None:
    process = psutil.Process()
    while not stop_event.is_set():
        peak_memory[0] = max(peak_memory[0], _total_rss_bytes(process))
        time.sleep(0.05)
    peak_memory[0] = max(peak_memory[0], _total_rss_bytes(process))


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def _atomic_write_pickle(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)


def _read_pickle(path: Path) -> object:
    with open(path, "rb") as f:
        return pickle.load(f)


def get_hf_token(token_env: str | None) -> str | None:
    if token_env is None:
        return None

    env_names = [token_env]
    for fallback_env in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        if fallback_env not in env_names:
            env_names.append(fallback_env)

    for env_name in env_names:
        token = os.environ.get(env_name)
        if token:
            return token
    return None


def list_openwebtext_shards(
    max_shards: int | None = None,
    hf_token: str | None = None,
) -> list[str]:
    api = HfApi()
    entries = api.list_repo_tree(
        repo_id=REPO_ID,
        path_in_repo=DATASET_SUBDIR,
        repo_type=REPO_TYPE,
        token=hf_token,
    )
    shards = sorted(
        entry.path
        for entry in entries
        if entry.path.startswith(f"{DATASET_SUBDIR}/") and entry.path.endswith(".parquet")
    )
    if max_shards is not None:
        shards = shards[:max_shards]
    return shards


def download_shard(
    shard_path: str,
    data_dir: Path,
    local_files_only: bool = False,
    hf_token: str | None = None,
) -> Path:
    downloaded_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=shard_path,
        repo_type=REPO_TYPE,
        local_dir=data_dir,
        local_files_only=local_files_only,
        token=hf_token,
    )
    return Path(downloaded_path)


def shard_counter_path(work_dir: Path, shard_path: str) -> Path:
    return work_dir / "counts" / f"{Path(shard_path).stem}.pkl"


def shard_metadata_path(work_dir: Path, shard_path: str) -> Path:
    return work_dir / "counts" / f"{Path(shard_path).stem}.json"


def _iter_text_batches(
    parquet_path: Path,
    batch_size: int,
) -> Iterator[tuple[list[str], list[str]]]:
    parquet_file = pq.ParquetFile(parquet_path)
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=[TEXT_COLUMN]):
        texts = [text for text in batch.column(0).to_pylist() if text]
        if texts:
            yield texts, SPECIAL_TOKENS


def _count_text_batch(args: tuple[list[str], list[str]]) -> tuple[Counter[str], int]:
    texts, special_tokens = args
    return count_pretokens_in_texts(texts, special_tokens), len(texts)


def count_pretokens_in_parquet_shard(
    parquet_path: Path,
    batch_size: int,
    num_processes: int,
) -> tuple[Counter[str], int]:
    total_counts: Counter[str] = Counter()
    total_documents = 0
    jobs = _iter_text_batches(parquet_path=parquet_path, batch_size=batch_size)

    if num_processes == 1:
        for job in jobs:
            batch_counts, batch_documents = _count_text_batch(job)
            total_counts.update(batch_counts)
            total_documents += batch_documents
    else:
        with mp.Pool(processes=num_processes) as pool:
            for batch_counts, batch_documents in pool.imap_unordered(_count_text_batch, jobs):
                total_counts.update(batch_counts)
                total_documents += batch_documents

    return total_counts, total_documents


def load_completed_counts(
    work_dir: Path,
    shards: list[str],
) -> Counter[str]:
    total_counts: Counter[str] = Counter()
    for shard_path in shards:
        counter_path = shard_counter_path(work_dir, shard_path)
        if not counter_path.exists():
            raise FileNotFoundError(f"Missing checkpoint for {shard_path}: {counter_path}")
        total_counts.update(_read_pickle(counter_path))
    return total_counts


def write_state(
    work_dir: Path,
    shards: list[str],
    completed_shards: list[str],
    documents_seen: int,
    status: str,
) -> None:
    state = {
        "repo_id": REPO_ID,
        "dataset_subdir": DATASET_SUBDIR,
        "num_shards": len(shards),
        "completed_shards": completed_shards,
        "num_completed_shards": len(completed_shards),
        "documents_seen": documents_seen,
        "status": status,
        "updated_at_unix": time.time(),
    }
    _atomic_write_json(work_dir / "state.json", state)


def build_pretoken_checkpoints(
    shards: list[str],
    data_dir: Path,
    work_dir: Path,
    batch_size: int,
    num_processes: int,
    force_recount: bool,
    local_files_only: bool,
    delete_shards_after_counting: bool,
    hf_token: str | None,
) -> tuple[list[str], int]:
    completed_shards: list[str] = []
    documents_seen = 0

    for shard_index, shard_path in enumerate(shards, start=1):
        counter_path = shard_counter_path(work_dir, shard_path)
        if counter_path.exists() and not force_recount:
            metadata_path = shard_metadata_path(work_dir, shard_path)
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                documents_seen += int(metadata.get("num_documents", 0))
            completed_shards.append(shard_path)
            write_state(work_dir, shards, completed_shards, documents_seen, "counting")
            print(f"[{shard_index}/{len(shards)}] Reusing checkpoint {counter_path}")
            continue

        print(f"[{shard_index}/{len(shards)}] Downloading {shard_path}")
        parquet_path = download_shard(
            shard_path=shard_path,
            data_dir=data_dir,
            local_files_only=local_files_only,
            hf_token=hf_token,
        )

        print(f"[{shard_index}/{len(shards)}] Counting pre-tokens in {parquet_path}")
        start_time = time.perf_counter()
        counts, num_documents = count_pretokens_in_parquet_shard(
            parquet_path=parquet_path,
            batch_size=batch_size,
            num_processes=num_processes,
        )
        elapsed_seconds = time.perf_counter() - start_time

        _atomic_write_pickle(counter_path, counts)
        _atomic_write_json(
            shard_metadata_path(work_dir, shard_path),
            {
                "shard": shard_path,
                "num_documents": num_documents,
                "num_unique_pretokens": len(counts),
                "counting_time_seconds": elapsed_seconds,
            },
        )
        completed_shards.append(shard_path)
        documents_seen += num_documents
        write_state(work_dir, shards, completed_shards, documents_seen, "counting")
        print(
            f"[{shard_index}/{len(shards)}] Saved {counter_path} "
            f"({num_documents} documents, {len(counts)} unique pre-tokens, {elapsed_seconds:.2f}s)"
        )

        if delete_shards_after_counting:
            parquet_path.unlink(missing_ok=True)

    write_state(work_dir, shards, completed_shards, documents_seen, "counting_complete")
    return completed_shards, documents_seen


def write_bpe_result(
    output_path: Path,
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    metrics: dict,
    shards: list[str],
) -> None:
    result = {
        "vocab": {str(token_id): token.hex() for token_id, token in vocab.items()},
        "merges": [[left.hex(), right.hex()] for left, right in merges],
        "metrics": metrics,
        "source": {
            "repo_id": REPO_ID,
            "repo_type": REPO_TYPE,
            "dataset_subdir": DATASET_SUBDIR,
            "num_shards": len(shards),
            "shards": shards,
            "special_tokens": SPECIAL_TOKENS,
        },
    }
    _atomic_write_json(output_path, result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-processes", type=int, default=DEFAULT_NUM_PROCESSES)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--force-recount", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--skip-final-train", action="store_true")
    parser.add_argument("--delete-shards-after-counting", action="store_true")
    parser.add_argument("--hf-token-env", default=DEFAULT_HF_TOKEN_ENV)
    args = parser.parse_args()

    hf_token = get_hf_token(args.hf_token_env)
    if hf_token is None and not args.local_files_only:
        print(
            "No Hugging Face token found. Set HF_TOKEN or pass --hf-token-env "
            "with the name of an environment variable containing your token."
        )

    shards = list_openwebtext_shards(max_shards=args.max_shards, hf_token=hf_token)
    if not shards:
        raise RuntimeError("No OpenWebText parquet shards found.")

    stop_event = threading.Event()
    peak_memory = [_total_rss_bytes(psutil.Process())]
    memory_sampler = threading.Thread(
        target=_sample_peak_memory,
        args=(stop_event, peak_memory),
    )
    memory_sampler.start()
    start_time = time.perf_counter()
    try:
        completed_shards, documents_seen = build_pretoken_checkpoints(
            shards=shards,
            data_dir=args.data_dir,
            work_dir=args.work_dir,
            batch_size=args.batch_size,
            num_processes=args.num_processes,
            force_recount=args.force_recount,
            local_files_only=args.local_files_only,
            delete_shards_after_counting=args.delete_shards_after_counting,
            hf_token=hf_token,
        )

        if args.skip_final_train:
            write_state(args.work_dir, shards, completed_shards, documents_seen, "skipped_final_train")
            return

        print("Loading pre-token checkpoints")
        pretoken_counts = load_completed_counts(args.work_dir, shards)

        print(f"Training BPE with vocab_size={args.vocab_size}")
        train_start = time.perf_counter()
        vocab, merges = train_bpe_from_pretoken_counts(
            pretoken_counts=pretoken_counts,
            vocab_size=args.vocab_size,
            special_tokens=SPECIAL_TOKENS,
            show_progress=True,
        )
        train_elapsed_seconds = time.perf_counter() - train_start
    finally:
        elapsed_seconds = time.perf_counter() - start_time
        stop_event.set()
        memory_sampler.join()

    metrics = {
        "total_time_seconds": elapsed_seconds,
        "bpe_training_time_seconds": train_elapsed_seconds,
        "peak_memory_mb": peak_memory[0] / (1024 * 1024),
        "num_unique_pretokens": len(pretoken_counts),
    }
    write_bpe_result(args.output_path, vocab, merges, metrics, shards)
    write_state(args.work_dir, shards, shards, documents_seen, "complete")
    print(f"Output path: {args.output_path}")
    print(f"Total time: {elapsed_seconds:.2f} seconds")
    print(f"BPE training time: {train_elapsed_seconds:.2f} seconds")
    print(f"Peak memory: {peak_memory[0] / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    main()
