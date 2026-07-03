# Train a byte-level BPE tokenizer on the TinyStories dataset, 
# using a maximum vocabulary size of 10,000. 

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
from pathlib import Path
import threading
import time

import psutil
from train_bpe import train_bpe

# path  = "cs336_basics/TinyStories-valid.txt"

#DEFAULT_INPUT_PATH = Path("cs336_basics/test.txt")
DEFAULT_INPUT_PATH = Path("cs336_basics/TinyStories-valid.txt")

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


def result_path_for_input(input_path: str | Path) -> Path:
    input_path = Path(input_path)
    return input_path.with_name(f"{input_path.stem}_res.json")


def main(input_path: str | Path = DEFAULT_INPUT_PATH) -> None:
    stop_event = threading.Event()
    peak_memory = [_total_rss_bytes(psutil.Process())]
    memory_sampler = threading.Thread(
        target=_sample_peak_memory,
        args=(stop_event, peak_memory),
    )
    memory_sampler.start()
    start_time = time.perf_counter()
    try:
        vocab, merges = train_bpe(
            input_path=str(input_path),
            vocab_size=10000,
            special_tokens=["<|endoftext|>"],
            show_progress=True,
            num_processes=1,
        )
    finally:
        elapsed_seconds = time.perf_counter() - start_time
        stop_event.set()
        memory_sampler.join()

    output_path = result_path_for_input(input_path)
    result = {
        "vocab": {str(token_id): token.hex() for token_id, token in vocab.items()},
        "merges": [[left.hex(), right.hex()] for left, right in merges],
        "metrics": {
            "training_time_seconds": elapsed_seconds,
            "peak_memory_mb": peak_memory[0] / (1024 * 1024),
        },
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Output path: {output_path}")
    print(f"Training time: {elapsed_seconds:.2f} seconds")
    print(f"Peak memory: {peak_memory[0] / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    main()
