#!/usr/bin/env python3
"""
Encode TinyStories and OpenWebText datasets into sequences of integer token IDs.
Saves as NumPy arrays with uint16 datatype for efficient storage.
"""

import sys
import os
import numpy as np
import gzip
from pathlib import Path
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tokenizer import Tokenizer

def check_vocab_size_compatibility(tokenizer, max_token_id=None):
    """
    Check if tokenizer vocabulary fits in uint16 (max value 65535).
    
    Args:
        tokenizer: Tokenizer instance
        max_token_id: Optional override for max token ID
    
    Returns:
        bool: True if compatible with uint16
    """
    if max_token_id is None:
        max_token_id = max(tokenizer.vocab.keys()) if tokenizer.vocab else 0
    
    max_uint16 = 2**16 - 1  # 65535
    
    print(f"Max token ID in vocabulary: {max_token_id:,}")
    print(f"Max uint16 value: {max_uint16:,}")
    
    if max_token_id > max_uint16:
        print(f"WARNING: Max token ID ({max_token_id}) exceeds uint16 range!")
        print("Consider using uint32 instead of uint16.")
        return False
    else:
        print("OK: Vocabulary fits in uint16 range.")
        return True

def encode_dataset_streaming(tokenizer, input_path, output_path, chunk_size=1024*1024):
    """
    Encode a dataset using streaming to handle large files efficiently.
    
    Args:
        tokenizer: Tokenizer instance
        input_path: Path to input text file
        output_path: Path to save encoded NumPy array
        chunk_size: Size of text chunks to process at once (in characters)
    """
    print(f"Encoding {input_path} -> {output_path}")
    
    all_token_ids = []
    total_chars = 0
    
    # Check if input file is gzipped
    is_gzipped = str(input_path).endswith('.gz')
    
    if is_gzipped:
        file_opener = lambda: gzip.open(input_path, 'rt', encoding='utf-8')
    else:
        file_opener = lambda: open(input_path, 'r', encoding='utf-8')
    
    # Get file size for progress bar
    if is_gzipped:
        # For gzipped files, we can't easily get uncompressed size
        print("Processing gzipped file (progress estimates may be inaccurate)...")
        pbar = None
    else:
        file_size = os.path.getsize(input_path)
        pbar = tqdm(total=file_size, unit='B', unit_scale=True, desc="Encoding")
    
    try:
        with file_opener() as f:
            buffer = ""
            
            while True:
                # Read chunk
                chunk = f.read(chunk_size)
                if not chunk:  # End of file
                    break
                
                buffer += chunk
                total_chars += len(chunk)
                
                if pbar:
                    pbar.update(len(chunk.encode('utf-8')))
                
                # Process complete lines to avoid splitting tokens
                lines = buffer.split('\n')
                buffer = lines[-1]  # Keep incomplete line in buffer
                
                for line in lines[:-1]:
                    if line.strip():  # Skip empty lines
                        token_ids = tokenizer.encode(line + '\n')
                        all_token_ids.extend(token_ids)
            
            # Process remaining buffer
            if buffer.strip():
                token_ids = tokenizer.encode(buffer)
                all_token_ids.extend(token_ids)
    
    finally:
        if pbar:
            pbar.close()
    
    # Convert to NumPy array with uint16 datatype
    print(f"Converting {len(all_token_ids):,} tokens to uint16 NumPy array...")
    
    try:
        token_array = np.array(all_token_ids, dtype=np.uint16)
    except OverflowError as e:
        print(f"Error: Token IDs too large for uint16! {e}")
        print("Using uint32 instead...")
        token_array = np.array(all_token_ids, dtype=np.uint32)
        # Update output path to reflect datatype change
        output_path = str(output_path).replace('.npy', '_uint32.npy')
    
    # Save to disk
    np.save(output_path, token_array)
    
    # Print statistics
    print(f"Saved {len(token_array):,} tokens to {output_path}")
    print(f"Array shape: {token_array.shape}")
    print(f"Array dtype: {token_array.dtype}")
    print(f"File size: {os.path.getsize(output_path) / (1024*1024):.1f} MB")
    print(f"Compression ratio: {total_chars / len(token_array):.2f} chars/token")
    
    return token_array

def encode_dataset_memory_efficient(tokenizer, input_path, output_path, max_tokens_in_memory=10_000_000):
    """
    Encode dataset with memory-efficient approach for very large files.
    Processes file in chunks and saves directly to disk.
    """
    print(f"Memory-efficient encoding: {input_path} -> {output_path}")
    
    # Check if input file is gzipped  
    is_gzipped = str(input_path).endswith('.gz')
    
    if is_gzipped:
        file_opener = lambda: gzip.open(input_path, 'rt', encoding='utf-8')
    else:
        file_opener = lambda: open(input_path, 'r', encoding='utf-8')
    
    # Create temporary files for chunked encoding
    temp_dir = Path(output_path).parent / "temp_encoding"
    temp_dir.mkdir(exist_ok=True)
    
    chunk_files = []
    total_tokens = 0
    chunk_num = 0
    
    try:
        with file_opener() as f:
            current_chunk = []
            
            # Use encode_iterable for memory efficiency
            for token_id in tokenizer.encode_iterable(f):
                current_chunk.append(token_id)
                
                # Save chunk when it reaches max size
                if len(current_chunk) >= max_tokens_in_memory:
                    chunk_file = temp_dir / f"chunk_{chunk_num}.npy"
                    chunk_array = np.array(current_chunk, dtype=np.uint16)
                    np.save(chunk_file, chunk_array)
                    chunk_files.append(chunk_file)
                    
                    total_tokens += len(current_chunk)
                    chunk_num += 1
                    current_chunk = []
                    
                    print(f"Saved chunk {chunk_num}: {total_tokens:,} tokens so far...")
            
            # Save final chunk
            if current_chunk:
                chunk_file = temp_dir / f"chunk_{chunk_num}.npy"
                chunk_array = np.array(current_chunk, dtype=np.uint16)
                np.save(chunk_file, chunk_array)
                chunk_files.append(chunk_file)
                total_tokens += len(current_chunk)
        
        # Combine all chunks into final array
        print(f"Combining {len(chunk_files)} chunks into final array...")
        all_chunks = []
        for chunk_file in chunk_files:
            chunk_data = np.load(chunk_file)
            all_chunks.append(chunk_data)
        
        final_array = np.concatenate(all_chunks)
        np.save(output_path, final_array)
        
        # Clean up temp files
        for chunk_file in chunk_files:
            chunk_file.unlink()
        temp_dir.rmdir()
        
        print(f"Saved {len(final_array):,} tokens to {output_path}")
        print(f"Array shape: {final_array.shape}")
        print(f"Array dtype: {final_array.dtype}")
        print(f"File size: {os.path.getsize(output_path) / (1024*1024):.1f} MB")
        
        return final_array
        
    except Exception as e:
        # Clean up temp files on error
        for chunk_file in chunk_files:
            if chunk_file.exists():
                chunk_file.unlink()
        if temp_dir.exists():
            temp_dir.rmdir()
        raise e

def main():
    """Main encoding function"""
    print("=" * 60)
    print("DATASET ENCODING TO NUMPY ARRAYS")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path("../encoded_datasets")
    output_dir.mkdir(exist_ok=True)
    
    # Dataset paths
    data_dir = Path("../data")
    
    datasets = {
        "tinystories_train": data_dir / "TinyStoriesV2-GPT4-train.txt",
        "tinystories_valid": data_dir / "TinyStoriesV2-GPT4-valid.txt",
        "openwebtext_train": data_dir / "owt_train.txt",
        "openwebtext_valid": data_dir / "owt_valid.txt",
    }
    
    # Check which datasets exist
    available_datasets = {}
    for name, path in datasets.items():
        if path.exists():
            available_datasets[name] = path
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"Found {name}: {path} ({size_mb:.1f} MB)")
        else:
            print(f"Missing {name}: {path}")
    
    if not available_datasets:
        print("No datasets found! Please check data directory.")
        return
    
    # Load tokenizers
    tokenizers = {}
    
    # TinyStories tokenizer
    ts_vocab_path = Path("../results/tinystories_bpe_output/vocab.json")
    ts_merges_path = Path("../results/tinystories_bpe_output/merges.json")
    
    if ts_vocab_path.exists() and ts_merges_path.exists():
        print(f"\\nLoading TinyStories tokenizer...")
        tokenizers["tinystories"] = Tokenizer.from_files(
            str(ts_vocab_path), 
            str(ts_merges_path)
        )
        check_vocab_size_compatibility(tokenizers["tinystories"])
    else:
        print("TinyStories tokenizer not found!")
    
    # OpenWebText tokenizer (check if it exists)
    owt_vocab_path = Path("owt_bpe_output/vocab.json")
    owt_merges_path = Path("owt_bpe_output/merges.json")
    
    if owt_vocab_path.exists() and owt_merges_path.exists():
        print(f"\\nLoading OpenWebText tokenizer...")
        tokenizers["openwebtext"] = Tokenizer.from_files(
            str(owt_vocab_path), 
            str(owt_merges_path)
        )
        check_vocab_size_compatibility(tokenizers["openwebtext"])
    else:
        print("OpenWebText tokenizer not found - will train it first or use TinyStories tokenizer")
    
    if not tokenizers:
        print("No tokenizers available!")
        return
    
    # Encode datasets
    print(f"\\n" + "=" * 60)
    print("ENCODING DATASETS")
    print("=" * 60)
    
    for dataset_name, dataset_path in available_datasets.items():
        print(f"\\nProcessing {dataset_name}...")
        
        # Choose appropriate tokenizer
        if dataset_name.startswith("tinystories") and "tinystories" in tokenizers:
            tokenizer = tokenizers["tinystories"]
            tokenizer_name = "tinystories"
        elif dataset_name.startswith("openwebtext") and "openwebtext" in tokenizers:
            tokenizer = tokenizers["openwebtext"] 
            tokenizer_name = "openwebtext"
        elif "tinystories" in tokenizers:
            # Fallback to TinyStories tokenizer
            tokenizer = tokenizers["tinystories"]
            tokenizer_name = "tinystories"
            print(f"Using TinyStories tokenizer for {dataset_name}")
        else:
            print(f"No suitable tokenizer for {dataset_name}")
            continue
        
        # Output file path
        output_path = output_dir / f"{dataset_name}_{tokenizer_name}_uint16.npy"
        
        # Skip if already encoded
        if output_path.exists():
            print(f"Already encoded: {output_path}")
            continue
        
        # Choose encoding method based on file size
        file_size_mb = os.path.getsize(dataset_path) / (1024 * 1024)
        
        try:
            if file_size_mb > 100:  # Use memory-efficient method for large files
                encode_dataset_memory_efficient(tokenizer, dataset_path, output_path)
            else:
                encode_dataset_streaming(tokenizer, dataset_path, output_path)
                
        except Exception as e:
            print(f"Error encoding {dataset_name}: {e}")
            continue
    
    print(f"\\n" + "=" * 60)
    print("ENCODING COMPLETE")
    print("=" * 60)
    
    # List all encoded files
    encoded_files = list(output_dir.glob("*.npy"))
    if encoded_files:
        print("\\nEncoded datasets:")
        for file_path in sorted(encoded_files):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            array = np.load(file_path, mmap_mode='r')  # Memory-mapped for large files
            print(f"  {file_path.name}: {len(array):,} tokens, {size_mb:.1f} MB")
    else:
        print("No datasets were encoded.")

if __name__ == "__main__":
    main()