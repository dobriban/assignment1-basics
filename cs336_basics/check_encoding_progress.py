#!/usr/bin/env python3
"""
Check the progress of dataset encoding.
"""

import os
import numpy as np
from pathlib import Path
import time

def check_progress():
    """Check encoding progress"""
    print("ENCODING PROGRESS CHECK")
    print("=" * 40)
    
    output_dir = Path("../encoded_datasets")
    
    if not output_dir.exists():
        print("No encoded_datasets directory found")
        return
    
    # Check for any .npy files
    npy_files = list(output_dir.glob("*.npy"))
    temp_dirs = list(output_dir.glob("temp_*"))
    
    if npy_files:
        print("COMPLETED FILES:")
        total_tokens = 0
        for file_path in sorted(npy_files):
            try:
                # Use memory mapping for large files
                array = np.load(file_path, mmap_mode='r')
                size_mb = os.path.getsize(file_path) / (1024*1024)
                print(f"  {file_path.name}")
                print(f"    Tokens: {len(array):,}")
                print(f"    Size: {size_mb:.1f} MB")
                print(f"    Dtype: {array.dtype}")
                total_tokens += len(array)
            except Exception as e:
                print(f"  {file_path.name}: Error reading - {e}")
        
        if total_tokens > 0:
            print(f"\n  Total tokens encoded: {total_tokens:,}")
    else:
        print("No completed .npy files found")
    
    if temp_dirs:
        print("\nIN PROGRESS:")
        for temp_dir in temp_dirs:
            chunk_files = list(temp_dir.glob("*.npy"))
            print(f"  {temp_dir.name}: {len(chunk_files)} chunks")
    
    # Check original dataset sizes for reference
    print("\nREFERENCE - Original Dataset Sizes:")
    data_dir = Path("../data")
    datasets = [
        ("TinyStories Train", "TinyStoriesV2-GPT4-train.txt"),
        ("TinyStories Valid", "TinyStoriesV2-GPT4-valid.txt"),
    ]
    
    for name, filename in datasets:
        file_path = data_dir / filename
        if file_path.exists():
            size_mb = os.path.getsize(file_path) / (1024*1024)
            print(f"  {name}: {size_mb:.1f} MB")
    
    # Estimated completion
    print("\nESTIMATES:")
    print("  Expected validation tokens: ~5.6M")
    print("  Expected training tokens: ~556M") 
    print("  Expected validation size: ~11 MB (uint16)")
    print("  Expected training size: ~1.1 GB (uint16)")

if __name__ == "__main__":
    check_progress()