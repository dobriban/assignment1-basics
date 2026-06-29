#!/usr/bin/env python3
"""
Tokenizer experiments to analyze compression ratios on TinyStories and OpenWebText.
"""

import sys
import os
import random
import gzip
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tokenizer import Tokenizer
from pathlib import Path

def load_tinystories_tokenizer():
    """Load the TinyStories tokenizer (10K vocab)"""
    results_dir = Path("../results/tinystories_bpe_output")
    vocab_path = results_dir / "vocab.json" 
    merges_path = results_dir / "merges.json"
    
    if not vocab_path.exists() or not merges_path.exists():
        # Try the sample output
        results_dir = Path("../results/tinystories_bpe_sample_output")
        vocab_path = results_dir / "vocab.json"
        merges_path = results_dir / "merges.json"
        
    if not vocab_path.exists() or not merges_path.exists():
        raise FileNotFoundError("TinyStories tokenizer not found. Please train it first.")
        
    return Tokenizer.from_files(str(vocab_path), str(merges_path))

def sample_documents_tinystories(n_docs=10):
    """Sample n documents from TinyStories dataset"""
    data_path = Path("../data/TinyStoriesV2-GPT4-train.txt")
    
    print(f"Sampling {n_docs} documents from TinyStories...")
    
    # Read the entire file and split by story boundaries
    with open(data_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Stories are typically separated by double newlines or story markers
    # Let's split by double newlines and filter out very short segments
    stories = content.split('\n\n')
    stories = [story.strip() for story in stories if len(story.strip()) > 100]  # Filter short segments
    
    print(f"Found {len(stories)} potential stories")
    
    # Randomly sample n documents
    if len(stories) < n_docs:
        print(f"Warning: Only {len(stories)} stories available, using all")
        sampled = stories
    else:
        sampled = random.sample(stories, n_docs)
    
    return sampled

def sample_documents_openwebtext(n_docs=10):
    """Sample n documents from OpenWebText dataset"""
    data_path = Path("../data/owt_train.txt.gz")
    
    print(f"Sampling {n_docs} documents from OpenWebText...")
    
    documents = []
    current_doc = []
    
    # Read compressed file line by line
    with gzip.open(data_path, 'rt', encoding='utf-8') as f:
        lines_read = 0
        for line in f:
            lines_read += 1
            line = line.strip()
            
            # Empty line often indicates document boundary
            if not line:
                if current_doc:
                    doc_text = '\\n'.join(current_doc)
                    if len(doc_text) > 100:  # Filter very short documents
                        documents.append(doc_text)
                    current_doc = []
            else:
                current_doc.append(line)
            
            # Stop after reading a reasonable number of lines to get enough documents
            if len(documents) >= n_docs * 5:  # Get more than needed for sampling
                break
                
            if lines_read % 10000 == 0:
                print(f"Read {lines_read} lines, found {len(documents)} documents...")
    
    # Add final document if exists
    if current_doc:
        doc_text = '\\n'.join(current_doc)
        if len(doc_text) > 100:
            documents.append(doc_text)
    
    print(f"Found {len(documents)} documents")
    
    # Randomly sample n documents
    if len(documents) < n_docs:
        print(f"Warning: Only {len(documents)} documents available, using all")
        return documents
    else:
        return random.sample(documents, n_docs)

def calculate_compression_ratio(documents, tokenizer, tokenizer_name):
    """Calculate compression ratio (bytes/token) for given documents and tokenizer"""
    print(f"\\nCalculating compression ratio for {tokenizer_name}...")
    
    total_bytes = 0
    total_tokens = 0
    
    results = []
    
    for i, doc in enumerate(documents):
        # Calculate bytes (UTF-8 encoding)
        doc_bytes = len(doc.encode('utf-8'))
        
        # Tokenize document
        token_ids = tokenizer.encode(doc)
        num_tokens = len(token_ids)
        
        # Calculate compression ratio for this document
        ratio = doc_bytes / num_tokens if num_tokens > 0 else 0
        
        results.append({
            'doc_id': i + 1,
            'bytes': doc_bytes,
            'tokens': num_tokens,
            'ratio': ratio,
            'preview': doc[:100] + "..." if len(doc) > 100 else doc
        })
        
        total_bytes += doc_bytes
        total_tokens += num_tokens
        
        print(f"Doc {i+1:2d}: {doc_bytes:5d} bytes, {num_tokens:4d} tokens, ratio: {ratio:.2f}")
    
    overall_ratio = total_bytes / total_tokens if total_tokens > 0 else 0
    
    print(f"\\nOverall for {tokenizer_name}:")
    print(f"Total bytes: {total_bytes:,}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Compression ratio: {overall_ratio:.3f} bytes/token")
    
    return {
        'tokenizer_name': tokenizer_name,
        'total_bytes': total_bytes,
        'total_tokens': total_tokens,
        'overall_ratio': overall_ratio,
        'documents': results
    }

def main():
    """Run tokenizer experiments"""
    print("=" * 60)
    print("TOKENIZER COMPRESSION EXPERIMENTS")
    print("=" * 60)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    try:
        # Load TinyStories tokenizer
        print("\\n1. Loading TinyStories tokenizer...")
        tinystories_tokenizer = load_tinystories_tokenizer()
        print("OK: TinyStories tokenizer loaded")
        
        # For now, we'll use the TinyStories tokenizer for OpenWebText too
        # since we don't have a trained OpenWebText tokenizer yet
        print("\\n2. Using TinyStories tokenizer for both datasets")
        print("   (OpenWebText tokenizer would need to be trained separately)")
        
        # Sample documents from TinyStories
        print("\\n3. Sampling documents from TinyStories...")
        tinystories_docs = sample_documents_tinystories(10)
        print(f"OK: Sampled {len(tinystories_docs)} TinyStories documents")
        
        # Sample documents from OpenWebText  
        print("\\n4. Sampling documents from OpenWebText...")
        openwebtext_docs = sample_documents_openwebtext(10)
        print(f"OK: Sampled {len(openwebtext_docs)} OpenWebText documents")
        
        # Calculate compression ratios
        print("\\n5. Calculating compression ratios...")
        
        ts_results = calculate_compression_ratio(
            tinystories_docs, 
            tinystories_tokenizer, 
            "TinyStories Tokenizer on TinyStories"
        )
        
        owt_results = calculate_compression_ratio(
            openwebtext_docs, 
            tinystories_tokenizer, 
            "TinyStories Tokenizer on OpenWebText"
        )
        
        # Summary comparison
        print("\\n" + "=" * 60)
        print("SUMMARY COMPARISON")
        print("=" * 60)
        print(f"TinyStories tokenizer on TinyStories:   {ts_results['overall_ratio']:.3f} bytes/token")
        print(f"TinyStories tokenizer on OpenWebText:   {owt_results['overall_ratio']:.3f} bytes/token")
        
        ratio_diff = owt_results['overall_ratio'] - ts_results['overall_ratio']
        print(f"\\nDifference: {ratio_diff:+.3f} bytes/token")
        if ratio_diff > 0:
            print("=> Tokenizer is less efficient on OpenWebText (out-of-domain)")
        else:
            print("=> Tokenizer is more efficient on OpenWebText")
            
        # Save results
        results = {
            'tinystories_results': ts_results,
            'openwebtext_results': owt_results,
            'summary': {
                'ts_ratio': ts_results['overall_ratio'],
                'owt_ratio': owt_results['overall_ratio'],
                'difference': ratio_diff
            }
        }
        
        with open('tokenizer_experiment_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\\nOK: Results saved to tokenizer_experiment_results.json")
        
    except Exception as e:
        print(f"\\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()