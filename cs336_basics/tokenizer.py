import json
import pickle
import regex as re
from typing import Iterator, Iterable


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        """
        Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens.
        
        Args:
            vocab: dict[int, bytes] - mapping from token ID to token bytes
            merges: list[tuple[bytes, bytes]] - list of BPE merges in order of creation
            special_tokens: list[str] | None - list of special tokens to handle
        """
        self.vocab = vocab.copy()
        self.merges = merges.copy()
        self.special_tokens = special_tokens or []
        
        # Create reverse vocab for encoding (bytes -> token_id)
        self.byte_to_id = {v: k for k, v in self.vocab.items()}
        
        # Add special tokens to vocab if not already present
        next_token_id = max(self.vocab.keys()) + 1 if self.vocab else 0
        for special_token in self.special_tokens:
            special_bytes = special_token.encode("utf-8")
            if special_bytes not in self.byte_to_id:
                self.vocab[next_token_id] = special_bytes
                self.byte_to_id[special_bytes] = next_token_id
                next_token_id += 1
        
        # Pre-tokenization pattern (same as in train_bpe.py)
        self.pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        
        # Create merge lookup for faster BPE application
        self.merge_dict = {}
        for merge in self.merges:
            token1, token2 = merge
            # Find token IDs for the merge pair
            token1_id = self.byte_to_id.get(token1)
            token2_id = self.byte_to_id.get(token2)
            if token1_id is not None and token2_id is not None:
                merged_bytes = token1 + token2
                merged_id = self.byte_to_id.get(merged_bytes)
                if merged_id is not None:
                    self.merge_dict[(token1_id, token2_id)] = merged_id

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        """
        Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges.
        
        Args:
            vocab_filepath: str - path to vocabulary file
            merges_filepath: str - path to merges file
            special_tokens: list[str] | None - list of special tokens
            
        Returns:
            Tokenizer instance
        """
        # Load vocab
        if vocab_filepath.endswith('.json'):
            with open(vocab_filepath, 'r', encoding='utf-8') as f:
                vocab_raw = json.load(f)
            # Convert string keys to int and string values to bytes
            vocab = {}
            for k, v in vocab_raw.items():
                if isinstance(v, str):
                    vocab[int(k)] = v.encode('utf-8')
                else:  # already bytes
                    vocab[int(k)] = bytes(v)
        elif vocab_filepath.endswith('.pkl'):
            with open(vocab_filepath, 'rb') as f:
                vocab = pickle.load(f)
        else:
            raise ValueError(f"Unsupported vocab file format: {vocab_filepath}")
        
        # Load merges
        if merges_filepath.endswith('.json'):
            with open(merges_filepath, 'r', encoding='utf-8') as f:
                merges_raw = json.load(f)
            merges = []
            for merge in merges_raw:
                if len(merge) == 2:
                    if isinstance(merge[0], str):
                        merges.append((merge[0].encode('utf-8'), merge[1].encode('utf-8')))
                    else:
                        merges.append((bytes(merge[0]), bytes(merge[1])))
        elif merges_filepath.endswith('.pkl'):
            with open(merges_filepath, 'rb') as f:
                merges = pickle.load(f)
        else:
            raise ValueError(f"Unsupported merges file format: {merges_filepath}")
        
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.
        
        Args:
            text: str - input text to encode
            
        Returns:
            list[int] - sequence of token IDs
        """
        if not text:
            return []
        
        # Handle special tokens by splitting text around them
        if self.special_tokens:
            # Create pattern that matches any special token
            special_pattern = "|".join(re.escape(token) for token in self.special_tokens)
            # Split text while keeping delimiters
            parts = re.split(f'({special_pattern})', text)
            
            token_ids = []
            for part in parts:
                if not part:
                    continue
                if part in self.special_tokens:
                    # This is a special token
                    special_bytes = part.encode('utf-8')
                    token_id = self.byte_to_id.get(special_bytes)
                    if token_id is not None:
                        token_ids.append(token_id)
                else:
                    # Regular text - apply BPE
                    token_ids.extend(self._encode_text(part))
            
            return token_ids
        else:
            return self._encode_text(text)

    def _encode_text(self, text: str) -> list[int]:
        """
        Encode regular text (no special tokens) using BPE.
        
        Args:
            text: str - text to encode
            
        Returns:
            list[int] - sequence of token IDs
        """
        if not text:
            return []
        
        # Pre-tokenize using regex pattern
        pre_tokens = re.findall(self.pat, text)
        
        all_token_ids = []
        for pre_token in pre_tokens:
            # Convert pre-token to UTF-8 bytes
            pre_token_bytes = pre_token.encode('utf-8')
            
            # Initialize with individual byte tokens
            token_sequence = []
            for byte_val in pre_token_bytes:
                token_sequence.append(self.byte_to_id.get(bytes([byte_val]), byte_val))
            
            # Apply BPE merges in order
            for merge in self.merges:
                token1_bytes, token2_bytes = merge
                token1_id = self.byte_to_id.get(token1_bytes)
                token2_id = self.byte_to_id.get(token2_bytes)
                merged_bytes = token1_bytes + token2_bytes
                merged_id = self.byte_to_id.get(merged_bytes)
                
                if token1_id is None or token2_id is None or merged_id is None:
                    continue
                
                # Apply this merge to the token sequence
                new_sequence = []
                i = 0
                while i < len(token_sequence):
                    if (i < len(token_sequence) - 1 and 
                        token_sequence[i] == token1_id and 
                        token_sequence[i + 1] == token2_id):
                        new_sequence.append(merged_id)
                        i += 2
                    else:
                        new_sequence.append(token_sequence[i])
                        i += 1
                token_sequence = new_sequence
            
            all_token_ids.extend(token_sequence)
        
        return all_token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs.
        
        Args:
            iterable: Iterable[str] - iterable of strings
            
        Returns:
            Iterator[int] - generator yielding token IDs
        """
        buffer = ""
        
        for chunk in iterable:
            buffer += chunk
            
            # Process complete lines/chunks to avoid breaking tokens across boundaries
            # For now, we'll process the buffer when we have newlines or when we reach a reasonable size
            while '\n' in buffer or len(buffer) > 1024:
                if '\n' in buffer:
                    # Process up to and including the last newline
                    last_newline = buffer.rfind('\n')
                    to_process = buffer[:last_newline + 1]
                    buffer = buffer[last_newline + 1:]
                else:
                    # Process most of the buffer, but keep some in case we're in the middle of a token
                    to_process = buffer[:1024]
                    buffer = buffer[1024:]
                
                if to_process:
                    token_ids = self.encode(to_process)
                    yield from token_ids
        
        # Process any remaining buffer
        if buffer:
            token_ids = self.encode(buffer)
            yield from token_ids

    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        
        Args:
            ids: list[int] - sequence of token IDs
            
        Returns:
            str - decoded text
        """
        if not ids:
            return ""
        
        # Collect all bytes
        all_bytes = bytearray()
        for token_id in ids:
            token_bytes = self.vocab.get(token_id)
            if token_bytes is not None:
                all_bytes.extend(token_bytes)
            else:
                # Unknown token ID - skip it (could also add a replacement character)
                continue
        
        # Decode bytes to string, replacing malformed sequences
        try:
            return bytes(all_bytes).decode('utf-8', errors='replace')
        except UnicodeDecodeError:
            return bytes(all_bytes).decode('utf-8', errors='replace')