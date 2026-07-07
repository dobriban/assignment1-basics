# Problem (unicode1): Understanding Unicode (1 point)
# %%
a = chr(313131)  #Transform into a unicode character. NUL character
print(a) #But it is empty when printed. 

b = a.__repr__() #'\x00'
print(b)


print("this is a test" + a + "string") #Does not show up 

############
# %%
test_string = "hこ"
utf_encoded = test_string.encode("utf-8")
print(utf_encoded)
#print(type(utf_encoded))
# Get the byte values for the encoded string (integers from 0 to 255).
b = list(utf_encoded)
print(b)

s = utf_encoded.decode("utf-8")
print(s)

# Problem (unicode2): Unicode Encodings (3 points)
# %%
# (a)
utf_encoded = test_string.encode("utf-16")
print(utf_encoded)
utf_encoded = test_string.encode("utf-32")
print(utf_encoded)

# %%
# (b)
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])

a = decode_utf8_bytes_to_str_wrong("こ".encode("utf-8"))
print(a)

#Same issue that I had: it tries to decode bytes one at a time. 
#It should join the bytes first. 

# %%
# (c)
def is_valid_utf8(b: bytes) -> bool:
    try:
        b.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False

n = 256
find_val = []
for x in range(n):
    for y in range(n):
        b = bytes([x,y])
        if not is_valid_utf8(b):
            find_val = [x,y]  
        
print(find_val)

#Answer: found [255, 255]

# %%
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

import regex as re
from collections import Counter

# a = re.findall(PAT, "some text that i'll pre-tokenize")
# print(a)

a = re.finditer(PAT, "some text that i'll pre-tokenize")
print(a)

for match in a:
    pretoken = match.group(0)
    print(repr(pretoken), match.span())

#use re.finditer to avoid storing the pre-tokenized words as you construct your mapping from pre-tokens to their count 

# %%

# TinyStories (training set) BPE answer:
# From TinyStories-train_res.json, the 10,000-token vocabulary has a
# four-way tie for longest token. Each is 15 bytes:
# " accomplishment", " responsibility", " disappointment",
# and " recommendation".
#
# Yes, this makes sense. BPE learns frequent byte sequences, so common
# long words in TinyStories can become full tokens. The leading space also
# makes sense because the pre-tokenizer groups an optional space with the
# following word.
#
# Resource check: training took 48.60 seconds and peaked at 980.86 MB RAM,
# so it is well within the <= 12 hours, no-GPU, <= 100 GB RAM requirement.

# TinyStories profiling answer:
# Profiling shows that pre-tokenization/counting dominates runtime: it took about
# 44.1 seconds, compared with about 3.2 seconds for the BPE merge loop. 



# %%
# Problem (train_bpe_tinystories): BPE Training on TinyStories (2 points)

# (a)
# From openwebtext_32000_res.json, the 32,000-token vocabulary has a
# two-way tie for longest token. Each is 64 bytes:
# "----------------------------------------------------------------"
# and the mojibake-looking byte sequence c382c383 repeated 16 times
# (decoded as "\xc2\xc3" repeated 16 times).
#
# This also makes sense, but for a different reason than TinyStories. OpenWebText
# is noisy web text, so frequent repeated separators and encoding artifacts can
# be common enough for byte-level BPE to merge into long tokens. These tokens are
# not semantically meaningful words; they reflect frequent byte patterns in the
# corpus.
#
# Resource check: the full run took 2719.79 seconds total, with 1369.81 seconds
# spent in final BPE training, and peaked at 24015.13 MB RAM. This is also within
# the <= 12 hours, no-GPU, <= 100 GB RAM requirement.

# (b) Tokenizer comparison:
# The TinyStories tokenizer learns many whole, common story-like English words,
# while the OpenWebText tokenizer also spends vocabulary slots on web artifacts
# such as long separators and encoding noise. Both are byte-level BPE tokenizers,
# but their learned vocabularies reflect the style and cleanliness of their
# training corpora.


# %%
# Problem (tokenizer_experiments): Experiments with tokenizers (4 points)
# (a) Sample 10 documents from TinyStories and OpenWebText. Using your previously-trained
# TinyStories and OpenWebText tokenizers (10K and 32K vocabulary size, respectively),
# encode these sampled documents into integer IDs. What is each tokenizer’s compression ratio
# (bytes/token)?
# Deliverable: A one-to-two sentence response.

# Using a fixed random seed of 0 and 10 sampled documents 
# from each validation set, 
# the TinyStories 10K tokenizer achieves 4.244 bytes/token on TinyStories, 
# while 
# the OpenWebText 32K tokenizer achieves 4.681 bytes/token on OpenWebText.


# (b) What happens if you tokenize your OpenWebText sample with the TinyStories tokenizer?
# Compare the compression ratio and/or qualitatively describe what happens.
# Deliverable: A one-to-two sentence response.

# (c) Estimate the throughput of your tokenizer (e.g., in bytes/second). How long would it take to
# tokenize the Pile dataset (825GB of text)?
# Deliverable: A one-to-two sentence response.

# (d) Using your TinyStories and OpenWebText tokenizers, encode the respective training and
# development datasets into a sequence of integer token IDs. We’ll use this later to train our
# language model. We recommend serializing the token IDs as a NumPy array of datatype
# uint16. Why is uint16 an appropriate choice?
# Deliverable: A one-to-two sentence response.