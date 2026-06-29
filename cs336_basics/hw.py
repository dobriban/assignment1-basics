# Problem (unicode1): Understanding Unicode (1 point)
# %%
a = chr(313131)  #Transform into a unicode character. NUL character
print(a) #But it is empty when printed. 

b = a.__repr__() #'\x00'
print(b)


print("this is a test" + a + "string") #Does not show up 

############
# %%
test_string = "hello! こんにちは!"
utf_encoded = test_string.encode("utf-8")
print(utf_encoded)
#print(type(utf_encoded))
# Get the byte values for the encoded string (integers from 0 to 255).
list(utf_encoded)

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
# a = re.findall(PAT, "some text that i'll pre-tokenize")
# print(a)

a = re.finditer(PAT, "some text that i'll pre-tokenize")
print(a)

#use re.finditer to avoid storing the pre-tokenized words as you construct your mapping from pre-tokens to their count 

# %%
