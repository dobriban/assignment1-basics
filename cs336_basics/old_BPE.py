#BPE
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass

#Start with some BPE utilities. 
class Tokenizer(ABC):
    @abstractmethod
    def encode(self, string:str) -> list[int]:
        raise NotImplementedError

    @abstractmethod
    def decode(self, indices: list[int]) -> str:
        raise NotImplementedError

@dataclass(frozen=True)
class BPETokenizerParams:
    vocab: dict[int,bytes]
    merges: list[tuple[tuple[int,int],int]]



#Merge the tuple of integers p into i in ints. 
def merge(ints:list[int],p:tuple[int,int],i:int)->list[int]:
    ints_merged = []
    l=0
    k=0
    while l < len(ints):
        if l+1 < len(ints) and ints[l]==p[0] and ints[l+1]==p[1]:
            ints_merged.append(i)
            l+=2
        else:
            ints_merged.append(ints[l])
            l+=1
        k+=1
    return ints_merged

# ints = [1,2,3,2,3]
# p=(2,3)
# i= 4
# print(merge(ints,p,i))

#Find the most frequently occurring pair of integers in `ints`. 
def top_pair(ints:list[int])->tuple[int,int]:
    cand = defaultdict(int)
    for i in range(len(ints)):
        if i+1< len(ints):
            cand[(ints[i],ints[i+1])]+=1
        #A bit of a strange behavior here. 
        #[1,1,1] Counts as two pairs of [1,1],
        # Even though when applying BPE you only end up with one pair
    top = max(cand, key=cand.get)
    return top, cand[top]



# ints = [1,2,3,2,3]
# print(top_pair(ints))

#s = "1111222334"

# Code for training the BPE tokenizer 
#s = "The cat sat on the mat. " #String to train tokenizer on  


class BPETokenizer(Tokenizer):
    def __init__(self, params: BPETokenizerParams):
        self.params = params
    
    def encode(self,s:str)->list[int]:
        ints = list(map(int,s.encode("utf-8")))
        original_length = len(ints)
        for i in range(len(self.params.merges)):
            p,new_ind = self.params.merges[i]
            ints = merge(ints,p,new_ind)
        compressed_length = len(ints)
        if original_length>0:
            compression_ratio = compressed_length/original_length
        return ints, compression_ratio

    def decode(self,ints:list[int])-> str:
        tokens = b"".join(self.params.vocab[i] for i in ints)
        return tokens.decode("utf-8")

#print(ints)
# Actual code for training tokenizer  

def trainBPE(s:str, num_merges:int)->BPETokenizerParams:
    ints = list(map(int,s.encode("utf-8"))) #We'll work on a list of integers. 
    vocab = {x:bytes([x]) for x in range(256)} #Dictionary for encoding and decoding integers to bytes. 
    merges = []

    for i in range(num_merges):
        if len(ints)>1:
            p, numi  = top_pair(ints)
            if numi==1:
                break
            else: 
                new_ind = 256+i
                ints = merge(ints,p,new_ind)
                vocab[new_ind] = vocab[p[0]]+vocab[p[1]]
                merges.append((p,new_ind))
                #print(ints)
    return BPETokenizerParams(vocab=vocab,merges=merges)

# print(vocab)
# print(merges)




# Code for testing, encoding, decoding, and tokenization 
#s = "The quick brown fox "
s = "Names of type variables introduced in PEP 484 should normally use CapWords preferring short names: T, AnyStr, Num. It is recommended to add suffixes _co or _contra to the variables used to declare covariant or contravariant behavior correspondingly: from typing import TypeVar VT_co = TypeVar('VT_co', covariant=True) KT_contra = TypeVar('KT_contra', contravariant=True)" 
num_merges = 200

par = trainBPE(s,num_merges)
tok = BPETokenizer(par)

s = "When the conditional part of an if-statement is long enough to require that it be written across multiple lines, it’s worth noting that the combination of a two character keyword (i.e. if), plus a single space, plus an opening parenthesis creates a natural 4-space indent for the subsequent lines of the multiline conditional. This can produce a visual conflict with the indented suite of code nested inside the if-statement, which would also naturally be indented to 4 spaces. This PEP takes no explicit position on how (or whether) to further visually distinguish such conditional lines from the nested suite inside the if-statement. Acceptable options in this situation include, but are not limited to: # No extra indentation. if (this_is_one_thing and that_is_another_thing): do_something() # Add a comment, which will provide some distinction in editors # supporting syntax highlighting. if (this_is_one_thing and that_is_another_thing): # Since both conditions are true, we can frobnicate. do_something() # Add some extra indentation on the conditional continuation line. if (this_is_one_thing and that_is_another_thing): do_something()"

ints, c = tok.encode(s) #print(ints)

print(f"Compression ratio is {c:.2f}")

s_dec = tok.decode(ints)
#print(s_dec)
if s == s_dec:
    print("Correct")
else:
    print("Incorrect")




