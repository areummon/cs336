import regex as re
from collections.abc import Iterable, Iterator
from bpe_tokenizer import bytes_to_unicode, _generate_chunks
import json, time
from functools import partial
import os
from multiprocessing import Pool
import numpy as np

class Tokenizer:

    _PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens=None):
        self.vocab = vocab
        self.vocab_ids = {value: key for key, value in vocab.items()}
        # deliverate design decision to speed up the merges in the pre-tokenize
        # step
        self.merges = {pair: rank for rank, pair in enumerate(merges)}
        self.special_tokens = special_tokens

    @classmethod
    def from_files(cls, vocab_filepath="vocab.json", merges_filepath="merges.txt", special_tokens=None):
        btu = bytes_to_unicode()
        decoder = {char: byte for byte, char in btu.items()}
        with open(vocab_filepath) as f:
            raw = json.load(f)
        vocab = {int(id_str): bytes(decoder[ch] for ch in encoded) for id_str, encoded in raw.items()}
        merges = []
        with open(merges_filepath) as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                left, right = line.split(" ")
                merges.append((bytes(decoder[ch] for ch in left), bytes(decoder[ch] for ch in right)))
        return Tokenizer(vocab, merges, special_tokens)

    def _encode_ids(self, piece: str):
        parts = []
        if self.special_tokens:
            # The special tokens should be checked from the longest to the smallest
            # because there are edge cases where two same special tokens could
            # be one after another as a unique special token.
            sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
            pattern = "(" + "|".join(re.escape(t) for t in sorted_specials) + ")" 
            parts = re.split(pattern, piece)
        else:
            parts = [piece]
        for part in parts:
            if part == "": #maybe not necessary
                continue
            if self.special_tokens and part in self.special_tokens:
                yield self.vocab_ids[part.encode("utf-8")]
                continue
            for token in re.finditer(self._PAT, part):
                token_bytes = tuple(bytes([b]) for b in token.group().encode("utf-8"))
                new_token = token_bytes
                merge = True
                while merge:
                    new_token_list = []
                    ranks = []
                    merge = False
                    i = 0
                    # Gets the possible merges a string or token can apply
                    for pair in zip(new_token[:-1], new_token[1:]):
                        if pair in self.merges:
                            ranks.append((pair, self.merges[pair]))
                    if not ranks:
                        break
                    # We need to get the minimum in the rank (i.e. the first to
                    # appear in the merge list)
                    min_pair = min(ranks, key=lambda x: x[1])
                    x,y = min_pair[0]
                    while i < len(new_token):
                        if i+1 < len(new_token) and new_token[i] == x and new_token[i+1] == y:
                            new_token_list.append(x+y)
                            i += 1
                            merge = True
                        else:
                            new_token_list.append(new_token[i])
                        i += 1
                    new_token = tuple(new_token_list)
                for sym in new_token:
                    yield self.vocab_ids[sym]
    
    def encode(self, text: str) -> list[int]:
        return list(self._encode_ids(text))

    # We need to yield the ids because it returns an iteratior of ints
    # This is in order to be memory efficient.
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for piece in iterable:
            yield from self._encode_ids(piece)

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

def sample_documents(path, num_docs=10):
    docs = []
    with open(path, "rb") as f:
        for chunk in _generate_chunks(f, 4):
            for doc in chunk.split("<|endoftext|>"):
                doc = doc.strip()
                if not doc:
                    continue
                docs.append(doc)
                if len(docs) >= num_docs:
                    return docs
    return docs

def _encode_chunk(tokenizer, chunk):
    return np.fromiter(tokenizer.encode(chunk), dtype=np.uint16)

# Currently this function works only when the special token is <|endoftext|>
def serialize(in_path, out_path, tokenizer):
    num_processes = os.cpu_count()
    frozen = partial(_encode_chunk, tokenizer)
    with open(out_path, "wb") as out, Pool(num_processes) as pool: 
        with open(in_path, "rb") as f:
            for ids in pool.imap(frozen, _generate_chunks(f, num_processes*4)):
                out.write(ids.tobytes())

if __name__ == "__main__":
   #string = 'the cat ate <|endoftext|> some food'
   # string = "the cat ate"
   # vocab = {0: b' ', 1: b'a', 2:  b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at', 11: "<|endoftext|>".encode("utf-8")}
   # merges = [(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't'), (b'o', b'm'), (b'g', b'a')]
   # tokenizer = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])
   # ids = tokenizer.encode(string)
   # print(ids)
   tinystories_tokenizer = Tokenizer.from_files("tinystories/vocab.json", "tinystories/merges.txt")
   ow_tokenizer = Tokenizer.from_files("owt/vocab.json", "owt/merges.txt")
   #tinystories_docs = sample_documents("../data/TinyStoriesV2-GPT4-train.txt")
   #owt_docs = sample_documents("../data/owt_valid.txt")
   #total_bytes_tiny = sum(len(doc.encode("utf-8")) for doc in tinystories_docs)
   #all_docs = "".join(doc for doc in tinystories_docs)
   ##t0 = time.perf_counter()
   #ids = tinystories_tokenizer.encode(all_docs)
   #total_tokens_tiny = sum(len(tinystories_tokenizer.encode(doc)) for doc in tinystories_docs)
   ##t1 = time.perf_counter()
   ##elapsed_seconds = t1 - t0
   ##throughput = total_bytes_tiny / elapsed_seconds
   #ratio_tinystories = total_bytes_tiny / total_tokens_tiny
   #total_bytes_owt = sum(len(doc.encode("utf-8")) for doc in owt_docs)
   #total_tokens_owt = sum(len(ow_tokenizer.encode(doc)) for doc in owt_docs)
   #ratio_owt = total_bytes_owt / total_tokens_owt
   #print(f"TinyStories: Total Bytes {total_bytes_tiny} \t Total Tokens {total_tokens_tiny}")
   #print(f"Owt: Total Bytes {total_bytes_owt} \t Total Tokens {total_tokens_owt}")
   #print(f"The compression ratio for tinystories is {ratio_tinystories} and for owt is {ratio_owt}")
   #total_bytes = sum(len(doc.encode("utf-8")) for doc in owt_docs)
   #total_tokens = sum(len(tinystories_tokenizer.encode(doc)) for doc in owt_docs)
   #ratio = total_bytes / total_tokens
   #print(f"Total Bytes {total_bytes} \t Total Tokens {total_tokens}")
   #print(f"Compression ration: {ratio}")
   #print(825000000000/429305.0384856463)
   serialize("../data/TinyStoriesV2-GPT4-train.txt", "tinystories-train-encoded.bin", tinystories_tokenizer)
   serialize("../data/TinyStoriesV2-GPT4-valid.txt", "tinystories-valid-encoded.bin", tinystories_tokenizer)
   serialize("../data/owt_train.txt", "owt-train-encoded.bin", ow_tokenizer)
   serialize("../data/owt_valid.txt", "owt-valid-encoded.bin", ow_tokenizer)

