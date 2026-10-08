from collections import defaultdict, Counter
from typing import BinaryIO
from multiprocessing import Pool
from functools import partial
import regex as re 
import time, sys, os, heapq, resource, json, codecs

# Class to help getting the max_pair lexicographically in the heap.
# This should not be needed if the version of python is 3.14 because
# heappop_max exists. Basically A < B <-> B < A.
class InvertedPair:
    __slots__ = ("pair",)
    def __init__(self, pair): self.pair = pair
    def __lt__(self, other): return other.pair < self.pair
    def __eq__(self, other): return self.pair == other.pair

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

# Function to generate the chunks and yielding them to pass them to the workers
# or threads
def _generate_chunks(file, num_chunks):
    boundaries = find_chunk_boundaries(file, num_chunks, b"<|endoftext|>")
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        file.seek(start)
        yield file.read(end - start).decode("utf-8", errors="ignore")

# The pre_tokenizer by chunk, each worker has an assigned chunk and pre-tokenizes
# it, returning the Counter
def _pre_tokenize_chunk(PAT, delimiter, chunk) -> Counter:
    chunk_tokens = Counter()
    for doc in re.split(delimiter, chunk):
        for string in re.finditer(PAT, doc):
            chunk_tokens[string.group()] += 1
    return chunk_tokens

def _pretokenize_parallel(input_path, special_tokens, num_processses):
    t_pre_tokenize_init = time.time()
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    delimiter = "|".join(re.escape(tok) for tok in special_tokens)
    pre_tokens = Counter()
    with open(input_path, "rb") as f:
        num_chunks = num_processses * 4
        with Pool(num_processses) as p:
            # Use partial() to pass multiple arguments to the worker, in this case
            # because I have to pass PAT and delimiter
            frozen_func = partial(_pre_tokenize_chunk, PAT, delimiter)
            # It need to use imap because the chunks are generated with a 
            # generating function
            chunk_pre_tokens = p.imap_unordered(frozen_func, _generate_chunks(f, num_chunks))
            t0 = time.time()
            for cpr in chunk_pre_tokens:
                pre_tokens.update(cpr)
            pre_tokens = {tuple(bytes([b]) for b in w.encode("utf-8")): n for w, n in pre_tokens.items()}
            t_parent = time.time() - t0 
            print(f"The parent took {t_parent:.2f}s to update the counter")
            t_pre_tokenize_end = time.time()
    print(f"time of pre-tokenization: {t_pre_tokenize_end - t_pre_tokenize_init:.2f}")
    return pre_tokens

def _merge_loop(pre_tokens, vocab_size, special_tokens):
    vocab = {n: bytes([n]) for n in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")
    t_merge_init = time.time()
    pair_counts = defaultdict(int)
    pair_index = defaultdict(set)
    merges_count = vocab_size - 256 - len(special_tokens)
    merges = []
    for token in pre_tokens:
        for pair in zip(token[:-1], token[1:]):
            pair_counts[pair] += pre_tokens[token]
            pair_index[pair].add(token)
    # Usage of a max heap to get the max of the pair_counts.
    # This amortizes and offers a better complexity time.
    heap = [(-count, InvertedPair(pair)) for pair, count in pair_counts.items()]
    heapq.heapify(heap)
    for j in range(merges_count):
        print(j)
        max_pair = None
        while heap:
            neg_count, inverted_pair = heapq.heappop(heap)
            pair = inverted_pair.pair
            # The pair with the maximum value could have changed in the
            # update parte, and no longer be valid, thats why
            # I discard the elements that are no longer valid.
            if -neg_count == pair_counts.get(pair, 0):
                max_pair = pair
                break
        x,y = max_pair
        affected = pair_index[max_pair].copy() #pair_index not mutable
        for pre_token in affected:
            n = pre_tokens[pre_token]
            for pair in zip(pre_token[:-1], pre_token[1:]):
                pair_counts[pair] -= n
                if pair_counts[pair] <= 0:
                   del pair_counts[pair]
                else: # Because if its <= 0 we dont need to add it
                    heapq.heappush(heap, (-pair_counts[pair], InvertedPair(pair)))
                pair_index[pair].discard(pre_token)
            i = 0
            new_pre_token = []
            while i < len(pre_token):
                if i+1 < len(pre_token) and pre_token[i] == x and pre_token[i+1] == y:
                    new_pre_token.append(x+y)
                    i += 1
                else:
                    new_pre_token.append(pre_token[i])
                i += 1
            new_pre_token = tuple(new_pre_token)
            pre_tokens[new_pre_token] = pre_tokens.pop(pre_token)
            for pair in zip(new_pre_token[:-1], new_pre_token[1:]):
                pair_counts[pair] += n
                heapq.heappush(heap, (-pair_counts[pair], InvertedPair(pair))) # a pair update
                pair_index[pair].add(new_pre_token)
        merges.append(max_pair)
        vocab[len(vocab)] = x + y
    t_merge_final = time.time()
    print(f"time of merges: {t_merge_final - t_merge_init:.2f}")
    return (vocab, merges)

# The main bpe tokenizer function.
def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    pre_tokens = _pretokenize_parallel(input_path, special_tokens, os.cpu_count())
    vocab, merges = _merge_loop(pre_tokens, vocab_size, special_tokens)
    return vocab, merges
    
# Serialize function
def bytes_to_unicode():
    bs = list(range(ord('!'), ord('~')+1)) + list(range(ord('¡'), ord('¬')+1)) + list(range(ord('®'), ord('ÿ')+1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, map(chr, cs)))

def serialize(vocab, merges, vocab_path="vocab.json", merges_path="merges.txt"):
    btu = bytes_to_unicode()
    encoder = {i: ''.join(btu[byte] for byte in token) for i, token in vocab.items()}
    with open(vocab_path, "w") as f:
        json.dump({str(i): s for i, s in encoder.items()}, f, indent=2)
    with open(merges_path, "w") as f:
        for left, right in merges:
            f.write(f"{''.join(btu[b] for b in left)} {''.join(btu[b] for b in right)}\n")

if __name__ == "__main__":
    t0 = time.time()
    vocab, merges = train_bpe("../data/owt_train_1G.txt", vocab_size=32000, special_tokens=["<|endoftext|>"])
    t1 = time.time()
    print("peak RSS parent (GB):", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2)
    print("peak RSS children (GB):", resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024**2)
    print(f"time: {t1 - t0:.2f}s, vocab={len(vocab)}, merges={len(merges)}")
    print(heapq.nlargest(5, vocab.values(), key=len))
    serialize(vocab, merges)
