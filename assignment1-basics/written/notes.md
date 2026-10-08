Problem (unicode1):  Understanding Unicode (1 point).

(a) What Unicode character does chr(0) return? 
    '\x00' the empty character.
(b) How does this character’s string representation (__repr__()) differ from its printed representation? 
    The repr prints the representation character of the number 0, while the print represents what unicode string the codepoint 0 represents.
(c) What happens when this character occurs in text? It may be helpful to play around with the  following in your Python interpreter and see if it matches your expectations: 
    chr(0) prints the unicode point representation '\x00' while the print function represents the empty space, i.e. what string the '\x00' unicode point maps to.

---------------------------------------------------------------------------------

Problem (unicode2):  Unicode Encodings.

(a) What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than  UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various  input strings. 
    Because the length of the vocabulary is small (only 256 values) and with 16/32 that size increases significally, resulting in a very sparse corpus. Even with working with other languages, the improvements are null, because the byte-pair encoding already merges some bytes together, so the merges absort the cost of habing multiple bytes for a specific character from another language.

(b) Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string  into a Unicode string. Why is this function incorrect? Provide an example of an input byte  string that yields incorrect results. 
    It fails because it tries to decode by each byte, but because the utf-8 encodes certain characters with multiple bytes, certain caracters like the ones on CJKgive a UnicodeDecodeError. for example "친구".

Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string  into a Unicode string. Why is this function incorrect? Provide an example of an input byte  string that yields incorrect results. 

(c) Give a two-byte sequence that does not decode to any Unicode character(s).
    [0xC0, 0x80] because the 0xC0 in binary is 11000000 and the prefix 110 confirms that is a 2-byte sequence, meaning that 0x80 is a continuation, representated by the prefix 10 (10000000). Exctranting the prefixes and adding them, it results in 00000000, that equals 0 in decimal value, and U+0000 unicode point, but U+0000 is already 0x00, so it returns an error.

----------------------------------------------------------------------------------

Example (bpe_example):  BPE training example:
I will use the first 255 bytes as the vocab, and after that, the special characters like <|endoftext|>

----------------------------------------------------------------------------------

Problem (train_bpe):  BPE Tokenizer Training (15 points) 

My first implementation used the pair_counts only, and obtained the max by iterating over the entire dictionary, and that was O(p) where p was the number of pairs. That was slow so I changed it to use a max heap, resulting in an amortized log(p) for that particular operation. Also changed the way the parent merged the counters of each doc for each chunk, instead of converting each string as its bytecode representation for each chunk counter, just construct the counter as a counter of strings and, after each process returns its chunk counter, the parent merges them and converts each string to its byte encoding, resulting in a more optimized tokenizer. Before using the update and patch approach, I was removing every neighbor element of the winning pair, and adding them again with the new_pre_token already merged, so technically was correct and worked, but after talking to a model about it (specifically glm-5.3-flash) it said that this approach could have some errors, so I changed it to use the update and patch approach.

----------------------------------------------------------------------------------

Problem (train_bpe_tinystories):  BPE Training on TinyStories (2 points):

(a) Train a byte-level BPE tokenizer on the TinyStories dataset, using a maximum vocabulary size of 10,000. Make sure to add the TinyStories <|endoftext|> special token to the vocabulary. Serialize the resulting vocabulary and merges to disk for further inspection. How much time and memory did training take? What is the longest token in the vocabulary? Does it make sense?Resource requirements: ≤30 minutes (no GPUs), ≤30 GB RAM9
    It took 117.10s but sometimes it can go to only 70s, so between 70-120s. In memory, it only took around 0.4569GB maximum, so it is very far from the threshold. The longest merged word was " accomplishment", and yeah, it makes sense because the GPT2 regex counts spaces as a part of a same word token, and yes, the word appears around 1500 times in the dataset.

(b) Profile your code. What part of the tokenizer training process takes the most time?
    The pre-tokenize step, even with the usage of multiprocessor and using multiple parallel cores to do the computation, it still takes a significant number of time of the totall tokenizer flow, mostly the 80% of the time of the entire computation of the function.

----------------------------------------------------------------------------------

Problem (train_bpe_expts_owt):  BPE Training on OpenWebText (2 points):

(a) Train a byte-level BPE tokenizer on the OpenWebText dataset, using a maximum vocabulary size of 32,000. Serialize the resulting vocabulary and merges to disk for further inspection. What is the longest token in the vocabulary? Does it make sense?Resource requirements: ≤12 hours (no GPUs), ≤100 GB RAM
    I could't train my tokenizer using the OpenWebText dataset because my laptop would run out of memory or it would become very slow because it would start to use swap memory to store and acess the merges. Contrary to the TinyStories dataset, the heavy part was the merge rather than the pre-tokenization. The results were that the pre-tokenization took 11.42s and the merge took 550.52s. The longest strings were " responsibility", " disappointment" and " accomplishment", and they make sense, curiosly, " accomplishment" also appears as the longest merge string in the tinystories dataset.

(b) Compare and contrast the tokenizer that you get training on TinyStories versus  OpenWebText.
    As I said in my previous response, in tinystories the pre-tokenization took more compute time and resources, contraty to the openwebtext that was the inverse. Probably because of the number of iterations of the merge-loop.

----------------------------------------------------------------------------------

Problem (tokenizer_experiments):  Experiments with tokenizers (4 points)

(a) Sample 10 documents from TinyStories and OpenWebText. Using your previously-trained TinyStories and OpenWebText tokenizers (10K and 32K vocabulary size, respectively), encode these sampled documents into integer IDs. What is each tokenizer’s compression ratio (bytes/token)?
    dataset      vocab  bytes   tokens  bytes/token
    TinyStories  10k    7412    1786    4.150055991041433
    OWT          32k    50460   11207   4.502543053448737
Important to note that I couldn't train on the entire training file because of 
ram constraints on my machine. So, I used only 1.1G of the original training dataset to train the owt tokenizer.

(b) What happens if you tokenize your OpenWebText sample with the TinyStories tokenizer?  Compare the compression ratio and/or qualitatively describe what happens. 
    Compression ration: 3.404628567573038
    The compression ratio decreases significantly, probably because of the smaller merges and vocab available to to do the merges in the pre-tokenize encoding step.

Estimate the throughput of your tokenizer (e.g., in bytes/second). How long would it take to  tokenize the Pile dataset (825GB of text)? 
    Using the tinystories tokenizer, for 7412 bytes it takes 0.017s, so, my throughput would be 7412/0.017, and that would be approximately 436000 bytes per second (bytes/s). Using the exact computation it would be 429305.0384856463 tokens per second.
    To tokenize 825GB of text, it would be 825000000000 bytes, so the tokenizer would take 1921710.4996255096s, approximately 533 hours or 22-23 days of encoding.

(d) Using your TinyStories and OpenWebText tokenizers, encode the respective training and  development datasets into a sequence of integer token IDs. We’ll use this later to train our language model. We recommend serializing the token IDs as a NumPy array of datatype  uint16. Why is uint16 an appropriate choice? 
    Its an approiate approach because with 16 bits, we can represent values from 0 to 65000, and given that our tokenizer from owt has nearly 31k different match ids, its a good number to encode those ids with that type.
    Node: I implemented some parallelism to the computation of the ids in order to make the process faster and less time expensive, but even with that improvement, encoding the full files of both, the owt and tinystories took some 3 hours to complete in my machine.

----------------------------------------------------------------------------------
