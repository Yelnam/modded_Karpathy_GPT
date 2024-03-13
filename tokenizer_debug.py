import os

input_dir = 'inputs'
output_dir = 'outputs'
input_text = 'orwell_1984' #'elizabethans' #'wstein_tractatus' #'lit_english' #'orwell_84_mini' #'n_ecce_homo' #'herbert_dune' #'tolstoy_wp' #'shakespeare' #'moliere' 

in_file_path = os.path.join(input_dir, f'{input_text}.txt')
with open(in_file_path, 'r', encoding='utf-8') as f:
    text = f.read()

# temp text for testing
    
text = 'abcdefABCDEF xyzXYZ the cat dog, the cat, the dog hello hello world!'

tokens = text.encode('utf-8')
print(f'tokens1 = {tokens}')
tokens = list(map(int, tokens))
print(f'tokens2 = {tokens}')

print(f'\nlen text: {len(text)}, len tokens: {len(tokens)}')

def get_stats(ids):
    counts = {}
    for pair in zip(ids,ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

stats = get_stats(tokens)
print(f'\nstats: {stats}')
print(f'top pairs = {sorted(((v,k) for k,v in stats.items()), reverse = True)[:5]}')

top_pair = max(stats, key = stats.get)
print(f' top_pair = {top_pair}')

def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) -1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

print(f'\ntokens3 = {tokens}\n')

ids = list(tokens)
vocab_size_orig = len(set(tokens))
vocab_size_utf = 256 # I'm simply copying K here so as not to break anything
# (I know where the number comes from in general, not sure how exactly here)
vocab_size_fin = 300
n_merges = vocab_size_fin - vocab_size_utf

merges = {}
for i in range(n_merges):
    stats = get_stats(ids)
    pair = max(stats, key = stats.get)
    idx = vocab_size_utf + i
    print(f'merging {pair} into a new token {idx}')
    ids = merge(ids, pair, idx)
    merges[pair] = idx

print(f'\ntokens4 = {tokens}')
print(f'\nids = {ids}')

print(f'\nlen tokens: {len(tokens)}')
print(f'len ids: {len(ids)}')
print(f'compression ratio: {len(tokens) / len(ids) :.2f}X')

vocab = {idx: bytes([idx]) for idx in range(256)}

print(f'vocab: {vocab}')

for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]

print(f'vocab2: {vocab}')
    
def decode(ids):
    tokens = b"".join(vocab[idx] for idx in ids)
    text = tokens.decode('utf-8', errors = 'replace') # to do with how invalid byte sequences (e.g. 128) are handled
    return text

def encode(text):
    tokens = list(text.encode('utf-8'))
    while len (tokens) >= 2:
        stats = get_stats(tokens)
        pair = min(stats, key = lambda p: merges.get(p, float('inf')))
        if pair not in merges:
            break
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens

print(f'\nhello world! encoded: {encode("hello world!!")}')
print(f'\nthe cat encoded: {encode("the cat")}')
print(f'\nabcdefABCDEF xyzXYZ the cat dog, the cat, the dog encoded: {encode("abcdefABCDEF xyzXYZ the cat dog, the cat, the dog")}')

text2 = decode(encode(text))
print(f'text2 == text ? {text2 == text}')