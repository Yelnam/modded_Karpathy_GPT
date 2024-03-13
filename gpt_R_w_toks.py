if __name__ == "__main__":
    import torch
    import torch.nn as nn
    from torch.nn import functional as F
    from torch.optim.lr_scheduler import CosineAnnealingLR
    import os
    import time
    from datetime import datetime
    from logs.vocab_hugo_lesMis_640 import vocab, merges # ONLY IF YOU HAVE MERGES AND TOKENS FROM A PRV RUN ON THE SAME TEXT, AND ARE NOT CHANGING VOCAB SIZE

        # ----------------

    # hyperparameters
    n_embd = 1024
    n_head = 8
    n_layer = 8
    dropout = 0.2
    fwd_expansion = 4
    batch_size = 64 # n independent sequences to process in parallel
    block_size = 128 # maximum context length for predictions
    vocab_size_utf = 256 # not changeable
    vocab_size = 1024 # n tokens to use, min 256
    max_iters = 5000
    lr_init = 3e-4
    lr_min = 5e-6
    eval_iters = 100 # number of iterations to use for generating loss calculations at intervals

    dir_inputs = 'inputs'
    dir_outputs = 'outputs'
    logs_dir = 'logs'
    dir_models = 'models'

    input_text = input(f'Enter name of plain text data file from {dir_inputs}: ')
    in_file_path = os.path.join(dir_inputs, f'{input_text}.txt')
    with open(in_file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    vocab_prv_yn = input('Use a previous version of vocab/merges? Y/N: ')
    data_prv_yn = input('Use a previous version of encoded data? Y/N: ')
    if data_prv_yn in ('Y', 'y'): 
        data_filename = input('Enter encoded filename: ')

    eval_interval = 500
    gen_interval = 250
    iter_text = 'y'

    patience = 500 # break after n iterations without improvement in val loss 
    patience_yn = 'n'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device = {device}')
    exit_file = "exit_training.txt"
    torch.manual_seed(1337)

    # ------------

    print(f'text head: {text[:250]}')

    token_start_time = time.time()

    tokens = text.encode('utf-8')
    tokens = list(map(int, tokens))

    def get_stats(ids):
        counts = {}
        for pair in zip(ids,ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    stats = get_stats(tokens)

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
    
    ids = list(tokens)
    n_merges = vocab_size - vocab_size_utf

    if vocab_prv_yn not in ('Y', 'y'): 
        merges = {}
        for i in range(n_merges):
            stats = get_stats(ids)
            pair = max(stats, key = stats.get)
            idx = vocab_size_utf + i
            print(f'merging {pair} into a new token {idx}')
            ids = merge(ids, pair, idx)
            merges[pair] = idx

        vocab = {idx: bytes([idx]) for idx in range(256)}

        for (p0, p1), idx in merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]

    token_time_merge = time.time() - token_start_time
    print(f'vocab_size {len(vocab)} in merge time {token_time_merge:.2f} seconds')

    os.makedirs(logs_dir, exist_ok=True)
    log_file_path = os.path.join(logs_dir, f"vocab_{input_text}_{vocab_size}.py")
    with open(log_file_path, 'w', encoding='utf-8') as f:
        f.write((   f'vocab = {vocab}\n'
                    f'merges = {merges}\n'
                    ))
        
    start_time_encode = time.time()

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

    # Train and test splits
    if data_prv_yn not in ('Y', 'y'): 
        data = torch.tensor(encode(text), dtype=torch.long)
        torch.save(data, os.path.join(logs_dir, f'data_{input_text}_{vocab_size}.pt'))
    else:
        data = torch.load(os.path.join(logs_dir, f'{data_filename}.pt'))

    n = int(0.9*len(data)) # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]

    token_time_encode = time.time() - start_time_encode
    print(f'text encoded in {token_time_encode:.2f} seconds')

    # data loading
    def get_batch(split):
        # generate a small batch of data of inputs x and targets y
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    class Head(nn.Module):
        """ one head of self-attention """

        def __init__(self, head_size):
            super().__init__()
            self.key = nn.Linear(n_embd, head_size, bias=False)
            self.query = nn.Linear(n_embd, head_size, bias=False)
            self.value = nn.Linear(n_embd, head_size, bias=False)
            self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            # input of size (batch, time-step, channels)
            # output of size (batch, time-step, head size)
            B,T,C = x.shape
            k = self.key(x)   # (B,T,hs)
            q = self.query(x) # (B,T,hs)
            # compute attention scores ("affinities")
            wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
            wei = F.softmax(wei, dim=-1) # (B, T, T)
            wei = self.dropout(wei)
            # perform the weighted aggregation of the values
            v = self.value(x) # (B,T,hs)
            out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
            return out

    class MultiHeadAttention(nn.Module):
        """ multiple heads of self-attention in parallel """

        def __init__(self, num_heads, head_size):
            super().__init__()
            self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
            self.proj = nn.Linear(head_size * num_heads, n_embd)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            out = torch.cat([h(x) for h in self.heads], dim=-1)
            out = self.dropout(self.proj(out))
            return out

    class FeedFoward(nn.Module):
        """ a simple linear layer followed by a non-linearity """

        def __init__(self, n_embd):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_embd, fwd_expansion * n_embd),
                nn.ReLU(),
                nn.Linear(fwd_expansion * n_embd, n_embd),
                nn.Dropout(dropout),
            )

        def forward(self, x):
            return self.net(x)

    class Block(nn.Module):
        """ Transformer block: communication followed by computation """

        def __init__(self, n_embd, n_head):
            # n_embd: embedding dimension, n_head: the number of heads we'd like
            super().__init__()
            head_size = n_embd // n_head
            self.sa = MultiHeadAttention(n_head, head_size)
            self.ffwd = FeedFoward(n_embd)
            self.ln1 = nn.LayerNorm(n_embd)
            self.ln2 = nn.LayerNorm(n_embd)

        def forward(self, x):
            x = x + self.sa(self.ln1(x))
            x = x + self.ffwd(self.ln2(x))
            return x

    class GPTLanguageModel(nn.Module):

        def __init__(self):
            super().__init__()
            # each token directly reads off the logits for the next token from a lookup table
            self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
            self.position_embedding_table = nn.Embedding(block_size, n_embd)
            self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
            self.ln_f = nn.LayerNorm(n_embd) # final layer norm
            self.lm_head = nn.Linear(n_embd, vocab_size)

            # better init, not covered in the original GPT video, but important, will cover in followup video
            self.apply(self._init_weights)

        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        def forward(self, idx, targets=None):
            B, T = idx.shape

            # idx and targets are both (B,T) tensor of integers
            tok_emb = self.token_embedding_table(idx) # (B,T,C)
            pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
            x = tok_emb + pos_emb # (B,T,C)
            x = self.blocks(x) # (B,T,C)
            x = self.ln_f(x) # (B,T,C)
            logits = self.lm_head(x) # (B,T,vocab_size)

            if targets is None:
                loss = None
            else:
                B, T, C = logits.shape
                logits = logits.view(B*T, C)
                targets = targets.view(B*T)
                loss = F.cross_entropy(logits, targets)
            return logits, loss

        def generate(self, idx, max_new_tokens):
            # idx is (B, T) array of indices in the current context
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -block_size:] # crop idx to the last block_size tokens
                logits, loss = self(idx_cond) # get the predictions
                logits = logits[:, -1, :] # becomes (B, C) # focus only on the last time step
                probs = F.softmax(logits, dim=-1) # (B, C) # apply softmax to get probabilities
                idx_next = torch.multinomial(probs, num_samples=1) # (B, 1) # sample from the distribution
                idx = torch.cat((idx, idx_next), dim=1) # (B, T+1) # append sampled index to the running sequence
            return idx

    model = GPTLanguageModel()
    m = model.to(device)
    # print the number of parameters in the model
    param_count = sum(p.numel() for p in m.parameters())/1e6
    print(f'parameter count = {param_count} m')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_init)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=lr_min)

    model_start_time = time.time()
    iters_complete = 0
    loss_log = ''
    gen_log = ''
    best_val_loss = float('inf')
    intervals_since_improvement = 0

    for iter in range(max_iters):
        
        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            model_time = time.time() - model_start_time
            str_loss = f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f} - elapsed time: {model_time:.2f}s"
            print(str_loss)
            loss_log = "\n".join((loss_log, str_loss))

            # Check for the presence of "exit_training.txt" in root directory to allow early exit
            if os.path.exists(exit_file):
                print("Exit file found. Ending training early.")
                os.remove(exit_file)  # clean up: delete the file after detecting it
                break  # exit the training loop

        if iter % gen_interval == 0 or iter == max_iters - 1 and iter != 1:
            if iter_text == 'y':
                # Generate text from the model
                context = torch.zeros((1, 1), dtype=torch.long, device=device)  # Example context
                generated_text = decode(m.generate(context, max_new_tokens=250)[0].tolist())  # Generate a short text for quick inspection
                str_gen = f"\nGenerated text at step {iter}:\n{generated_text}\n...\n"
                print(str_gen)
                gen_log = "\n".join((gen_log, str_gen))

        if patience_yn == 'y':
            # Check for improvement
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                intervals_since_improvement = 0  # Reset counter
            else:
                intervals_since_improvement += 1
            
            # Early stopping condition
            if intervals_since_improvement >= patience:
                print("Stopping early due to lack of improvement in validation loss.")
                break

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        iter_time = time.time() - model_start_time
        str_loss = f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f} - elapsed time: {model_time:.2f}s"
        if iter % 10 == 0:
            print(f'iteration: {iters_complete}, time elapsed {iter_time:.2f}')

        iters_complete += 1

    dt_string = str(datetime.now().strftime("%Y%m%d_%H%M%S"))

    # time consuming, and we've already created a generation at the final step

    # # generate from the model
    # context = torch.zeros((1, 1), dtype=torch.long, device=device)
    # example_output = decode(m.generate(context, max_new_tokens=500)[0].tolist())
    # print(f'example_output:\n{example_output}')

    os.makedirs(dir_models, exist_ok=True)
    model_name = f"GPTR{int(param_count)}m_{input_text}_{losses['val']:.2f}_{dt_string}.pth"
    model_path = os.path.join(dir_models, model_name) 
    torch.save(m.state_dict(), model_path)

    # takes much longer that generating equivalent output from generator.py, so do that instead

    # os.makedirs(dir_outputs, exist_ok=True)
    # out_file_path = os.path.join(dir_outputs, f'GPTR{int(param_count)}m_{input_text}_{dt_string}.txt')
    # with open(out_file_path, 'w', encoding='utf-8') as f:
    #     f.write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))

    os.makedirs(logs_dir, exist_ok=True)
    log_file_path = os.path.join(logs_dir, f"GPTR{int(param_count)}m_{input_text}_{dt_string}.txt")
    with open(log_file_path, 'w', encoding='utf-8') as f:
        f.write((   f'param_count = {param_count:.2f}m\n'
                    f'token_time_merge = {token_time_merge:.2f}sec\n'
                    f'token_time_encode = {token_time_encode:.2f}sec\n'
                    f'model_time = {model_time:.2f}sec\n'
                    f'total_training_time = {token_time_merge + token_time_encode + model_time:.2f}sec\n'
                    f'patience_yn = {patience_yn}\n'
                    f'patience = {patience}\n'
                    f'iters_complete = {iters_complete}\n'
                    f"train loss = {losses['train']:.2f}\n"
                    f"validation loss = {losses['val']:.2f}\n"
                    f'n_embd = {n_embd}\n'
                    f'n_head = {n_head}\n'
                    f'n_layer = {n_layer}\n'
                    f'dropout = {dropout}\n'
                    f'batch_size = {batch_size}\n'
                    f'block_size = {block_size}\n'
                    f'vocab_size_utf = {vocab_size_utf}\n'
                    f'vocab_size = {vocab_size}\n'
                    f'max_iters = {max_iters}\n'
                    f'lr_init = {lr_init}\n'
                    f'lr_min = {lr_min}\n'
                    f'eval_interval = {eval_interval}\n'
                    f'gen_interval = {gen_interval}\n'
                    f'eval_iters = {eval_iters}\n'
                    f'loss_log = {loss_log}\n'
                    f'gen_log: {gen_log}\n'
                    'See .py file of same name for vocab, merges and data'
                    ))

    os.makedirs(logs_dir, exist_ok=True)
    log_file_path = os.path.join(logs_dir, f"GPTR{int(param_count)}m_{input_text}_{dt_string}.py")
    with open(log_file_path, 'w', encoding='utf-8') as f:
        f.write((   f'n_embd = {n_embd}\n'
                    f'n_head = {n_head}\n'
                    f'n_layer = {n_layer}\n'
                    f'dropout = {dropout}\n'
                    f'block_size = {block_size}\n'
                    f'vocab_size = {vocab_size}\n'
                    ))
     