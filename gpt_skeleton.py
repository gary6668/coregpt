import torch
import torch.nn as nn
import torch.nn.functional as F

# define idx, test logits test, loss with the assistance of targets

class SingleHeadCausalSelfAttention(nn.Module):
    """
    Single-head causal self-attention.

    Input:  B x T x C
    Output: B x T x head_size
    """

    def __init__(self, n_embd, head_size, block_size):
        super().__init__()

        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)      # B × T × head_size
        q = self.query(x)    # B × T × head_size
        v = self.value(x)    # B × T × head_size

        wei = q @ k.transpose(-2, -1) * self.head_size ** -0.5  # B × T × T (BTC @ BCT)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)  # B × T × T

        out = wei @ v  # B × T × head_size
        return out
# class FakeSelfAttention(nn.Module):
#     def __init__(self, n_embd):
#         super().__init__()

#     def forward(self, x):
#         return torch.zeros_like(x)
class MultiHeadCausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()

        assert n_embd % n_head == 0

        head_size = n_embd // n_head

        self.heads = nn.ModuleList([
            SingleHeadCausalSelfAttention(n_embd, head_size, block_size)
            for _ in range(n_head)
        ])

        self.proj = nn.Linear(n_embd, n_embd)
    
    def forward(self, x):
        
        # each head(x): B × T × head_size
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        
        # out after concat: B × T × C
        out = self.proj(out)

        #out after projection: B × T × C   
        return out

class MLP(nn.Module):
    """
    Feed-forward network inside a Transformer Block.

    Input:  B x T x C
    Output: B x T x C
    """

    def __init__(self, n_embd):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)
# class FakeMLP(nn.Module):
#     def __init__(self, n_embd):
#         super().__init__()

#     def forward(self, x):
#         return torch.zeros_like(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.sa = MultiHeadCausalSelfAttention(n_embd, n_head, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
# class Block(nn.Module):
#     """
#     Temporary placeholder Block.

#     For now, it does not implement attention or MLP.
#     It simply returns x unchanged, so we can test the outer GPT pipeline first.
#     """

#     def __init__(self, n_embd):
#         super().__init__()

#     def forward(self, x):
#         return x


class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer):
        super().__init__()

        self.block_size = block_size

        # token_embedding_table: vocab_size × n_embd
        # token ids: B × T -> token embeddings: B × T × C
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        # position_embedding_table: block_size × n_embd
        # position ids: T -> position embeddings: T × C
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # blocks: n_layer Transformer blocks
        # current version : multi-head causal self-attention + MLP
        # B × T × C -> B × T × C
        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head, block_size) for _ in range(n_layer)
        ])

        # final LayerNorm: B × T × C -> B × T × C
        self.ln_f = nn.LayerNorm(n_embd)

        # lm_head: C -> vocab_size
        # B × T × C -> B × T × vocab_size
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        # idx: B × T
        B, T = idx.shape

        # tok_emb: B × T × C
        tok_emb = self.token_embedding_table(idx)

        # pos: T
        # Example: if T = 4, pos = [0, 1, 2, 3]
        pos = torch.arange(T, device=idx.device)

        # pos_emb: T × C
        pos_emb = self.position_embedding_table(pos)

        # x: B × T × C
        # pos_emb broadcasts from T × C to B × T × C
        x = tok_emb + pos_emb

        # x: B × T × C
        x = self.blocks(x)

        # x: B × T × C
        x = self.ln_f(x)

        # logits: B × T × vocab_size
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # logits: B × T × V
            B, T, V = logits.shape

            # logits_flat: B*T × V
            logits_flat = logits.view(B * T, V)

            # targets_flat: B*T
            targets_flat = targets.view(B * T)

            # loss: scalar
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss
    
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):

            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)  #logits: (B, T, V)
            logits = logits[:, -1, :]  #(B, T, V) -> (B, V)

            logits = logits / temperature

            probs = F.softmax(logits, dim=-1)  #probs: (B, V)

            if top_k is not None:
                v, ix = torch.topk(probs, top_k)
                probs_ = torch.zeros_like(probs)
                probs_.scatter_(-1, ix, v)
                probs = probs_ / probs_.sum(dim=-1, keepdim=True)

            idx_next = torch.multinomial(probs, num_samples=1)  #idx_next: (B, 1)

            idx = torch.cat([idx, idx_next], dim=1) #idx: (B, T)

        return idx



if __name__ == "__main__":
    torch.manual_seed(42)

    text = "hello world. this is a tiny gpt test. hello again."

    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    # print("chars:", chars)
    # print("vocab_size:", vocab_size)

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    encode = lambda s: [stoi[c] for c in s]
    decode = lambda ids: "".join([itos[i] for i in ids])

    data = torch.tensor(encode(text), dtype=torch.long)
    
    # vocab_size = 100
    block_size = 8
    batch_size = 4



    def get_batch(data, batch_size, block_size):
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        return x, y


    n_embd = 16
    n_head = 4
    n_layer = 2

    model = GPT(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
    )

    # idx, targets = get_batch(data, batch_size, block_size)

    # # idx: B × T
    # idx = torch.tensor([
    #     [10, 23, 45, 8],
    #     [7, 4, 9, 11],
    # ])

    # # targets = torch.tensor([
    # #     [10, 23, 45, 8],
    # #     [7, 4, 9, 11],
    # # ])

    # # targets: B × T
    # # Each position tries to predict the next token.
    # targets = torch.tensor([
    #     [23, 45, 8, 1],
    #     [4, 9, 11, 2],
    # ])

    # print("idx shape:", idx.shape)

    # logits, loss = model(idx, targets)
    # print("logits shape:", logits.shape)
    # print("initial loss:", loss.item())

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for step in range(101):
        idx, targets = get_batch(data, batch_size, block_size)
        logits, loss = model(idx, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:

            print(f"step {step}: loss = {loss.item():.4f}")

    context = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(
        context,
        max_new_tokens=100,
        temperature=0.8,
        top_k=5
    )

    print(decode(out[0].tolist()))





    # logits, loss = model(idx, targets)

    # print("idx shape:", idx.shape)
    # print("logits shape:", logits.shape)
    # print("loss:", loss)
