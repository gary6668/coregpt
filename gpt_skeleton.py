import torch
import torch.nn as nn
import torch.nn.functional as F

# define idx, test logits test, loss with the assistance of targets


class FakeSelfAttention(nn.Module):
    def __init__(self, n_embd):
        super().__init__()

    def forward(self, x):
        return torch.zeros_like(x)


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
    def __init__(self, n_embd):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.sa = FakeSelfAttention(n_embd)
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
    def __init__(self, vocab_size, block_size, n_embd, n_layer):
        super().__init__()

        self.block_size = block_size

        # token_embedding_table: vocab_size × n_embd
        # token ids: B × T -> token embeddings: B × T × C
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        # position_embedding_table: block_size × n_embd
        # position ids: T -> position embeddings: T × C
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # blocks: n_layer Transformer blocks
        # currently these are fake Blocks: B × T × C -> B × T × C
        self.blocks = nn.Sequential(*[
            Block(n_embd) for _ in range(n_layer)
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


if __name__ == "__main__":
    vocab_size = 100
    block_size = 8
    n_embd = 16
    n_layer = 2

    model = GPT(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=n_embd,
        n_layer=n_layer,
    )

    # idx: B × T
    idx = torch.tensor([
        [10, 23, 45, 8],
        [7, 4, 9, 11],
    ])
    
    # targets = torch.tensor([
    #     [10, 23, 45, 8],
    #     [7, 4, 9, 11],
    # ])

    # targets: B × T
    # Each position tries to predict the next token.
    targets = torch.tensor([
        [23, 45, 8, 1],
        [4, 9, 11, 2],
    ])

    logits, loss = model(idx, targets)

    print("idx shape:", idx.shape)
    print("logits shape:", logits.shape)
    print("loss:", loss)
