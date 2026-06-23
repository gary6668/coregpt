## for multi-head or single_head attention

### Goal:
    rewrite the Fake attention module to a real one

### input:
    idx B*T*C

### Internal shapes:
    q: B × T × head_size
    k: B × T × head_size
    v: B × T × head_size
    wei: B × T × T
    out: B × T × head_size

    For the first version:
        head_size = n_embd
        so output shape is B × T × C


### 0619 after dinner

    now status : idx -> embd -> Block -> qkv,s,ao -> ln,sa -> ln;c,4c,GeLU,4c,c(mlp) -> output

    next ：turn single-head to multi-head causal self-attention 

### 0620

    status: complete and understood multi-head causal self attention

    next: try to training loop?? /// remember to set the random seed(torch.manual_seed(42))

### 0623 

    status : add token incoming path, randomize token fetch

    next : make generation module

### 0623_2 

    status: add generation module, set up seed, generation context.

    next: scaling and engineering 