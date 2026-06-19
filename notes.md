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
