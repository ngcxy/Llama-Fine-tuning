# Phase 2

### Code modification

#### File: generation.py

- Summary: In the way of generating the next token, I change the passing parameter from only the last token to the entire sequence. Consiquently, the parts dealing with `prev_pos`  are all removed.

Original code snip:

```python
prev_pos = 0
...
with torch.no_grad():
		logits = self(tokens[:, prev_pos:cur_pos], prev_pos)
...
prev_pos = cur_pos
```

Modified code snip:

```python
# prev_pos = 0
...
with torch.no_grad():
		logits = self(tokens[:, :cur_pos], 0)
...
# prev_pos = cur_pos
```

#### File: model.py

- Summary: In the Attention class, I remove the attributes of `self.cache_k` and `self.cache_v`. Then I directly assign keys and values with xk and xv. In the Llama class, I comment out the code which pads the mask to (seqlen, cache_len + seqlen), since we only need the mask to be (seqlen, seqlen) when there's no caching.

Original code snip:

```python
self.cache_k = torch.zeros(
  (
    args.max_batch_size,
    args.max_seq_len,
    self.n_local_kv_heads,
    self.head_dim,
  )
).cuda()
self.cache_v = torch.zeros(
  (
    args.max_batch_size,
    args.max_seq_len,
    self.n_local_kv_heads,
    self.head_dim,
  )
).cuda()
	
...

self.cache_k = self.cache_k.to(xq)
self.cache_v = self.cache_v.to(xq)

self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

keys = self.cache_k[:bsz, : start_pos + seqlen]
values = self.cache_v[:bsz, : start_pos + seqlen]

...

mask = torch.hstack([
	torch.zeros((seqlen, start_pos), device=tokens.device),
	mask
]).type_as(h)
```

Modified code snip:

```python
# self.cache_k = torch.zeros(
#   (
#     args.max_batch_size,
#     args.max_seq_len,
#     self.n_local_kv_heads,
#     self.head_dim,
#   )
# ).cuda()
# self.cache_v = torch.zeros(
#   (
#     args.max_batch_size,
#     args.max_seq_len,
#     self.n_local_kv_heads,
#     self.head_dim,
#   )
# ).cuda()
	
...

# self.cache_k = self.cache_k.to(xq)
# self.cache_v = self.cache_v.to(xq)

# self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
# self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

keys = xk
values = xv

...

# mask = torch.hstack([
# 	torch.zeros((seqlen, start_pos), device=tokens.device),
# 	mask
# ]).type_as(h)
```

### Test Prompts and Outputs

![image-20240421152805712](/Users/caixinyi/Library/Application Support/typora-user-images/image-20240421152805712.png)

![image-20240421152650990](/Users/caixinyi/Library/Application Support/typora-user-images/Screenshot 2024-04-21 at 3.27.50 PM.png)
