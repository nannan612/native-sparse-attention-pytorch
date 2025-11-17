from __future__ import annotations

from copy import deepcopy
from math import ceil
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn, arange, stack, cat, tensor, Tensor
from torch.nn import Module, ModuleList

from local_attention import LocalAttention

from rotary_embedding_torch import RotaryEmbedding

# einstein notation

import einx
from einops import einsum, repeat, rearrange, reduce, pack, unpack
from einops.layers.torch import Rearrange

# b - batch
# h - heads
# qh - grouped query heads
# n - sequence (token level or compressed)
# w - windows, for fine or compressed
# i, j - query / key sequence
# d - feature dimension
# s - strategies

# flex attention
# https://pytorch.org/blog/flexattention/

flex_attention = None

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    if torch.cuda.is_available():
        flex_attention = torch.compile(flex_attention)
except ImportError:
    pass

# flex attn sliding attention mask


def create_sliding_mask(seq_len, window_size, causal = True):

    def sliding_mask(_, __, q_idx, kv_idx):

        distance = q_idx - kv_idx
        backward_sliding_mask = distance <= window_size

        forward_distance = 0 if causal else -window_size
        forward_sliding_mask = distance >= forward_distance

        return backward_sliding_mask & forward_sliding_mask

    block_mask = create_block_mask(sliding_mask, B = None, H = None, Q_LEN = seq_len, KV_LEN = seq_len, _compile = True)
    return block_mask

def create_compress_mask(seq_len, kv_seq_len, compress_block_sliding_stride, mem_kv_len = 0, causal = True):

    if not causal:
        return None

    # cannot be used as using attention logits for importance score
    # but just to show the immense potential of flex attention

    def compress_mask(_, __, q_idx, kv_idx):
        is_mem_kv = kv_idx < mem_kv_len

        kv_without_mem = kv_idx - mem_kv_len
        compress_kv_idx = (kv_without_mem * compress_block_sliding_stride) + (compress_block_sliding_stride - 1)

        causal_mask = q_idx > compress_kv_idx
        return causal_mask | is_mem_kv

    block_mask = create_block_mask(compress_mask, B = None, H = None, Q_LEN = seq_len, KV_LEN = kv_seq_len + mem_kv_len, _compile = True)
    return block_mask

def create_fine_mask(seq_len, fine_block_size, causal = True):

    def inner(selected_block_indices: Tensor, num_grouped_queries = 1):
        device = selected_block_indices.device
        batch, kv_heads = selected_block_indices.shape[:2]

        one_hot_selected_block_indices = torch.zeros((*selected_block_indices.shape[:-1], seq_len // fine_block_size), device = device, dtype = torch.bool)
        one_hot_selected_block_indices.scatter_(-1, selected_block_indices, True)

        def fine_mask(b_idx, h_idx, q_idx, kv_idx):

            compressed_q_idx = q_idx // fine_block_size
            compressed_kv_idx = kv_idx // fine_block_size
            kv_head_idx = h_idx // num_grouped_queries

            is_selected = one_hot_selected_block_indices[b_idx, kv_head_idx, q_idx, compressed_kv_idx]

            if not causal:
                return is_selected

            causal_mask = q_idx >= kv_idx
            block_diagonal = compressed_q_idx == compressed_kv_idx

            return (causal_mask & (block_diagonal | is_selected))

        block_mask = create_block_mask(fine_mask, B = batch, H = kv_heads * num_grouped_queries, Q_LEN = seq_len, KV_LEN = seq_len, _compile = True)
        return block_mask

    return inner

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def round_down_mult(n, mult):
    return n // mult * mult

def round_up_mult(n, mult):
    return ceil(n / mult) * mult

def divisible_by(num, den):
    return (num % den) == 0

def is_empty(t):
    return t.numel() == 0

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

def pack_one_with_inverse(t, pattern):
    packed, ps = pack([t], pattern)
    def inverse(out):
        return unpack(out, ps, pattern)[0]

    return packed, inverse

# tensor helpers

def pad_at_dim(t, pad, dim = -1, value = 0.):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

def straight_through(t, target):
    return t + (target - t).detach()

def batched_gather(t, indices):
    """Gather helper that supports batched [b, h, n, d] tensors."""
    if indices.numel() == 0:
        return torch.zeros((*indices.shape, t.shape[-1]), device = t.device, dtype = t.dtype)

    b, h, num_tokens, dim = t.shape
    query_len, topk = indices.shape[-2:]

    flat_t = rearrange(t, 'b h n d -> (b h) n d')
    flat_indices = rearrange(indices, 'b h q k -> (b h) (q k)')
    flat_indices = flat_indices.unsqueeze(-1).expand(-1, -1, dim)

    gathered = torch.gather(flat_t, dim = 1, index = flat_indices)
    gathered = rearrange(gathered, '(b h) (q k) d -> b h q k d', b = b, h = h, q = query_len, k = topk)

    return gathered

class MultiStageResidualCompressor(Module):
    def __init__(
        self,
        heads,
        dim_head,
        block_size,
        levels = 2,
        expand_factor = 2.
    ):
        super().__init__()
        self.levels = levels
        self.block_size = block_size
        self.dim_head = dim_head

        block_dim = block_size * dim_head
        hidden_dim = int(block_dim * expand_factor)

        self.compress_mlps = ModuleList([])
        self.decompress_mlps = ModuleList([])

        for _ in range(levels):
            self.compress_mlps.append(nn.Sequential(
                nn.Linear(block_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim_head)
            ))

            self.decompress_mlps.append(nn.Sequential(
                nn.Linear(dim_head, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, block_dim)
            ))

    def forward(self, kv_windows):
        """
        kv_windows: Float['b h w n d']
        returns list of compressed representations per level with shape Float['b h w d']
        """
        b, h, w, n, d = kv_windows.shape
        residual = kv_windows
        compressed = []

        for level in range(self.levels):
            flat = rearrange(residual, 'b h w n d -> (b h w) (n d)')
            comp = self.compress_mlps[level](flat)
            comp = rearrange(comp, '(b h w) d -> b h w d', b = b, h = h, w = w)
            compressed.append(comp)

            approx = self.decompress_level(level, comp)
            residual = residual - approx

        return compressed

    def decompress_level(self, level, tokens):
        *rest, dim = tokens.shape
        assert dim == self.dim_head, 'decompress expects last dim to be dim_head'
        flat = tokens.reshape(-1, dim)
        decoded = self.decompress_mlps[level](flat)
        decoded = decoded.reshape(*rest, self.block_size * self.dim_head)
        decoded = rearrange(decoded, '... (n d) -> ... n d', n = self.block_size, d = self.dim_head)
        return decoded

# attend function

def attend(
    q, k, v,
    mask = None,
    return_sim = False,
    scale = None
):
    scale = default(scale, q.shape[-1] ** -0.5)

    q_heads, k_heads = q.shape[1], k.shape[1]
    num_grouped_queries = q_heads // k_heads

    q = rearrange(q, 'b (h qh) ... -> b h qh ...', qh = num_grouped_queries)

    sim = einsum(q, k, 'b h qh i d, b h j d -> b h qh i j') * scale

    mask_value = max_neg_value(sim)

    if exists(mask):
        sim = sim.masked_fill(~mask, mask_value // 10)

    attn = sim.softmax(dim = -1)

    attn_out = einsum(attn, v, 'b h qh i j, b h j d -> b h qh i d')

    attn_out = rearrange(attn_out, 'b h qh ... -> b (h qh) ...')

    if not return_sim:
        return attn_out

    sim = rearrange(sim, 'b h qh ... -> b (h qh) ...')

    return attn_out, sim

# classes

class SparseAttention(Module):
    def __init__(
        self,
        dim,
        dim_head,
        heads,
        sliding_window_size,
        compress_block_size,
        compress_block_sliding_stride,
        selection_block_size,
        num_selected_blocks,
        kv_heads = None,
        num_compressed_mem_kv = 1,
        causal = False,
        norm = True,
        use_diff_topk = False,
        use_triton_kernel = False,
        query_heads_share_selected_kv = True,
        compress_mlp: Module | None = None,
        compress_mlp_expand_factor = 1.,
        strategy_combine_mlp: Module | None = None,
        residual_compression_levels = 2,
        compressed_topk = 4,
        kl_mixing_steps = 0,
        kl_weight = 1.
    ):
        super().__init__()

        kv_heads = default(kv_heads, heads)
        assert kv_heads <= heads and divisible_by(heads, kv_heads)

        assert residual_compression_levels > 0, 'residual compression must have at least one level'
        assert compressed_topk > 0, 'compressed top-k must be greater than 0'

        self.heads = heads
        self.dim_head = dim_head
        self.kv_heads = kv_heads
        self.num_grouped_queries = heads // kv_heads
        self.residual_levels = residual_compression_levels
        self.compressed_topk = compressed_topk
        self.compressed_topk_per_level = max(1, compressed_topk // residual_compression_levels)

        self.scale = dim_head ** -0.5

        dim_inner = dim_head * heads
        dim_kv_inner = dim_head * kv_heads

        self.norm = nn.RMSNorm(dim) if norm else nn.Identity()

        self.causal = causal

        self.rotary_emb = RotaryEmbedding(dim_head)

        qkv_split = (dim_inner, dim_kv_inner, dim_kv_inner)
        self.to_qkv = nn.Linear(dim, sum(qkv_split), bias = False)
        self.qkv_split = qkv_split

        self.sliding_window = LocalAttention(
            dim = dim_head,
            window_size = sliding_window_size,
            causal = causal,
            exact_windowsize = True,
            autopad = True,
            use_rotary_pos_emb = False
        )
        self.sliding_window_size = sliding_window_size

        self.compress_block_size = compress_block_size
        self.compress_block_sliding_stride = compress_block_sliding_stride
        assert self.compress_block_size >= self.compress_block_sliding_stride, 'compress_block_size must be >= compress_block_sliding_stride'
        assert self.compress_block_sliding_stride > 0, 'compress_block_sliding_stride must be greater than 0'

        self.split_compress_window = nn.Sequential(
            Rearrange('b h n d -> (b h) d 1 n'),
            nn.ZeroPad2d(((compress_block_size - compress_block_sliding_stride), 0, 0, 0)),
            nn.Unfold(kernel_size=(1, self.compress_block_size), stride=(1, self.compress_block_sliding_stride)),
            Rearrange('(b h) (d n) w -> b h w n d', d = dim_head, h = kv_heads, n = self.compress_block_size)
        )

        self.k_residual_compressor = MultiStageResidualCompressor(
            heads = kv_heads,
            dim_head = dim_head,
            block_size = self.compress_block_size,
            levels = residual_compression_levels,
            expand_factor = compress_mlp_expand_factor
        )

        self.v_residual_compressor = MultiStageResidualCompressor(
            heads = kv_heads,
            dim_head = dim_head,
            block_size = self.compress_block_size,
            levels = residual_compression_levels,
            expand_factor = compress_mlp_expand_factor
        )

        if not exists(strategy_combine_mlp):
            strategy_combine_mlp = nn.Linear(dim, 2 * heads)
            nn.init.zeros_(strategy_combine_mlp.weight)
            strategy_combine_mlp.bias.data.copy_(tensor([-2., 2.] * heads))

        self.to_strategy_combine = nn.Sequential(
            strategy_combine_mlp,
            nn.Sigmoid(),
            Rearrange('b n (h s) -> b h n s', h = heads)
        )

        self.split_heads = Rearrange('b n (h d) -> b h n d', d = dim_head)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')
        self.combine_heads = nn.Linear(dim_inner, dim, bias = False)

        self.kl_mixing_steps = kl_mixing_steps
        self.kl_weight = kl_weight
        self.register_buffer('kl_step', torch.tensor(0.), persistent = False)
        self._extra_loss = None

    def _repeat_kv(self, t):
        return repeat(t, 'b h ... -> b (h gh) ...', gh = self.num_grouped_queries)

    def _sliding_attention(self, q, k, v, sliding_window_flex_mask = None):
        if exists(sliding_window_flex_mask) and exists(flex_attention):
            repeated_k = self._repeat_kv(k)
            repeated_v = self._repeat_kv(v)
            return flex_attention(q, repeated_k, repeated_v, block_mask = sliding_window_flex_mask, enable_gqa = True)

        repeated_k = self._repeat_kv(k)
        repeated_v = self._repeat_kv(v)
        return self.sliding_window(q, repeated_k, repeated_v)

    def _compress_windows(self, k, v):
        k_windows = self.split_compress_window(k)
        v_windows = self.split_compress_window(v)
        k_levels = self.k_residual_compressor(k_windows)
        v_levels = self.v_residual_compressor(v_windows)
        return k_levels, v_levels

    def _compress_block(self, block_k, block_v):
        block_k = rearrange(block_k, 'b h n d -> b h 1 n d')
        block_v = rearrange(block_v, 'b h n d -> b h 1 n d')
        k_levels = self.k_residual_compressor(block_k)
        v_levels = self.v_residual_compressor(block_v)
        return k_levels, v_levels

    def _query_conditioned_attention(self, q, k_levels, v_levels):
        if len(k_levels) == 0:
            return torch.zeros_like(q)

        q_grouped = rearrange(q, 'b (h gh) n d -> b h gh n d', h = self.kv_heads, gh = self.num_grouped_queries)
        q_importance = q_grouped.mean(dim = 2)

        level_outputs = []

        for level, (ck, cv) in enumerate(zip(k_levels, v_levels)):
            if ck.shape[-2] == 0:
                continue

            sim = einsum(q_importance, ck, 'b h i d, b h j d -> b h i j') * self.scale
            topk = min(self.compressed_topk_per_level, sim.shape[-1])
            if topk == 0:
                continue

            _, indices = sim.topk(topk, dim = -1)
            sel_ck = batched_gather(ck, indices)
            sel_cv = batched_gather(cv, indices)

            decoded_k = self.k_residual_compressor.decompress_level(level, sel_ck)
            decoded_v = self.v_residual_compressor.decompress_level(level, sel_cv)

            decoded_k = rearrange(decoded_k, 'b h i topk block d -> b h i (topk block) d')
            decoded_v = rearrange(decoded_v, 'b h i topk block d -> b h i (topk block) d')

            decoded_k = repeat(decoded_k, 'b h ... -> b (h gh) ...', gh = self.num_grouped_queries)
            decoded_v = repeat(decoded_v, 'b h ... -> b (h gh) ...', gh = self.num_grouped_queries)

            level_sim = einsum(q, decoded_k, 'b h i d, b h i j d -> b h i j') * self.scale
            level_attn = level_sim.softmax(dim = -1)
            level_out = einsum(level_attn, decoded_v, 'b h i j, b h i j d -> b h i d')
            level_outputs.append(level_out)

        if len(level_outputs) == 0:
            return torch.zeros_like(q)

        return sum(level_outputs)

    def _combine_with_strategy(self, inp, compressed_out, sliding_out):
        strategies = self.to_strategy_combine(inp)
        stacked = stack([compressed_out, sliding_out])
        return einsum(strategies, stacked, 'b h n s, s b h n d -> b h n d')

    def _dense_reference(self, q, k, v):
        dense_k = self._repeat_kv(k)
        dense_v = self._repeat_kv(v)
        return F.scaled_dot_product_attention(q, dense_k, dense_v, is_causal = self.causal)

    def _kl_loss(self, target, current):
        log_probs = F.log_softmax(current, dim = -1)
        target_probs = F.softmax(target, dim = -1)
        return self.kl_weight * F.kl_div(log_probs, target_probs, reduction = 'batchmean')

    def _mix_with_dense(self, q, k, v, combined_out):
        if not self.training or self.kl_mixing_steps <= 0:
            self._extra_loss = None
            return combined_out

        step = int(self.kl_step.item())
        if step >= self.kl_mixing_steps:
            self._extra_loss = None
            return combined_out

        dense_out = self._dense_reference(q, k, v)
        mix_alpha = 1. - (step / max(1, self.kl_mixing_steps))
        self.kl_step += 1
        self._extra_loss = self._kl_loss(dense_out.detach(), combined_out)
        return (mix_alpha * dense_out) + ((1. - mix_alpha) * combined_out)

    def forward_train(
        self,
        inp,
        return_cache = False,
        sliding_window_flex_mask = None,
        **_
    ):
        inp = self.norm(inp)
        q, k, v = self.to_qkv(inp).split(self.qkv_split, dim = -1)
        q, k, v = map(self.split_heads, (q, k, v))
        q, k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)

        sliding_out = self._sliding_attention(q, k, v, sliding_window_flex_mask = sliding_window_flex_mask)
        k_levels, v_levels = self._compress_windows(k, v)
        compressed_out = self._query_conditioned_attention(q, k_levels, v_levels)
        combined = self._combine_with_strategy(inp, compressed_out, sliding_out)
        attn_out = self._mix_with_dense(q, k, v, combined)

        out = self.merge_heads(attn_out)
        out = self.combine_heads(out)

        if not return_cache:
            return out

        return out, None

    def _init_cache_levels(self, batch, device, dtype):
        levels = []
        for _ in range(self.residual_levels):
            levels.append(torch.zeros((batch, self.kv_heads, 0, self.dim_head), device = device, dtype = dtype))
        return levels

    def forward_inference(
        self,
        inp,
        cache = None,
        return_cache = True,
        **_
    ):
        assert self.causal, 'inference only relevant for autoregressive use-cases'

        cache = default(cache, {})
        batch, device, dtype = inp.shape[0], inp.device, inp.dtype

        sliding_k = cache.get('sliding_k')
        sliding_v = cache.get('sliding_v')
        run_k = cache.get('run_k', torch.zeros((batch, self.kv_heads, 0, self.dim_head), device = device, dtype = dtype))
        run_v = cache.get('run_v', torch.zeros((batch, self.kv_heads, 0, self.dim_head), device = device, dtype = dtype))
        compressed_k_levels = cache.get('compressed_k')
        compressed_v_levels = cache.get('compressed_v')
        offset = cache.get('offset', 0)

        if compressed_k_levels is None:
            compressed_k_levels = self._init_cache_levels(batch, device, dtype)
        if compressed_v_levels is None:
            compressed_v_levels = self._init_cache_levels(batch, device, dtype)
        if sliding_k is None:
            sliding_k = torch.zeros((batch, self.kv_heads, 0, self.dim_head), device = device, dtype = dtype)
        if sliding_v is None:
            sliding_v = torch.zeros((batch, self.kv_heads, 0, self.dim_head), device = device, dtype = dtype)

        inp_norm = self.norm(inp)
        q, k, v = self.to_qkv(inp_norm).split(self.qkv_split, dim = -1)
        q, k, v = map(self.split_heads, (q, k, v))

        run_k = cat((run_k, k), dim = -2)
        run_v = cat((run_v, v), dim = -2)

        while run_k.shape[-2] >= self.compress_block_size:
            block_k = run_k[..., :self.compress_block_size, :]
            block_v = run_v[..., :self.compress_block_size, :]
            block_k_levels, block_v_levels = self._compress_block(block_k, block_v)
            for level in range(self.residual_levels):
                compressed_k_levels[level] = cat((compressed_k_levels[level], block_k_levels[level]), dim = -2)
                compressed_v_levels[level] = cat((compressed_v_levels[level], block_v_levels[level]), dim = -2)
            run_k = run_k[..., self.compress_block_sliding_stride:, :]
            run_v = run_v[..., self.compress_block_sliding_stride:, :]

        q = self.rotary_emb.rotate_queries_or_keys(q, offset = offset)
        k_rot = self.rotary_emb.rotate_queries_or_keys(k, offset = offset)
        offset = offset + q.shape[-2]

        sliding_k = cat((sliding_k, k_rot), dim = -2)
        sliding_v = cat((sliding_v, v), dim = -2)

        if self.sliding_window_size > 0:
            sliding_k = sliding_k[..., -self.sliding_window_size:, :]
            sliding_v = sliding_v[..., -self.sliding_window_size:, :]

        sliding_out = self._sliding_attention(q, sliding_k, sliding_v)
        compressed_out = self._query_conditioned_attention(q, compressed_k_levels, compressed_v_levels)

        combined = self._combine_with_strategy(inp_norm, compressed_out, sliding_out)
        attn_out = combined

        out = self.merge_heads(attn_out)
        out = self.combine_heads(out)

        if not return_cache:
            return out

        next_cache = dict(
            sliding_k = sliding_k.detach(),
            sliding_v = sliding_v.detach(),
            compressed_k = [level.detach() for level in compressed_k_levels],
            compressed_v = [level.detach() for level in compressed_v_levels],
            run_k = run_k.detach(),
            run_v = run_v.detach(),
            offset = offset
        )

        return out, next_cache

    def forward(
        self,
        inp,
        cache = None,
        return_cache = False,
        sliding_window_flex_mask = None,
        **kwargs
    ):
        if exists(cache):
            return self.forward_inference(inp, cache = cache, return_cache = return_cache)

        return self.forward_train(
            inp,
            return_cache = return_cache,
            sliding_window_flex_mask = sliding_window_flex_mask,
            **kwargs
        )

    def pop_extra_loss(self):
        loss = self._extra_loss
        self._extra_loss = None
        return loss
