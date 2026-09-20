"""Qwen3-based attention, encoder, and decoder layers for OmniNovo."""
from typing import Callable, Optional, Tuple

import torch
from torch import nn

from .transformers_utils import ACT2FN, compute_default_rope_parameters

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input, unpad_input
except Exception as e:
    print(f"flash-attention not installed. Error: {e}")

rope_theta = 10000

def prepare_4d_padding_mask(
    attention_mask: Optional[torch.Tensor],
    dtype: torch.dtype = torch.bfloat16,
):
    if attention_mask is None:
        return None
    if attention_mask.dim() == 4:
        return attention_mask  # Already in the desired format

    padding_mask_4d = attention_mask[:, None, None, :].to(dtype)

    min_dtype = torch.finfo(dtype).min
    inverted_padding_mask = padding_mask_4d * min_dtype

    return inverted_padding_mask

class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """Qwen3RMSNorm is equivalent to T5LayerNorm."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

class Qwen3QK_RMSNorm(nn.Module):
    def __init__(self, hidden_size, use_bf16=False, eps=1e-6):
        """RMSNorm for query/key normalization with optional bf16 output."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.use_bf16 = use_bf16

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype if not self.use_bf16 else torch.bfloat16
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN["silu"]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q=None, k=None, cos=None, sin=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q: The query tensor.
        k: The key tensor.
        cos: The cosine part of the rotary embedding.
        sin: The sine part of the rotary embedding.
        unsqueeze_dim: The dimension along which to unsqueeze cos and sin
            so that they can be properly broadcasted to the dimensions of q and k.
    Returns:
        Tuple of the query and key tensors rotated using the Rotary Position Embedding.
    """
    dtype = q.dtype if q is not None else k.dtype
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = ((q * cos) + (rotate_half(q) * sin)).to(dtype) if q is not None else None
    k_embed = ((k * cos) + (rotate_half(k) * sin)).to(dtype) if k is not None else None
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim).
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class Qwen3RotaryEmbedding(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        inv_freq, self.attention_scaling = compute_default_rope_parameters(rope_theta=rope_theta, head_dim=config.head_dim, device=device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

def cross_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    memory_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if memory_mask is not None:
        causal_mask = memory_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

def flash_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    if not torch.all(attention_mask):
        batch_size, _, seq_len_q, _ = query.shape
        _, _, seq_len_k, _ = key.shape

        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)

        query_states = query.transpose(1, 2).reshape(batch_size, seq_len_q, module.num_attention_heads, module.head_dim).contiguous()
        key_states = key_states.transpose(1, 2).reshape(batch_size, seq_len_k, module.num_attention_heads, module.head_dim).contiguous()
        value_states = value_states.transpose(1, 2).reshape(batch_size, seq_len_k, module.num_attention_heads, module.head_dim).contiguous()

        query_states, indices_q, cu_seqlens_q, max_seqlen_in_batch_q, _ = unpad_input(query_states, attention_mask)
        key_states, indices_k, cu_seqlens_k, max_seqlen_in_batch_k, _ = unpad_input(key_states, attention_mask)
        value_states, indices_v, cu_seqlens_v, max_seqlen_in_batch_v, _ = unpad_input(value_states, attention_mask)

        attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=scaling,
                causal=False,
            )
        attn_output = pad_input(attn_output_unpad, indices_q, batch_size, seq_len_q)

    else:
        query_states = query.transpose(1, 2).contiguous()
        key_states = key.transpose(1, 2).contiguous()
        value_states = value.transpose(1, 2).contiguous()
        attn_output = flash_attn_func(
            query_states,
            key_states,
            value_states,
            dropout_p=dropout,
            softmax_scale=scaling,
            causal=False,
        )

    return attn_output, None

def flash_cross_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    memory_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    if not torch.all(memory_mask):
        batch_size, _, seq_len_q, _ = query.shape
        _, _, seq_len_k, _ = key.shape

        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)

        query_states = query.transpose(1, 2).reshape(batch_size, seq_len_q, module.num_attention_heads, module.head_dim).contiguous()
        key_states = key_states.transpose(1, 2).reshape(batch_size, seq_len_k, module.num_attention_heads, module.head_dim).contiguous()
        value_states = value_states.transpose(1, 2).reshape(batch_size, seq_len_k, module.num_attention_heads, module.head_dim).contiguous()

        query_states, indices_q, cu_seqlens_q, max_seqlen_in_batch_q, _ = unpad_input(query_states, attention_mask)
        key_states, indices_k, cu_seqlens_k, max_seqlen_in_batch_k, _ = unpad_input(key_states, memory_mask)
        value_states, indices_v, cu_seqlens_v, max_seqlen_in_batch_v, _ = unpad_input(value_states, memory_mask)

        attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=scaling,
                causal=False,
            )
        attn_output = pad_input(attn_output_unpad, indices_q, batch_size, seq_len_q)

    else:
        query_states = query.transpose(1, 2).contiguous()
        key_states = key.transpose(1, 2).contiguous()
        value_states = value.transpose(1, 2).contiguous()
        attn_output = flash_attn_func(
            query_states,
            key_states,
            value_states,
            dropout_p=dropout,
            softmax_scale=scaling,
            causal=False,
        )

    return attn_output, None

class Qwen3Attention(nn.Module):
    """Multi-headed self-attention with QK-norm and RoPE."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.num_attention_heads = config.num_attention_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=False
        )
        self.q_norm = Qwen3QK_RMSNorm(self.head_dim, self.config.use_bf16, eps=1e-06)
        self.k_norm = Qwen3QK_RMSNorm(self.head_dim, self.config.use_bf16, eps=1e-06)
        self.sliding_window = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if self.config.flash_attn:
            attention_interface = flash_attention_forward
        else:
            attention_interface = eager_attention_forward

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

class Qwen3CrossAttention(nn.Module):
    """Multi-headed cross-attention with QK-norm and RoPE."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.num_attention_heads = config.num_attention_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=False
        )
        self.q_norm = Qwen3QK_RMSNorm(self.head_dim, self.config.use_bf16, eps=1e-06)
        self.k_norm = Qwen3QK_RMSNorm(self.head_dim, self.config.use_bf16, eps=1e-06)
        self.sliding_window = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory: Optional[torch.Tensor],
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        memory_position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
        attention_mask: Optional[torch.Tensor],
        memory_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        memory_shape = memory.shape[:-1] + (-1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(memory).view(memory_shape)).transpose(1, 2)
        value_states = self.v_proj(memory).view(memory_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, _ = apply_rotary_pos_emb(query_states, None, cos, sin)

        memory_cos, memory_sin = memory_position_embeddings
        _, key_states = apply_rotary_pos_emb(None, key_states, memory_cos, memory_sin)

        if self.config.flash_attn:
            attention_interface = flash_cross_attention_forward
        else:
            attention_interface = cross_attention_forward

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            memory_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

class Qwen3EncoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=1e-06)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=1e-06)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs

class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)
        self.cross_attn = Qwen3CrossAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=1e-06)
        self.cross_attn_layernorm = Qwen3RMSNorm(config.hidden_size, eps=1e-06)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=1e-06)

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        memory_position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Cross Attention
        residual = hidden_states
        hidden_states = self.cross_attn_layernorm(hidden_states)
        hidden_states, cross_attn_weights = self.cross_attn(
            hidden_states=hidden_states,
            memory=memory,
            attention_mask=attention_mask,
            memory_mask=memory_mask,
            position_embeddings=position_embeddings,
            memory_position_embeddings=memory_position_embeddings
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights, )

        return outputs

class Qwen3Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.layers = nn.ModuleList(
            [Qwen3EncoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=1e-06)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)

    def forward(
        self,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = True,
        output_hidden_states: Optional[bool] = True,
    ):
        if position_ids is None:
            position_ids = torch.arange(
                inputs_embeds.shape[1], dtype=torch.long, device=inputs_embeds.device
            )
            position_ids = position_ids.unsqueeze(0).expand(inputs_embeds.shape[0], -1)

        hidden_states = inputs_embeds

        # Create position embeddings to be shared across the encoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Encoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        if not self.config.flash_attn:
            attention_mask = prepare_4d_padding_mask(attention_mask, dtype=hidden_states.dtype)
        else:
            if attention_mask.dtype == torch.bool:
                attention_mask = torch.logical_not(attention_mask)
            elif attention_mask.dtype == torch.int:
                attention_mask = torch.logical_not(attention_mask.bool())

        for encoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = encoder_layer(
                hidden_states,
                position_ids=position_ids,
                output_attentions=output_attentions,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # Add hidden states from the last encoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return hidden_states, all_hidden_states, all_self_attns

class Qwen3Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=1e-06)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)

    def forward(
        self,
        memory: Optional[torch.LongTensor] = None,
        memory_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = True,
        output_hidden_states: Optional[bool] = True,
    ):
        if position_ids is None:
            position_ids = torch.arange(
                inputs_embeds.shape[1], dtype=torch.long, device=inputs_embeds.device
            )
            position_ids = position_ids.unsqueeze(0).expand(inputs_embeds.shape[0], -1)

        memory_position_ids = torch.arange(
            memory.shape[1], dtype=torch.long, device=memory.device
        )
        memory_position_ids = memory_position_ids.unsqueeze(0).expand(memory.shape[0], -1)

        hidden_states = inputs_embeds

        # Create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        memory_position_embeddings = self.rotary_emb(memory, memory_position_ids)

        # Decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attns = () if output_attentions else None

        if not self.config.flash_attn:
            attention_mask = prepare_4d_padding_mask(attention_mask, dtype=hidden_states.dtype)
        else:
            if attention_mask.dtype == torch.bool:
                attention_mask = torch.logical_not(attention_mask)
            elif attention_mask.dtype == torch.int:
                attention_mask = torch.logical_not(attention_mask.bool())
        if not self.config.flash_attn:
            memory_mask = prepare_4d_padding_mask(memory_mask, dtype=hidden_states.dtype)
        else:
            if memory_mask.dtype == torch.bool:
                memory_mask = torch.logical_not(memory_mask)
            elif memory_mask.dtype == torch.int:
                memory_mask = torch.logical_not(memory_mask.bool())

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                memory=memory,
                memory_mask=memory_mask,
                output_attentions=output_attentions,
                position_embeddings=position_embeddings,
                memory_position_embeddings=memory_position_embeddings,
                attention_mask=attention_mask,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                all_cross_attns += (layer_outputs[2],)

        hidden_states = self.norm(hidden_states)

        # Add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return hidden_states, all_hidden_states, all_self_attns, all_cross_attns
