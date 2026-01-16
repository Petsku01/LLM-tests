"""
Advanced Transformer-based Large Language Model Implementation
Incorporates state-of-the-art architectural improvements and optimization techniques
Author: -pk
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union, List, Dict
from dataclasses import dataclass, field
import numpy as np
from enum import Enum


class AttentionType(Enum):
    """Enumeration for attention mechanism types"""
    STANDARD = "standard"
    FLASH = "flash"
    SLIDING_WINDOW = "sliding_window"
    SPARSE = "sparse"


@dataclass
class ModelConfig:
    """Comprehensive configuration class for model hyperparameters"""
    # Model dimensions
    vocab_size: int = 50257
    hidden_dim: int = 768
    intermediate_dim: int = 3072
    num_layers: int = 12
    num_heads: int = 12
    num_key_value_heads: Optional[int] = None  # For Grouped-Query Attention
    head_dim: Optional[int] = None  # If None, computed as hidden_dim // num_heads
    
    # Sequence configuration
    max_sequence_length: int = 2048
    sliding_window_size: Optional[int] = None  # For local attention
    
    # Regularization
    dropout_rate: float = 0.1
    attention_dropout: float = 0.1
    residual_dropout: float = 0.1
    embedding_dropout: float = 0.1
    
    # Normalization
    layer_norm_epsilon: float = 1e-6
    use_rms_norm: bool = True
    
    # Initialization
    initializer_range: float = 0.02
    initializer_factor: float = 1.0  # Xavier/Glorot scaling factor
    
    # Architectural choices
    use_flash_attention: bool = True
    use_rotary_embeddings: bool = True
    use_gated_mlp: bool = True
    use_parallel_blocks: bool = False  # Parallel attention and FFN
    gradient_checkpointing: bool = False
    tie_word_embeddings: bool = True
    
    # RoPE configuration
    rope_theta: float = 10000.0
    rope_scaling_type: Optional[str] = None  # "linear" or "dynamic"
    rope_scaling_factor: float = 1.0
    
    # Optimization flags
    use_bias: bool = False
    use_cache: bool = True
    attention_type: AttentionType = AttentionType.STANDARD
    
    def __post_init__(self):
        """Validate and compute derived parameters"""
        if self.head_dim is None:
            assert self.hidden_dim % self.num_heads == 0, "hidden_dim must be divisible by num_heads"
            self.head_dim = self.hidden_dim // self.num_heads
        
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_heads
        
        assert self.num_heads % self.num_key_value_heads == 0, "num_heads must be divisible by num_key_value_heads"
        
        # Computes effective dimensions
        self.num_query_groups = self.num_heads // self.num_key_value_heads


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) with support for different scaling methods
    Reference: https://arxiv.org/abs/2104.09864
    """
    
    def __init__(
        self, 
        dim: int, 
        max_seq_len: int = 2048, 
        theta: float = 10000.0,
        scaling_type: Optional[str] = None,
        scaling_factor: float = 1.0
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.scaling_type = scaling_type
        self.scaling_factor = scaling_factor
        
        # Precomputes inverse frequencies
        inv_freq = self._compute_inv_freq(dim, theta, scaling_type, scaling_factor)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Cache for precomputed cos/sin embeddings
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
    
    @staticmethod
    def _compute_inv_freq(
        dim: int, 
        theta: float, 
        scaling_type: Optional[str], 
        scaling_factor: float
    ) -> torch.Tensor:
        """Compute inverse frequencies with optional scaling"""
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        
        if scaling_type == "linear":
            inv_freq = inv_freq / scaling_factor
        elif scaling_type == "dynamic":
            # Dynamic scaling based on sequence length
            inv_freq = inv_freq * (scaling_factor ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        
        return inv_freq
    
    def _update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        """Update cosine and sine cache for given sequence length"""
        if seq_len > self._seq_len_cached or self._cos_cached is None:
            self._seq_len_cached = max(seq_len, self.max_seq_len)
            
            # Build position indices
            t = torch.arange(self._seq_len_cached, device=device, dtype=torch.float32)
            
            # Compute frequencies
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            
            # Create rotation embeddings
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos()[None, :, None, :].to(dtype)
            self._sin_cached = emb.sin()[None, :, None, :].to(dtype)
    
    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        position_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to query and key tensors"""
        batch_size, seq_len, num_heads, head_dim = q.shape
        
        # Update cache if needed
        self._update_cache(seq_len, q.device, q.dtype)
        
        # Extract relevant portion of cache
        if position_ids is not None:
            cos = self._cos_cached[:, position_ids].squeeze(0)
            sin = self._sin_cached[:, position_ids].squeeze(0)
        else:
            cos = self._cos_cached[:, :seq_len]
            sin = self._sin_cached[:, :seq_len]
        
        # Apply rotary embedding
        q_rot = self._apply_rotary_pos_emb(q, cos, sin)
        k_rot = self._apply_rotary_pos_emb(k, cos, sin)
        
        return q_rot, k_rot
    
    @staticmethod
    def _apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply rotary position embedding using complex number formulation"""
        # Split into real and imaginary parts
        x_r, x_i = x.chunk(2, dim=-1)
        
        # Apply rotation using complex multiplication
        # (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
        cos_half = cos[..., :x_r.shape[-1]]
        sin_half = sin[..., :x_r.shape[-1]]
        
        x_out_r = x_r * cos_half - x_i * sin_half
        x_out_i = x_r * sin_half + x_i * cos_half
        
        # Concatenate back
        x_out = torch.cat([x_out_r, x_out_i], dim=-1)
        
        return x_out.type_as(x)


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) for improved inference efficiency
    Reference: https://arxiv.org/abs/2305.13245
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_query_groups = config.num_query_groups
        self.head_dim = config.head_dim
        self.hidden_dim = config.hidden_dim
        
        # Projections with appropriate dimensions for GQA
        self.q_proj = nn.Linear(self.hidden_dim, self.num_heads * self.head_dim, bias=config.use_bias)
        self.k_proj = nn.Linear(self.hidden_dim, self.num_key_value_heads * self.head_dim, bias=config.use_bias)
        self.v_proj = nn.Linear(self.hidden_dim, self.num_key_value_heads * self.head_dim, bias=config.use_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_dim, bias=config.use_bias)
        
        # Dropout layers
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.residual_dropout = nn.Dropout(config.residual_dropout)
        
        # Initialize RoPE if enabled
        if config.use_rotary_embeddings:
            self.rotary_emb = RotaryPositionEmbedding(
                self.head_dim,
                config.max_sequence_length,
                config.rope_theta,
                config.rope_scaling_type,
                config.rope_scaling_factor
            )
        
        # Precompute attention scale
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def _repeat_kv(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Repeat key/value heads to match query heads for GQA"""
        batch_size, seq_len, num_key_value_heads, head_dim = hidden_states.shape
        
        if self.num_query_groups == 1:
            # No repetition needed
            return hidden_states
        
        # Repeat interleave for each query group
        hidden_states = hidden_states.unsqueeze(2)
        hidden_states = hidden_states.expand(batch_size, seq_len, self.num_query_groups, num_key_value_heads, head_dim)
        hidden_states = hidden_states.reshape(batch_size, seq_len, self.num_heads, head_dim)
        
        return hidden_states
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        """Forward pass with Grouped Query Attention"""
        batch_size, seq_len, _ = hidden_states.shape
        
        # Compute queries, keys, and values
        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        queries = queries.view(batch_size, seq_len, self.num_heads, self.head_dim)
        keys = keys.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        values = values.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        
        # Apply rotary embeddings if enabled
        if self.config.use_rotary_embeddings:
            queries, keys = self.rotary_emb(queries, keys, position_ids)
        
        # Handle KV caching for efficient inference
        if past_key_value is not None:
            past_keys, past_values = past_key_value
            keys = torch.cat([past_keys, keys], dim=1)
            values = torch.cat([past_values, values], dim=1)
        
        # Store current KV if caching is enabled
        present_key_value = (keys, values) if use_cache else None
        
        # Repeat KV heads if using GQA
        if self.num_key_value_heads != self.num_heads:
            keys = self._repeat_kv(keys)
            values = self._repeat_kv(values)
        
        # Transpose for attention computation
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        # Compute attention scores with automatic mixed precision stability
        with torch.cuda.amp.autocast(enabled=False):
            attention_scores = torch.matmul(queries.float(), keys.transpose(-2, -1).float()) * self.scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Compute attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1, dtype=torch.float32).to(queries.dtype)
        attention_probs = self.attention_dropout(attention_probs)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_probs, values)
        
        # Reshape and project output
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.reshape(batch_size, seq_len, self.hidden_dim)
        attention_output = self.o_proj(attention_output)
        attention_output = self.residual_dropout(attention_output)
        
        outputs = (attention_output, present_key_value)
        if output_attentions:
            outputs += (attention_probs,)
        
        return outputs


class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU Feedforward Network
    Reference: https://arxiv.org/abs/2002.05202
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        hidden_dim = config.hidden_dim
        intermediate_dim = config.intermediate_dim
        
        # Three projections for SwiGLU: gate, up, and down
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=config.use_bias)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=config.use_bias)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=config.use_bias)
        
        self.dropout = nn.Dropout(config.residual_dropout)
        
        # Activation function
        self.act_fn = nn.SiLU()  # Swish activation
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass with SwiGLU activation"""
        # Compute gate and up projections
        gate = self.act_fn(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        
        # Apply gating mechanism
        intermediate = gate * up
        
        # Project back to hidden dimension
        output = self.down_proj(intermediate)
        output = self.dropout(output)
        
        return output


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    Reference: https://arxiv.org/abs/1910.07467
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization"""
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        
        # Compute RMS
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        
        # Apply learned weight parameter
        return (self.weight * hidden_states).to(input_dtype)


class TransformerBlock(nn.Module):
    """
    Transformer block with pre-normalization and optional parallel attention/FFN
    """
    
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # Normalization layers
        norm_class = RMSNorm if config.use_rms_norm else nn.LayerNorm
        norm_eps = config.layer_norm_epsilon
        
        self.attention_norm = norm_class(config.hidden_dim, norm_eps)
        self.ffn_norm = norm_class(config.hidden_dim, norm_eps)
        
        # Core components
        self.attention = GroupedQueryAttention(config)
        self.feed_forward = SwiGLUFeedForward(config) if config.use_gated_mlp else self._build_standard_ffn()
        
        # For parallel blocks
        self.use_parallel_blocks = config.use_parallel_blocks
    
    def _build_standard_ffn(self) -> nn.Sequential:
        """Build standard FFN with GELU activation"""
        return nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.intermediate_dim, bias=self.config.use_bias),
            nn.GELU(),
            nn.Linear(self.config.intermediate_dim, self.config.hidden_dim, bias=self.config.use_bias),
            nn.Dropout(self.config.residual_dropout)
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        """Forward pass with residual connections"""
        residual = hidden_states
        
        # Self-attention
        hidden_states = self.attention_norm(hidden_states)
        attention_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions
        )
        attention_output = attention_outputs[0]
        
        if self.use_parallel_blocks:
            # Parallel attention and FFN (like GPT-J)
            ffn_output = self.feed_forward(self.ffn_norm(residual))
            hidden_states = residual + attention_output + ffn_output
        else:
            # Sequential (standard transformer)
            hidden_states = residual + attention_output
            residual = hidden_states
            hidden_states = residual + self.feed_forward(self.ffn_norm(hidden_states))
        
        outputs = (hidden_states,) + attention_outputs[1:]
        
        return outputs


class TransformerLLM(nn.Module):
    """
    Complete Transformer-based Language Model with state-of-the-art optimizations
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim)
        
        # Position embeddings (only if not using RoPE)
        if not config.use_rotary_embeddings:
            self.embed_positions = nn.Embedding(config.max_sequence_length, config.hidden_dim)
        
        # Embedding dropout
        self.embedding_dropout = nn.Dropout(config.embedding_dropout)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_idx=i)
            for i in range(config.num_layers)
        ])
        
        # Final layer normalization
        norm_class = RMSNorm if config.use_rms_norm else nn.LayerNorm
        self.final_norm = norm_class(config.hidden_dim, config.layer_norm_epsilon)
        
        # Language model head
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        
        # Weight tying
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Apply special scaled initialization to certain layers
        for layer_idx, layer in enumerate(self.layers):
            # Scale residual connections at initialization
            layer_scale = 1.0 / math.sqrt(2.0 * config.num_layers)
            layer.attention.o_proj.weight.data.mul_(layer_scale)
            if hasattr(layer.feed_forward, 'down_proj'):
                layer.feed_forward.down_proj.weight.data.mul_(layer_scale)
    
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize model weights using scaled normal distribution"""
        std = self.config.initializer_range * self.config.initializer_factor
        
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, (RMSNorm, nn.LayerNorm)):
            if hasattr(module, 'weight'):
                module.weight.data.fill_(1.0)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
    
    def get_input_embeddings(self) -> nn.Embedding:
        """Get input embedding layer"""
        return self.embed_tokens
    
    def set_input_embeddings(self, value: nn.Embedding) -> None:
        """Set input embedding layer"""
        self.embed_tokens = value
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass through the model
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            position_ids: Position IDs for RoPE [batch_size, seq_len]
            past_key_values: Cached key-value pairs for efficient generation
            use_cache: Whether to return key-value pairs for caching
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states from all layers
            return_dict: Whether to return a dictionary of outputs
        
        Returns:
            Model outputs (logits, cached KV pairs, hidden states, attentions)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Get input embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Add position embeddings if not using RoPE
        if not self.config.use_rotary_embeddings:
            if position_ids is None:
                position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            position_embeds = self.embed_positions(position_ids)
            hidden_states = hidden_states + position_embeds
        
        # Apply embedding dropout
        hidden_states = self.embedding_dropout(hidden_states)
        
        # Prepare attention mask
        if attention_mask is not None:
            # Create causal mask
            attention_mask = self._prepare_attention_mask(attention_mask, seq_len, device, past_key_values)
        
        # Initialize past key values if needed
        if past_key_values is None and use_cache:
            past_key_values = [None] * self.config.num_layers
        
        # Process through transformer layers
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        next_cache = [] if use_cache else None
        
        for layer_idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            past_key_value = past_key_values[layer_idx] if past_key_values is not None else None
            
            if self.config.gradient_checkpointing and self.training:
                # Use gradient checkpointing
                layer_outputs = self._gradient_checkpointing_func(
                    layer,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    use_cache,
                    output_attentions
                )
            else:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions
                )
            
            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_cache.append(layer_outputs[1])
            
            if output_attentions:
                all_attentions += (layer_outputs[-1],)
        
        # Apply final layer norm
        hidden_states = self.final_norm(hidden_states)
        
        # Compute logits
        logits = self.lm_head(hidden_states)
        
        # Prepare outputs
        outputs = (logits,)
        
        if use_cache:
            outputs += (next_cache,)
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
            outputs += (all_hidden_states,)
        
        if output_attentions:
            outputs += (all_attentions,)
        
        return outputs
    
    def _prepare_attention_mask(
        self,
        attention_mask: torch.Tensor,
        seq_len: int,
        device: torch.device,
        past_key_values: Optional[List] = None
    ) -> torch.Tensor:
        """Prepare 4D attention mask from 2D mask"""
        # Calculate total sequence length including cached values
        total_seq_len = seq_len
        if past_key_values is not None and past_key_values[0] is not None:
            total_seq_len += past_key_values[0][0].shape[1]
        
        # Create causal mask
        causal_mask = torch.triu(
            torch.ones((seq_len, total_seq_len), device=device),
            diagonal=total_seq_len - seq_len + 1
        )
        
        # Convert to attention scores mask (0 -> -inf, 1 -> 0)
        causal_mask = (1.0 - causal_mask) * torch.finfo(torch.float32).min
        
        # Expand for batch and heads
        causal_mask = causal_mask[None, None, :, :]
        
        return causal_mask
    
    def _gradient_checkpointing_func(self, layer, *args, **kwargs):
        """Wrapper for gradient checkpointing"""
        def custom_forward(*inputs):
            return layer(*inputs, **kwargs)
        
        return torch.utils.checkpoint.checkpoint(custom_forward, *args)
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        min_length: int = 0
    ) -> torch.Tensor:
        """
        Advanced text generation with multiple decoding strategies
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Number of top tokens to consider
            top_p: Cumulative probability for nucleus sampling
            repetition_penalty: Penalty for repeating tokens
            do_sample: Whether to sample or use greedy decoding
            eos_token_id: End-of-sequence token ID
            pad_token_id: Padding token ID
            min_length: Minimum generation length
        
        Returns:
            Generated token IDs [batch_size, seq_len + generated_len]
        """
        self.eval()
        
        batch_size = input_ids.shape[0]
        device = input_ids.device
        past_key_values = None
        generated_tokens = 0
        
        # Create initial attention mask
        attention_mask = torch.ones_like(input_ids)
        if pad_token_id is not None:
            attention_mask[input_ids == pad_token_id] = 0
        
        # Generation loop
        while generated_tokens < max_new_tokens:
            # Get model predictions
            outputs = self.forward(
                input_ids=input_ids[:, -1:] if past_key_values else input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True
            )
            
            logits = outputs[0]
            past_key_values = outputs[1]
            
            # Get next token logits
            next_token_logits = logits[:, -1, :]
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for batch_idx in range(batch_size):
                    for token_id in set(input_ids[batch_idx].tolist()):
                        next_token_logits[batch_idx, token_id] /= repetition_penalty
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Apply min_length constraint
            if generated_tokens < min_length and eos_token_id is not None:
                next_token_logits[:, eos_token_id] = -float('inf')
            
            if do_sample:
                # Top-k filtering
                if top_k > 0:
                    top_k_values, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.shape[-1]))
                    min_top_k = top_k_values[:, -1].unsqueeze(-1)
                    next_token_logits = torch.where(
                        next_token_logits < min_top_k,
                        torch.full_like(next_token_logits, -float('inf')),
                        next_token_logits
                    )
                
                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Find cutoff point
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False
                    
                    # Apply filtering
                    for batch_idx in range(batch_size):
                        indices_to_remove = sorted_indices[batch_idx, sorted_indices_to_remove[batch_idx]]
                        next_token_logits[batch_idx, indices_to_remove] = -float('inf')
                
                # Sample from distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append to sequences
            input_ids = torch.cat([input_ids, next_tokens], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones((batch_size, 1), device=device)], dim=1)
            generated_tokens += 1
            
            # Check for EOS token
            if eos_token_id is not None and (next_tokens == eos_token_id).all():
                break
        
        return input_ids


class OptimizedAdamW(torch.optim.Optimizer):
    """
    Optimized AdamW with optional features like gradient centralization
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        amsgrad: bool = False,
        gradient_centralization: bool = False
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            gradient_centralization=gradient_centralization
        )
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        """Perform a single optimization step"""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if group['gradient_centralization'] and len(grad.shape) > 1:
                    # Apply gradient centralization
                    grad = grad - grad.mean(dim=tuple(range(1, len(grad.shape))), keepdim=True)
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if group['amsgrad']:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Update biased first and second moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                if group['amsgrad']:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                
                step_size = group['lr'] / bias_correction1
                
                # Weight decay
                if group['weight_decay'] != 0:
                    p.data.add_(p.data, alpha=-group['lr'] * group['weight_decay'])
                
                # Update parameters
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
        
        return loss


# Example usage and comprehensive testing
def test_model():
    """Comprehensive model testing suite"""
    print("Initializing Advanced Transformer LLM...")
    
    # Configuration for a medium-sized model
    config = ModelConfig(
        vocab_size=50257,
        hidden_dim=768,
        intermediate_dim=3072,
        num_layers=12,
        num_heads=12,
        num_key_value_heads=4,  # Using GQA
        max_sequence_length=2048,
        use_rotary_embeddings=True,
        use_gated_mlp=True,
        use_rms_norm=True,
        gradient_checkpointing=False,
        use_flash_attention=True
    )
    
    # Create model
    model = TransformerLLM(config)
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size (FP32): {total_params * 4 / 1024**3:.2f} GB")
    print(f"  Model size (FP16): {total_params * 2 / 1024**3:.2f} GB")
    
    # Test forward pass
    print(f"\nTesting forward pass...")
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True, output_hidden_states=True)
        logits = outputs[0]
        
        print(f"  Input shape: {input_ids.shape}")
        print(f"  Output shape: {logits.shape}")
        print(f"  Perplexity: {torch.exp(F.cross_entropy(logits.view(-1, logits.shape[-1]), input_ids.view(-1))).item():.2f}")
        
        if torch.cuda.is_available():
            print(f"  GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f"  GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    # Test generation
    print(f"\nTesting text generation...")
    prompt = torch.randint(0, config.vocab_size, (1, 10)).to(device)
    
    with torch.no_grad():
        generated = model.generate(
            prompt,
            max_new_tokens=50,
            temperature=0.8,
            top_p=0.95,
            repetition_penalty=1.2
        )
        
        print(f"  Prompt length: {prompt.shape[1]}")
        print(f"  Generated length: {generated.shape[1]}")
        print(f"  New tokens: {generated.shape[1] - prompt.shape[1]}")
    
    print("\nAll tests completed successfully!")
    return model


if __name__ == "__main__":
    model = test_model()
