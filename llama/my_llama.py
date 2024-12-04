import math
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class TiedLinear:
    def __init__(self, tied_module: nn.Module):
        self.tied_module = tied_module

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.tied_module.weight)


class RMSNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.normalized_shape = (dim,)
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # computation is in fp32
        return F.rms_norm(
            x.float(),
            normalized_shape=self.normalized_shape,
            weight=self.scale,
            eps=1e-5,
        ).to(x.dtype)


class FeedForward(nn.Module):
    def __init__(self, *, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Llama3ScaledRoPE(nn.Module):
    """
    This class implements Rotary Positional Embeddings (RoPE)
    proposed in https://arxiv.org/abs/2104.09864 with additional
    scaling from https://github.com/meta-llama/llama-models/blob/dc42f22a3b05502e7296402b019a51f57fa045c9/models/llama3_1.

    In this implementation we cache the embeddings for each position upto
    ``max_seq_len`` by computing this during init.

    Default scaling factors are from the following Meta-Llama code:
    https://github.com/meta-llama/llama-models/blob/dc42f22a3b05502e7296402b019a51f57fa045c9/models/llama3_1/api/model.py#L41

    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ````embed_dim`` // ``num_heads````
        max_seq_len (int): Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed
        base (int): The base for the geometric progression used to compute
            the rotation angles
        scale_factor (int): scaling factor for theta. Default: 8
        low_freq_factor (int): low frequency factor for scaling theta. Default: 1
        high_freq_factor (int): high frequency factor for scaling theta. Default: 4
        old_context_len (int): old context length for scaling theta. Default: 8192
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: int = 10_000,
        scale_factor: int = 8,
        low_freq_factor: int = 1,
        high_freq_factor: int = 4,
        old_context_len: int = 8192,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len

        self.scale_factor = scale_factor
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.old_context_len = old_context_len
        self.is_cache_built = False
        self.rope_init()

    def rope_init(self):
        """
        Warning: this is called in recipes before torch.compile,
        so that the cache is built in advance.
        """
        freqs = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )

        # If we're on meta device return early.
        # We can't apply scaling until freqs is filled with real data
        if freqs.is_meta:
            return

        theta = self.apply_scaling(
            freqs,
            self.scale_factor,
            self.low_freq_factor,
            self.high_freq_factor,
            self.old_context_len,
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)
        self.is_cache_built = True

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )

        # Outer product of theta and position index; output tensor has
        # a shape of [max_seq_len, dim // 2]
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()

        # cache includes both the cos and sin components and so the output shape is
        # [max_seq_len, dim // 2, 2]
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def apply_scaling(
        self,
        freqs: torch.Tensor,
        scale_factor: int,
        low_freq_factor: int,
        high_freq_factor: int,
        old_context_len: int,
    ):

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor
        new_freqs = []
        for freq in freqs:
            wavelen = 2 * math.pi / freq
            if wavelen < high_freq_wavelen:
                new_freqs.append(freq)
            elif wavelen > low_freq_wavelen:
                new_freqs.append(freq / scale_factor)
            else:
                assert low_freq_wavelen != high_freq_wavelen
                smooth = (old_context_len / wavelen - low_freq_factor) / (
                    high_freq_factor - low_freq_factor
                )
                new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
        return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)

    def forward(
        self, x: torch.Tensor, *, input_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape
                [b, s, n_h, h_d]
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b, s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.

        Returns:
            Tensor: output tensor with RoPE applied

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - n_h: num heads
            - h_d: head dim

        Raises:
            RuntimeError: if RoPE cache is not initialized prior to forward call
        """

        if not self.is_cache_built:
            raise RuntimeError(
                "RoPE cache is not built. Please call rope_init() first."
            )

        # input tensor has shape [b, s, n_h, h_d]
        seq_len = x.size(1)

        # extract the values based on whether input_pos is set or not
        rope_cache = (
            self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        )

        # reshape input; the last dimension is used for computing the output.
        # Cast to float to match the reference implementation
        # tensor has shape [b, s, n_h, h_d // 2, 2]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

        # reshape the cache for broadcasting
        # tensor has shape [b, s, 1, h_d // 2, 2] if packed samples,
        # otherwise has shape [1, s, 1, h_d // 2, 2]
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)

        # tensor has shape [b, s, n_h, h_d // 2, 2]
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        # tensor has shape [b, s, n_h, h_d]
        x_out = x_out.flatten(3)
        return x_out.type_as(x)


class MultiHeadAttention(nn.Module):
    """Multi-headed attention layer with support for grouped query
    attention (GQA) introduced in https://arxiv.org/abs/2305.13245v1.

    GQA is a version of multiheaded attention (MHA) which uses fewer
    key/value heads than query heads by grouping n query heads for each
    key and value head. Multi-Query Attention is an extreme
    version where we have a single key and value head shared by all
    query heads.

    Following is an example of MHA, GQA and MQA with num_heads = 4

    (credit for the documentation:
    `litgpt.Config <https://github.com/Lightning-AI/litgpt/blob/eda1aaaf391fd689664f95487ab03dc137e213fd/litgpt/config.py>`_).


    ::

        ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
        │ v ││ v ││ v ││ v │     │ v │    │ v │             │ v │
        └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
        │    │    │    │         │        │                 │
        ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
        │ k ││ k ││ k ││ k │     │ k │    │ k │             │ k │
        └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
        │    │    │    │      ┌──┴──┐  ┌──┴──┐      ┌────┬──┴─┬────┐
        ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐
        │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │
        └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘
        ◀──────────────────▶  ◀──────────────────▶  ◀──────────────────▶
                MHA                    GQA                   MQA
        n_kv_heads =4          n_kv_heads=2           n_kv_heads=1

    Args:
        embed_dim (int): embedding dimension for the model
        num_heads (int): number of query heads. For MHA this is also the
            number of heads for key and value
        num_kv_heads (int): number of key and value heads. User should ensure
            ``num_heads % num_kv_heads == 0``. For standard MHA set ``num_kv_heads == num_heads``,
            for GQA ``num_kv_heads < num_heads``, and for MQA set ``num_kv_heads == 1``.
        head_dim (int): dimension of each head, calculated by ``embed_dim // num_heads``.
        q_proj (nn.Module): projection layer for query.
        k_proj (nn.Module): projection layer for key.
        v_proj (nn.Module): projection layer for value.
        output_proj (nn.Module): projection layer for output.
        pos_embeddings (Optional[nn.Module]): positional embeddings layer, e.g. RotaryPositionalEmbeddings.
        q_norm (Optional[nn.Module]): normalization layer for query, e.g. RMSNorm. For decoding, this is applied
            before updating from kv_cache. This means it will only support token wide normalization and not
            batch or sequence wide normalization.
        k_norm (Optional[nn.Module]): normalization layer for key, must be set if q_norm is.
        kv_cache (Optional[KVCache]): KVCache object used to cache key and value
        max_seq_len (int): maximum sequence length supported by the model.
            This is needed to compute the RoPE Cache. Default: 4096.
        is_causal (bool): sets the default mask to causal when no mask is provided
        attn_dropout (float): dropout value passed onto the scaled_dot_product_attention function.
            Default value is 0.0.

    Raises:
        ValueError: If ``num_heads % num_kv_heads != 0``
        ValueError: If ``embed_dim % num_heads != 0``
        ValueError: If ``attn_dropout < 0`` or ``attn_dropout > 1``
        ValueError: if q_norm is defined without k_norm or vice versa
    """

    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        q_proj: nn.Module,
        k_proj: nn.Module,
        v_proj: nn.Module,
        output_proj: nn.Module,
        pos_embeddings: Optional[nn.Module] = None,
        q_norm: Optional[nn.Module] = None,
        k_norm: Optional[nn.Module] = None,
        kv_cache=None,
        max_seq_len: int = 4096,
        is_causal: bool = True,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by "
                f"num_kv_heads ({num_kv_heads})"
            )

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )

        if attn_dropout < 0 or attn_dropout > 1:
            raise ValueError(f"attn_dropout ({embed_dim}) must be between 0.0 and 1.0")

        if bool(q_norm) ^ bool(k_norm):
            raise ValueError("q and k norm must be set together")

        # Set attributes
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.embed_dim = embed_dim
        self.attn_dropout = attn_dropout
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.is_causal = is_causal

        # Set layers
        self.kv_cache = kv_cache
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.output_proj = output_proj
        self.q_norm = q_norm
        self.k_norm = k_norm
        self.pos_embeddings = pos_embeddings

        # this flag indicates whether to update the kv-cache during forward
        # passes. when disabled, we can have the cache setup but still
        # perform normal forward passes
        self.cache_enabled = False

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        *,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape [b x s_x x d] for the query
            y (Optional[torch.Tensor]): second input tensor with shape [b x s_y x d], is the input
                for k and v. For self attention, x=y. Optional only with kv_cache enabled.
            mask (Optional[torch.Tensor]): Used to mask the scores after the query-key multiplication
                and before the softmax. Either:

                A boolean tensor with shape ``[b x s x s]``, ``[b x s x self.encoder_max_cache_seq_len]``,
                or ``[b x s x self.decoder_max_cache_seq_len]`` if using KV-cacheing with encoder/decoder layers.
                A value of True in row ``i`` and column ``j`` means token ``i`` attends to token ``j``. A value of False means
                token ``i`` does not attend to token ``j``. If no mask is specified, a causal mask
                is used by default.

                A :class:`~torch.nn.attention.flex_attention.BlockMask` for document masking in a packed sequence
                created via `create_block_mask <https://pytorch.org/blog/flexattention/#mask-mods>`_. We  use
                :func:`~torch.nn.attention.flex_attention.flex_attention` when computing attention with block masks.
                Default is None.
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b x s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.

        Raises:
            ValueError: If no ``y`` input and ``kv_cache`` is not enabled.

        Returns:
            torch.Tensor: output tensor with attention applied

        Notation used for tensor shapes:
            - b: batch size
            - s_x: sequence length for x
            - s_y: sequence length for y
            - n_h: num heads
            - n_kv: num kv heads
            - d: embed dim
            - h_d: head dim
        """
        # x has shape [b, s_x, d]
        # y has shape [b, s_y, d]
        b, s_x, _ = x.shape
        s_y = y.shape[1] if y is not None else 0

        # q has shape [b, s_x, num_heads * head_dim]
        q = self.q_proj(x)

        # number of queries per key/value
        q_per_kv = self.num_heads // self.num_kv_heads
        q = q.view(b, s_x, self.num_kv_heads * q_per_kv, self.head_dim)

        # Apply positional embeddings
        if self.pos_embeddings is not None:
            q = self.pos_embeddings(q, input_pos=input_pos)

        # [b, n_h, s_x, h_d]
        q = q.transpose(1, 2)

        # Normalize q
        if self.q_norm is not None:
            q = self.q_norm(q)

        if y is None:
            if self.kv_cache is None or not self.cache_enabled:
                raise ValueError(
                    "Must provide y input or use kv_cache to enable streaming decoding"
                )
            k = self.kv_cache.k_cache
            v = self.kv_cache.v_cache
        else:
            # Update k and v shape, positional embeddings, and normalization

            # k,v shape [b, s_y, num_kv_heads * head_dim]
            k = self.k_proj(y)
            v = self.v_proj(y)

            # Apply positional embeddings
            # k,v shape: [b, s_y, n_kv, h_d]
            k = k.view(b, s_y, -1, self.head_dim)
            v = v.view(b, s_y, -1, self.head_dim)
            if self.pos_embeddings is not None:
                k = self.pos_embeddings(k, input_pos=input_pos)

            # k,v shape: [b, n_kv, s_y, h_d]
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            # Normalize k
            if self.k_norm is not None:
                k = self.k_norm(k)

            # Update key-value cache
            if self.kv_cache is not None and self.cache_enabled:
                k, v = self.kv_cache.update(k, v)

        # If needed, expand the key and value tensors to have the same shape
        # as the query tensor by copying values across the relevant dim
        # k,v shape: [b, n_kv, s, h_d] -> [b, n_h, s, h_d]
        if self.num_heads != self.num_kv_heads:
            expand_shape = (b, self.num_kv_heads, q_per_kv, -1, self.head_dim)
            k = k.unsqueeze(2).expand(expand_shape).flatten(1, 2)
            v = v.unsqueeze(2).expand(expand_shape).flatten(1, 2)

        if mask is not None:
            mask = mask[:, None, :, :]
        output = nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=self.kv_cache is None and mask is None and self.is_causal,
        )

        # reshape the output to be the same shape as the input
        output = output.transpose(1, 2).contiguous().view(b, s_x, -1)
        return self.output_proj(output)


class TransformerSelfAttentionLayer(nn.Module):
    """
    Transformer layer derived from the Llama2 model. Normalization is applied before the attention **and** FF layer.

    Args:
        attn (MultiHeadAttention): Attention module.
        mlp (nn.Module): Feed-forward module.
        sa_norm (Optional[nn.Module]): Normalization to be applied before self-attention.
        mlp_norm (Optional[nn.Module]): Normalization to be applied before the feed-forward layer.
        sa_scale (Optional[nn.Module]): Module to scale self-attention output.
        mlp_scale (Optional[nn.Module]): Module to scale the feed-forward output.
    """

    def __init__(
        self,
        attn: MultiHeadAttention,
        mlp: nn.Module,
        *,
        sa_norm: Optional[nn.Module] = None,
        mlp_norm: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.attn = attn
        self.mlp = mlp
        self.sa_norm = sa_norm
        self.mlp_norm = mlp_norm

    def forward(
        self,
        x: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        **kwargs: Dict,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape
                [batch_size x seq_length x embed_dim]
            mask (Optional[torch.Tensor]): Used to mask the scores after the query-key multiplication
                and before the softmax. Either:

                A boolean tensor with shape ``[b x s x s]``, ``[b x s x self.encoder_max_cache_seq_len]``,
                or ``[b x s x self.encoder_max_cache_seq_len]`` if using KV-cacheing with encoder/decoder layers.
                A value of True in row ``i`` and column ``j`` means token ``i`` attends to token ``j``. A value of False means
                token ``i`` does not attend to token ``j``. If no mask is specified, a causal mask
                is used by default.

                A :class:`~torch.nn.attention.flex_attention.BlockMask` for document masking in a packed sequence
                created via `create_block_mask <https://pytorch.org/blog/flexattention/#mask-mods>`_. We  use
                :func:`~torch.nn.attention.flex_attention.flex_attention` when computing attention with block masks.
                Default is None.
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b x s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.
            **kwargs (Dict): transformer layer inputs not relevant to self attention.

        Returns:
            torch.Tensor: output tensor with same shape as input
                [batch_size x seq_length x embed_dim]
        """
        # Input tensor and attention output have the same shape
        # [b, s, d]
        # Norm applied before self-attention
        h = self.sa_norm(x)
        attn_out = self.attn(h, h, mask=mask, input_pos=input_pos)

        # Residual connection; shape: [batch_size, seq_length, embed_dim]
        h = attn_out + x

        # Norm applied before the feedforward layer
        mlp_out = self.mlp(self.mlp_norm(h))

        # Residual connection; shape: [batch_size, seq_length, embed_dim]
        out = h + mlp_out
        return out


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder derived from the Llama2 architecture.

    Args:
        tok_embeddings (nn.Embedding): PyTorch embedding layer, to be used to move
            tokens to an embedding space.
        layers (Union[nn.Module, List[nn.Module], nn.ModuleList]): A single transformer Decoder layer, an
            nn.ModuleList of layers or a list of layers. It is recommended to use an nn.ModuleList.
        max_seq_len (int): maximum sequence length the model will be run with, as used
            by :func:`~torchtune.modules.KVCache`
        num_heads (int): number of query heads. For MHA this is also the
            number of heads for key and value. This is used to setup the
            :func:`~torchtune.modules.KVCache`
        head_dim (int): embedding dimension for each head in self-attention. This is used
            to setup the :func:`~torchtune.modules.KVCache`
        norm (nn.Module): Callable that applies normalization to the output of the decoder,
            before final MLP.
        output (Union[nn.Linear, Callable]): Callable that applies a linear transformation to the output of
            the decoder.
        num_layers (Optional[int]): Number of Transformer Decoder layers, only define when
            layers is not a list.
        output_hidden_states (Optional[List[int]]): List of layers (indices) to include in the output

    Raises:
        AssertionError: num_layers is set and layer is a list
        AssertionError: num_layers is not set and layer is an nn.Module

    Note:
        Arg values are checked for correctness (eg: ``attn_dropout`` belongs to [0,1])
        in the module where they are used. This helps reduces the number of raise
        statements in code and improves readability.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        num_layers: int,
        rope_base: int,
        embed_dim: int,
        num_kv_heads: int,
        scale_factor: int,
        intermediate_dim: int,
        attn_dropout: float,
        max_seq_len: int,
        num_heads: int,
    ) -> None:
        super().__init__()

        self.max_seq_len = max_seq_len
        self.num_heads = num_heads

        self.tok_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.norm = RMSNorm(embed_dim)
        self.output = TiedLinear(self.tok_embeddings)
        self.head_dim = embed_dim // num_heads
        rope = Llama3ScaledRoPE(
            dim=self.head_dim,
            max_seq_len=max_seq_len,
            base=rope_base,
            scale_factor=scale_factor,
        )

        self.layers = []
        for _ in range(num_layers):
            self_attn = MultiHeadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=self.head_dim,
                q_proj=nn.Linear(embed_dim, num_heads * self.head_dim, bias=False),
                k_proj=nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=False),
                v_proj=nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=False),
                output_proj=nn.Linear(embed_dim, embed_dim, bias=False),
                pos_embeddings=rope,
                max_seq_len=max_seq_len,
                attn_dropout=attn_dropout,
            )
            layer = TransformerSelfAttentionLayer(
                attn=self_attn,
                mlp=FeedForward(dim=embed_dim, hidden_dim=intermediate_dim),
                sa_norm=RMSNorm(dim=embed_dim),
                mlp_norm=RMSNorm(dim=embed_dim),
            )
            self.layers.append(layer)
        self.layers = nn.ModuleList(self.layers)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
        encoder_input: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            tokens (torch.Tensor): input tensor with shape ``[b x s]``
            mask (Optional[torch.Tensor]): Used to mask the scores after the query-key multiplication
                and before the softmax. This parameter is required during inference if caches have been setup.
                Either:

                A boolean tensor with shape ``[b x s x s]``, ``[b x s x self.encoder_max_cache_seq_len]``,
                or ``[b x s x self.encoder_max_cache_seq_len]`` if using KV-cacheing with encoder/decoder layers.
                A value of True in row ``i`` and column ``j`` means token ``i`` attends to token ``j``. A value of False means
                token ``i`` does not attend to token ``j``. If no mask is specified, a causal mask
                is used by default.

                A :class:`~torch.nn.attention.flex_attention.BlockMask` for document masking in a packed sequence
                created via `create_block_mask <https://pytorch.org/blog/flexattention/#mask-mods>`_. We  use
                :func:`~torch.nn.attention.flex_attention.flex_attention` when computing attention with block masks.
                Default is None.
            encoder_input (Optional[torch.Tensor]): Optional input embeds from the encoder. Shape ``[b x s_e x d_e]``
            encoder_mask (Optional[torch.Tensor]):  Boolean tensor defining a relational matrix between
                tokens and encoder embeddings. A True value at position ``i,j`` means token ``i`` can attend
                to embedding ``j`` in the decoder. Mask has shape ``[b x s x s_e]``. Default is None,
                but this is required during inference if the model has been setup with any layers
                which use encoder embeddings and caches have been setup.
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape ``[b x s]``.
                During inference, this indicates the position of the current token.
                This parameter is required during inference if caches have been setup. Default is None.

        Returns:
            Union[torch.Tensor, List[torch.Tensor]]: output tensor with shape ``[b x s x v]`` or a list of layer
                output tensors defined by ``output_hidden_states`` with the
                final output tensor appended to the list.

        Note:
            At the very first step of inference, when the model is provided with a prompt,
            ``input_pos`` should contain the positions of all of the tokens in the prompt.
            For a single-batch prompt, or a batch of prompts with identical lengths, this
            will be ``torch.arange(prompt_length)``. For a batch of varying-length prompts,
            shorter prompts are left-padded and position ids are correspondingly right-shifted,
            thus positional ids should be of shape ``[b, padded_prompt_length]``.
            This is because we will need to retrieve the positional embeddings for each input id.
            In the subsequent steps, if the model has been setup with KV-caches, ``input_pos`` will contain
            the position(s) of the current token(s) ``torch.tensor([padded_prompt_length])``. Otherwise,
            ``input_pos`` will contain all the position ids up to the current token.

        Shape notation:
            - b: batch size
            - s: token sequence length
            - s_e: encoder sequence length
            - v: vocab size
            - d: token embed dim
            - d_e: encoder embed dim
            - m_s: max seq len
        """
        # input tensor of shape [b, s]

        if tokens.shape[1] > self.max_seq_len:
            raise ValueError(
                f"seq_len ({tokens.shape[1]}) of input tensor should be smaller "
                f"than max_seq_len ({self.max_seq_len})"
            )

        # shape: [b, s, d]
        h = self.tok_embeddings(tokens)

        for layer in self.layers:
            # shape: [b, s, d]
            h = layer(
                h,
                mask=mask,
                encoder_input=encoder_input,
                encoder_mask=encoder_mask,
                input_pos=input_pos,
            )

        # shape: [b, s, d]
        h = self.norm(h)

        # shape: [b, seq_len, out_dim]
        output = self.output(h).float()

        return output


def llama3_2_1b() -> TransformerDecoder:
    return TransformerDecoder(
        vocab_size=128_256,
        num_layers=16,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=2048,
        max_seq_len=131072,
        intermediate_dim=8192,
        attn_dropout=0.0,
        rope_base=500_000,
        scale_factor=32,
    )
