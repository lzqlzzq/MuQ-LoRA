import math

import torch
from torch import nn


class YaRNRotaryPositionalEmbedding(nn.Module):
    """YaRN-scaled rotary positional embedding for MuQ's Conformer RoPE path.

    MuQ's rotary attention applies RoPE before the Q/K projections and expects
    ``embed_positions(hidden_states)`` to return a stacked ``[cos, sin]`` tensor
    shaped ``[2, sequence, 1, 1, head_dim]``.
    """

    def __init__(
        self,
        config,
        factor: float,
        original_max_position_embeddings: int | None = None,
        attention_factor: float | None = None,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        mscale: float | None = None,
        mscale_all_dim: float | None = None,
        truncate: bool = True,
    ):
        super().__init__()

        if factor <= 0:
            raise ValueError("yarn_factor must be positive")
        if beta_fast <= 0 or beta_slow <= 0:
            raise ValueError("yarn_beta_fast and yarn_beta_slow must be positive")

        dim = config.hidden_size // config.num_attention_heads
        base = config.rotary_embedding_base
        if original_max_position_embeddings is None:
            original_max_position_embeddings = getattr(
                config,
                "max_position_embeddings",
                getattr(config, "max_source_positions", None),
            )
        if original_max_position_embeddings is None:
            raise ValueError(
                "yarn_original_max_position_embeddings is required when the MuQ config "
                "does not define max_position_embeddings or max_source_positions"
            )

        inv_freq = self._compute_yarn_inv_freq(
            dim=dim,
            base=base,
            factor=factor,
            original_max_position_embeddings=original_max_position_embeddings,
            beta_fast=beta_fast,
            beta_slow=beta_slow,
            truncate=truncate,
        )
        self.register_buffer("inv_freq", inv_freq)
        self.attention_factor = self._resolve_attention_factor(
            factor=factor,
            attention_factor=attention_factor,
            mscale=mscale,
            mscale_all_dim=mscale_all_dim,
        )
        self.cached_sequence_length = None
        self.cached_dtype = None
        self.cached_device = None
        self.cached_rotary_positional_embedding = None

    @staticmethod
    def _resolve_attention_factor(
        factor: float,
        attention_factor: float | None,
        mscale: float | None,
        mscale_all_dim: float | None,
    ) -> float:
        if attention_factor is not None:
            return attention_factor

        def get_mscale(scale: float, multiplier: float = 1.0) -> float:
            if scale <= 1:
                return 1.0
            return 0.1 * multiplier * math.log(scale) + 1.0

        if mscale is not None and mscale_all_dim is not None:
            return get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim)
        return get_mscale(factor)

    @staticmethod
    def _find_correction_dim(
        num_rotations: float,
        dim: int,
        base: float,
        original_max_position_embeddings: int,
    ) -> float:
        return (
            dim
            * math.log(original_max_position_embeddings / (num_rotations * 2 * math.pi))
            / (2 * math.log(base))
        )

    @classmethod
    def _find_correction_range(
        cls,
        low_rot: float,
        high_rot: float,
        dim: int,
        base: float,
        original_max_position_embeddings: int,
        truncate: bool,
    ) -> tuple[float, float]:
        low = cls._find_correction_dim(
            low_rot,
            dim,
            base,
            original_max_position_embeddings,
        )
        high = cls._find_correction_dim(
            high_rot,
            dim,
            base,
            original_max_position_embeddings,
        )
        if truncate:
            low = math.floor(low)
            high = math.ceil(high)
        return max(low, 0), min(high, dim - 1)

    @staticmethod
    def _linear_ramp_factor(low: float, high: float, dim: int) -> torch.Tensor:
        if low == high:
            high += 0.001
        ramp = (torch.arange(dim, dtype=torch.float32) - low) / (high - low)
        return torch.clamp(ramp, 0, 1)

    @classmethod
    def _compute_yarn_inv_freq(
        cls,
        dim: int,
        base: float,
        factor: float,
        original_max_position_embeddings: int,
        beta_fast: float,
        beta_slow: float,
        truncate: bool,
    ) -> torch.Tensor:
        position_frequencies = base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        inv_freq_extrapolation = 1.0 / position_frequencies
        inv_freq_interpolation = 1.0 / (factor * position_frequencies)

        low, high = cls._find_correction_range(
            beta_fast,
            beta_slow,
            dim,
            base,
            original_max_position_embeddings,
            truncate,
        )
        extrapolation_factor = 1 - cls._linear_ramp_factor(low, high, dim // 2)
        return (
            inv_freq_interpolation * (1 - extrapolation_factor)
            + inv_freq_extrapolation * extrapolation_factor
        )

    def forward(self, hidden_states):
        sequence_length = hidden_states.shape[1]
        if (
            sequence_length == self.cached_sequence_length
            and hidden_states.dtype == self.cached_dtype
            and hidden_states.device == self.cached_device
            and self.cached_rotary_positional_embedding is not None
        ):
            return self.cached_rotary_positional_embedding

        self.cached_sequence_length = sequence_length
        self.cached_dtype = hidden_states.dtype
        self.cached_device = hidden_states.device
        inv_freq = self.inv_freq.to(device=hidden_states.device, dtype=hidden_states.dtype)
        time_stamps = torch.arange(sequence_length, device=hidden_states.device, dtype=hidden_states.dtype)
        freqs = torch.einsum("i,j->ij", time_stamps, inv_freq)
        embeddings = torch.cat((freqs, freqs), dim=-1)

        cos_embeddings = embeddings.cos()[:, None, None, :] * self.attention_factor
        sin_embeddings = embeddings.sin()[:, None, None, :] * self.attention_factor
        self.cached_rotary_positional_embedding = torch.stack([cos_embeddings, sin_embeddings])
        return self.cached_rotary_positional_embedding
