import torch


MUQ_MEL_INPUT_CONFIG = {
    "sample_rate": 24000,
    "n_fft": 2048,
    "hop_length": 240,
    "n_mels": 128,
    "is_db": True,
}


def _validate_attention_mask(
    attention_mask: torch.Tensor,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    expected_shape = (x.shape[0], x.shape[-1])
    if attention_mask.ndim != 2 or tuple(attention_mask.shape) != expected_shape:
        raise ValueError(
            "attention_mask must have shape "
            f"{expected_shape}, got {tuple(attention_mask.shape)}"
        )
    if not bool(torch.isfinite(attention_mask).all()):
        raise ValueError("attention_mask must contain only finite values")
    if not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
        raise ValueError("attention_mask must contain only 0 and 1")

    mask = attention_mask.to(device=x.device, dtype=torch.bool)
    valid_input_lengths = mask.sum(dim=-1, dtype=torch.long)
    expected_prefix = (
        torch.arange(mask.shape[-1], device=mask.device).unsqueeze(0)
        < valid_input_lengths.unsqueeze(1)
    )
    if not torch.equal(mask, expected_prefix):
        raise ValueError("attention_mask valid frames must form one leading prefix")
    return mask, valid_input_lengths


def _feature_lengths(
    wrapper,
    valid_input_lengths: torch.Tensor,
    *,
    input_type: str,
) -> torch.Tensor:
    if input_type == "waveform":
        return torch.div(
            valid_input_lengths,
            int(wrapper.model.model.hop_length),
            rounding_mode="floor",
        )
    if input_type == "mel":
        return (valid_input_lengths - 1).clamp_min(0)
    if input_type == "muq_mel":
        return valid_input_lengths
    raise ValueError(f"unsupported input_type: {input_type!r}")


def _prefix_mask(lengths: torch.Tensor, frame_count: int) -> torch.Tensor:
    return (
        torch.arange(frame_count, device=lengths.device).unsqueeze(0)
        < lengths.unsqueeze(1)
    )


def prepare_encoder_inputs(
    wrapper,
    x,
    attention_mask: torch.Tensor | None = None,
    input_type: str = "waveform",
):
    """Convert waveform or mel features to Conformer-ready frames."""
    muq_model = wrapper.model.model

    if input_type == "waveform":
        features = muq_model.preprocessing(x, features=["melspec_2048"])
    elif input_type == "mel":
        features = {"melspec_2048": x[..., :-1]}
    elif input_type == "muq_mel":
        features = {"melspec_2048": x}
    else:
        raise ValueError(f"unsupported input_type: {input_type!r}")

    feature_attention_mask = None
    feature_lengths = None
    if attention_mask is not None:
        _, valid_input_lengths = _validate_attention_mask(attention_mask, x)
        feature_lengths = _feature_lengths(
            wrapper,
            valid_input_lengths,
            input_type=input_type,
        )
        feature_attention_mask = _prefix_mask(
            feature_lengths,
            features["melspec_2048"].shape[-1],
        )

    features = muq_model.normalize(features)
    mel_features = features["melspec_2048"].to(dtype=wrapper.frontend_dtype)
    if feature_attention_mask is not None:
        mel_features = mel_features.masked_fill(
            ~feature_attention_mask.unsqueeze(1),
            0,
        )
    hidden_states = muq_model.conv(mel_features).to(dtype=wrapper.base_dtype)

    encoder_attention_mask = None
    if feature_lengths is not None:
        subsampling_factor = int(muq_model.n_fold)
        encoder_lengths = torch.div(
            feature_lengths + subsampling_factor - 1,
            subsampling_factor,
            rounding_mode="floor",
        ).clamp_max(hidden_states.shape[1])
        encoder_attention_mask = _prefix_mask(
            encoder_lengths,
            hidden_states.shape[1],
        )

    return hidden_states, encoder_attention_mask


def encode(
    wrapper,
    x,
    attention_mask: torch.Tensor | None = None,
    input_type: str = "waveform",
    output_attentions: bool = False,
    output_hidden_states: bool = True,
    return_dict: bool = True,
    return_attention_mask: bool = False,
):
    """Run the MuQ encoder without computing the original codebook logits."""
    hidden_states, encoder_attention_mask = prepare_encoder_inputs(
        wrapper,
        x,
        attention_mask,
        input_type=input_type,
    )
    outputs = wrapper.model.model.conformer(
        hidden_states,
        attention_mask=encoder_attention_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    if return_attention_mask:
        return outputs, encoder_attention_mask
    return outputs


def pool_hidden_states(
    wrapper,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pool encoder frames before task heads."""
    if wrapper.pooling is None or wrapper.pooling == "none":
        return hidden_states

    if wrapper.pooling == "cls":
        return hidden_states[:, 0]

    if wrapper.pooling == "mean":
        if attention_mask is None:
            return hidden_states.mean(dim=1)

        mask = attention_mask.to(device=hidden_states.device, dtype=hidden_states.dtype)
        mask = mask.unsqueeze(-1)
        denominator = mask.sum(dim=1).clamp_min(1.0)
        return (hidden_states * mask).sum(dim=1) / denominator

    raise ValueError(f"unsupported pooling mode: {wrapper.pooling!r}")
