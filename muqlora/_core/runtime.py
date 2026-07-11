import torch


MUQ_MEL_INPUT_CONFIG = {
    "sample_rate": 24000,
    "n_fft": 2048,
    "hop_length": 240,
    "n_mels": 128,
    "is_db": True,
}


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

    features = muq_model.normalize(features)
    mel_features = features["melspec_2048"].to(dtype=wrapper.frontend_dtype)
    hidden_states = muq_model.conv(mel_features).to(dtype=wrapper.base_dtype)

    if attention_mask is not None:
        if input_type == "mel":
            attention_mask = attention_mask[..., :-1]
        attention_mask = attention_mask.bool()
        skip_n = int(attention_mask.size(-1) / hidden_states.size(1))
        if skip_n <= 0:
            raise ValueError("attention_mask is shorter than the encoded sequence")
        attention_mask = attention_mask[:, ::skip_n]
        attention_mask = attention_mask[:, : hidden_states.size(1)]

    return hidden_states, attention_mask


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
