"""Utility functions for 4-bit NF4 quantization via bitsandbytes."""

import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)


def _should_quantize_module(
    name: str, module: nn.Linear, skip_patterns: list[str]
) -> bool:
    """Determine if a given nn.Linear should be replaced with Linear4bit."""
    for pattern in skip_patterns:
        if pattern in name:
            return False
    return True


def replace_linear_with_4bit(
    model: nn.Module,
    skip_patterns: list[str] | None = None,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    """Recursively replace nn.Linear modules with bnb.nn.Linear4bit (NF4).

    Weights from existing nn.Linear modules are copied into the new
    Linear4bit as ``Params4bit`` tensors.  After calling this function,
    move the model to CUDA via ``model.to(device)`` to trigger the
    actual NF4 quantization.

    Args:
        model: The model to quantize.  Weights should already be loaded.
        skip_patterns: List of name patterns to skip (e.g., ["depformer"]).
        compute_dtype: dtype for the dequantized compute operations.

    Returns:
        The model with nn.Linear modules replaced by Linear4bit.
    """
    try:
        import bitsandbytes as bnb
    except ImportError:
        raise ImportError(
            "4-bit quantization requires the 'bitsandbytes' package. "
            "Install it with: pip install bitsandbytes>=0.43"
        )

    if skip_patterns is None:
        skip_patterns = []

    replaced_count = 0
    skipped_count = 0

    def _replace_recursive(parent: nn.Module, prefix: str = ""):
        nonlocal replaced_count, skipped_count
        for name, child in list(parent.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear):
                if _should_quantize_module(full_name, child, skip_patterns):
                    new_module = bnb.nn.Linear4bit(
                        child.in_features,
                        child.out_features,
                        bias=child.bias is not None,
                        compute_dtype=compute_dtype,
                        quant_type="nf4",
                        compress_statistics=True,
                    )
                    # Copy the loaded weight data as Params4bit so that
                    # a subsequent .to(device) triggers NF4 quantization.
                    new_module.weight = bnb.nn.Params4bit(
                        child.weight.data,
                        requires_grad=False,
                        quant_type="nf4",
                        compress_statistics=True,
                        blocksize=64,
                    )
                    if child.bias is not None:
                        new_module.bias = nn.Parameter(child.bias.data)
                    setattr(parent, name, new_module)
                    replaced_count += 1
                    logger.debug("Quantized %s (%d x %d)", full_name,
                                 child.in_features, child.out_features)
                else:
                    skipped_count += 1
                    logger.debug("Skipped %s", full_name)
            else:
                _replace_recursive(child, full_name)

    _replace_recursive(model)
    logger.info(
        "Quantization: replaced %d Linear modules with Linear4bit, skipped %d",
        replaced_count,
        skipped_count,
    )
    return model


def remap_state_dict_keys(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Remap old state_dict keys for the in_proj_weight -> in_proj.weight refactor.

    Only remaps keys for the main transformer attention (not depformer), since
    the depformer retains ``in_proj_weight`` as a raw Parameter.
    """
    remapped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        if "depformer" not in key:
            if "in_proj_weight" in key:
                new_key = key.replace("in_proj_weight", "in_proj.weight")
            elif "in_proj_bias" in key:
                new_key = key.replace("in_proj_bias", "in_proj.bias")
        remapped[new_key] = value
    return remapped
