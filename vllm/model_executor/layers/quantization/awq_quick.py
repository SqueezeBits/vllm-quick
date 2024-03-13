from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.nn.parameter import Parameter

from vllm._C import ops
from vllm.model_executor.layers.linear import (LinearMethodBase,
                                               set_weight_attrs)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

def QUICK_cat(input_layers: List[torch.Tensor], options: str, reshape_dims: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    """
    Concatenates multiple input layers after reshaping based on specified options for QUICK.
    Args:
        input_layers: Variable number of tensor layers to concatenate.
        options: A string indicating how the layers should be reshaped ('qweight', 'qzeros', or 'scales').
        reshape_dims: Optional tuple indicating custom dimensions for reshaping. If None, default settings are used.
    Returns:
        torch.Tensor: The concatenated and reshaped layers.
    Raises:
        ValueError: If the options provided are invalid or if input layers have incompatible shapes.
    """
    # Check if there are at least two layers to concatenate
    if len(input_layers) < 2:
        raise ValueError("At least two input layers are required")

    shapes = [layer.shape for layer in input_layers]

    # Check for shape compatibility on H
    OH, _ = input_layers[0].shape
    for layer in input_layers[1:]:
        if layer.shape[0] != OH:
            raise ValueError("All quick layers to concat must have the same height")

    # Determine reshape dimensions based on options
    if not reshape_dims:
        reshape_dims = [{
                'qweight': (H // 2, W * 2),
                'qzeros': (H * 4, W // 4),
                'scales': (H * 4, W // 4)
            }.get(options) for [H, W] in shapes
        ]

    if reshape_dims is None:
        raise ValueError("Unknown options provided or invalid reshape dimensions")

    # Reshape and concatenate the input layers
    if len(reshape_dims) > 1:
        layers_to_cat = [layer.reshape(*reshape_dim) for layer, reshape_dim in zip(input_layers, reshape_dims)]
    else:
        layers_to_cat = [layer.reshape(*reshape_dims) for layer in input_layers]
    output_layer = torch.cat(layers_to_cat, dim=1).reshape(OH, -1)

    return output_layer


class QUICKAWQConfig(QuantizationConfig):
    """Config class for QUICK-AWQ.
    """

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        zero_point: bool,
    ) -> None:
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.zero_point = zero_point

        if self.weight_bits != 4:
            raise ValueError(
                "Currently, only 4-bit weight quantization is supported for "
                f"AWQ, but got {self.weight_bits} bits.")
        self.pack_factor = 32 // self.weight_bits

    def __repr__(self) -> str:
        return (f"QuickAWQConfig(weight_bits={self.weight_bits}, "
                f"group_size={self.group_size}, "
                f"zero_point={self.zero_point})")

    def get_name(self) -> str:
        return "quick-awq"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.half]

    def get_min_capability(self) -> int:
        # The AWQ kernel only supports Turing or newer GPUs.
        return 75

    @staticmethod
    def get_config_filenames() -> List[str]:
        return [
            "quant_config.json",  # E.g., casperhansen/vicuna-7b-v1.5-awq
            "quantize_config.json",  # E.g., abhinavkulkarni/mosaicml-mpt-7b-instruct-w4-g128-awq
        ]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "QUICKAWQConfig":
        weight_bits = cls.get_from_keys(config, ["w_bit", "bits"])
        group_size = cls.get_from_keys(config, ["q_group_size", "group_size"])
        zero_point = cls.get_from_keys(config, ["zero_point"])
        return cls(weight_bits, group_size, zero_point)

    def get_linear_method(self) -> "QUICKAWQLinearMethod":
        return QUICKAWQLinearMethod(self)

    def get_scaled_act_names(self) -> List[str]:
        return ["gelu", "gelu_fast", "gelu_new", "gelu_pytorch_tanh"]


class QUICKAWQLinearMethod(LinearMethodBase):
    """Linear method for QUICK-AWQ.

    Args:
        quant_config: The QUICK-AWQ quantization config.
    """

    def __init__(self, quant_config: QUICKAWQConfig):
        self.quant_config = quant_config
        self.large = False

    def create_weights(self, input_size_per_partition: int,
                       output_size_per_partition: int, input_size: int,
                       output_size: int,
                       params_dtype: torch.dtype) -> Dict[str, Any]:
        if input_size_per_partition % self.quant_config.group_size != 0:
            raise ValueError(
                "The input size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size.")
        if output_size_per_partition % self.quant_config.pack_factor != 0:
            raise ValueError(
                "The output size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size.")
        if input_size_per_partition < output_size_per_partition:
            self.large = True

        qweight = Parameter(
            torch.empty(
                input_size_per_partition // 4,
                output_size_per_partition // self.quant_config.pack_factor * 4,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qweight, {
                "input_dim": 0,
                "output_dim": 1,
                "packed_dim": 1,
                "pack_factor": self.quant_config.pack_factor // 4, # Interleaved along dimension 1
            })
        qzeros = Parameter(
            torch.empty(
                input_size_per_partition // self.quant_config.group_size,
                output_size_per_partition * 2 // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qzeros, {
                "input_dim": 0,
                "output_dim": 1,
                "packed_dim": 1,
                "pack_factor": self.quant_config.pack_factor // 2, # Doubled
            })
        scales = Parameter(
            torch.empty(
                input_size_per_partition // self.quant_config.group_size,
                output_size_per_partition * 2,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(scales, {
            "input_dim": 0,
            "output_dim": None
        })
        return {
            "qweight": qweight,
            "qzeros": qzeros,
            "scales": scales,
        }


    def apply_weights(self,
                      weights: Dict[str, Any],
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        qweight = weights["qweight"]
        scales = weights["scales"]
        qzeros = weights["qzeros"]
        out_shape = (x.shape[:-1] + (scales.shape[-1] // 2, ))
        reshaped_x = x.reshape(-1, x.shape[-1])

        batch = reshaped_x.shape[0]
        split_k = 4
        if self.large:
            if batch > 512:
                split_k = 1
            elif batch > 256:
                split_k = 2
        else:
            if batch > 256:
                split_k = 2
        out = ops.awq_quick_gemm(reshaped_x, qweight, scales, qzeros, split_k)

        if bias is not None:
            out = out + bias
        return out.reshape(out_shape)
