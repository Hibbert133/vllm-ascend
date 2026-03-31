import torch
import torch.distributed as dist
from vllm.distributed.parallel_state import GroupCoordinator, get_dp_group
from vllm.forward_context import get_forward_context

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.distributed.parallel_state import get_fc3_quant_x_group

import torch_npu
from vllm.config import get_current_vllm_config
from vllm.distributed import tensor_model_parallel_all_gather

def fc3_all_gather_and_maybe_unpad_impl(
    x: torch.Tensor,
) -> torch.Tensor:
    try:
        forward_context = get_forward_context()
    except AssertionError:
        return x
    x = get_fc3_quant_x_group().all_gather(x, 0)
    dp_metadata = forward_context.dp_metadata
    if dp_metadata is None:
        pad_size = _EXTRA_CTX.pad_size
        if pad_size > 0:
            x = x[:-pad_size]
    else:
        # unpad
        num_tokens_across_dp_cpu = dp_metadata.num_tokens_across_dp_cpu
        result = torch.empty((num_tokens_across_dp_cpu.sum(), *x.shape[1:]), device=x.device, dtype=x.dtype)
        dp_size = get_dp_group().world_size
        x = x.view(dp_size, _EXTRA_CTX.padded_length, *x.shape[1:])
        offset = 0
        for idx in range(dp_size):
            num_tokens_dp = num_tokens_across_dp_cpu[idx]
            result[offset : offset + num_tokens_dp] = x[idx, :num_tokens_dp]
            offset += num_tokens_dp
        x = result

    return x


def all_gather_async(
    input: torch.Tensor, group: GroupCoordinator, output: torch.Tensor | None = None, async_op: bool = True
):
    if group.world_size == 1:
        return input, None
    if output is None:
        input_size = input.size()
        output_size = (input_size[0] * group.world_size,) + input_size[1:]
        output = torch.empty(output_size, dtype=input.dtype, device=input.device)
    return output, dist.all_gather_into_tensor(output, input, group=group.device_group, async_op=async_op)

def dict_tensor_model_parallel_all_gather(input: torch.Tensor, dim=-1):
    vllm_config = get_current_vllm_config()
    quant_type = getattr(vllm_config.model_config.hf_text_config, "moe_quantize", getattr(vllm_config.model_config.hf_text_config, "quantize", None))
    if quant_type != "w8a8_dynamic":
        return
    x_int8, pertoken_scale = torch_npu.npu_dynamic_quant(input)
    input_ = {"x_int8": x_int8, "pertoken_scale": pertoken_scale}
    if isinstance(input_, dict):
        for key in sorted(input_.keys()):
            value = input_[key]
            if isinstance(value, torch.Tensor):
                input_[key] = tensor_model_parallel_all_gather(value, dim=dim)
        input_["dtype_before_quant"]=input.dtype
        return input_