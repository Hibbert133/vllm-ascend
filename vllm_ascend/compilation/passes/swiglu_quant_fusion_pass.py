# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import operator

import torch
from torch._inductor.pattern_matcher import PatternMatcherPass
from vllm.compilation.passes.vllm_inductor_pass import VllmInductorPass
from vllm.config import VllmConfig
from vllm.config.compilation import Range
from vllm.logger import logger

from vllm_ascend.compilation.passes.base_pattern import BasePattern
from vllm_ascend.utils import enable_custom_op


class QuantMatmulSwiGLUDynamicQuantPattern(BasePattern):
    """Fuse the W8A8 gate/up projection epilogue for dense MLPs.

    Keep the gate/up matmul accumulator in INT32 and let
    ``npu_dequant_swiglu_quant`` perform dequantization, SwiGLU and dynamic
    INT8 quantization without materializing the intermediate BF16 tensors.
    """

    def get_inputs(self):
        quantized_x = torch.randint(-8, 8, (4, 64), device="npu", dtype=torch.int8)
        weight = torch.randint(-8, 8, (64, 128), device="npu", dtype=torch.int8)
        weight_scale = torch.rand(128, device="npu", dtype=self.dtype)
        per_token_scale = torch.rand(4, device="npu", dtype=torch.float32)
        return [quantized_x, weight, weight_scale, per_token_scale]

    def get_pattern(self):
        def pattern(
            quantized_x: torch.Tensor,
            weight: torch.Tensor,
            weight_scale: torch.Tensor,
            per_token_scale: torch.Tensor,
        ):
            gate_up = torch.ops.npu.npu_quant_matmul(
                quantized_x,
                weight,
                weight_scale,
                pertoken_scale=per_token_scale,
                bias=None,
                output_dtype=self.dtype,
            )
            activated = torch.ops.npu.npu_swiglu(gate_up)
            return torch.ops.npu.npu_dynamic_quant(activated)

        return pattern

    def get_replacement(self):
        def replacement(
            quantized_x: torch.Tensor,
            weight: torch.Tensor,
            weight_scale: torch.Tensor,
            per_token_scale: torch.Tensor,
        ):
            gate_up_int32 = torch.ops.npu.npu_quant_matmul(
                quantized_x,
                weight,
                weight_scale,
                pertoken_scale=None,
                bias=None,
                output_dtype=torch.int32,
            )
            return torch.ops._C_ascend.npu_dequant_swiglu_quant(
                x=gate_up_int32,
                weight_scale=weight_scale.to(torch.float32),
                activation_scale=per_token_scale,
                bias=None,
                quant_scale=None,
                quant_offset=None,
                group_index=None,
                activate_left=True,
                quant_mode=1,
                swiglu_mode=1,
                clamp_limit=0.0,
                glu_alpha=1.0,
                glu_bias=0.0,
            )

        return replacement


class SwiGLUQuantFusionPass(VllmInductorPass):
    """Fuse dense W8A8 QuantMatmul + SwiGLU + DynamicQuant graphs."""

    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)
        self.pattern_match_passes = PatternMatcherPass(pass_name="swiglu_quant_fusion_pass")

        self.dtype = vllm_config.model_config.dtype
        self.enabled = self.dtype in (torch.bfloat16, torch.float16)
        if not self.enabled:
            logger.debug("SwiGLU quant fusion not enabled: unsupported dtype %s", self.dtype)
            return
        if not enable_custom_op():
            logger.debug("SwiGLU quant fusion not enabled: custom ops unavailable")
            self.enabled = False
            return

        QuantMatmulSwiGLUDynamicQuantPattern(vllm_config).register(self.pattern_match_passes)

    def _fuse_matmul_reduce(self, graph: torch.fx.Graph) -> int:
        def is_target(node: torch.fx.Node, op: torch._ops.OpOverload) -> bool:
            target = node.target
            return target == op or getattr(target, "overloadpacket", None) == op.overloadpacket

        matched_count = 0
        for reduce_node in list(graph.nodes):
            if reduce_node.op != "call_function" or not is_target(
                reduce_node, torch.ops.vllm.matmul_and_reduce.default
            ):
                continue
            if len(reduce_node.args) != 2 or not isinstance(reduce_node.args[0], torch.fx.Node):
                continue
            swiglu_node = reduce_node.args[0]
            if not is_target(swiglu_node, torch.ops.npu.npu_swiglu.default) or len(swiglu_node.users) != 1:
                continue
            if not swiglu_node.args or not isinstance(swiglu_node.args[0], torch.fx.Node):
                continue
            quant_matmul_node = swiglu_node.args[0]
            if (
                not is_target(quant_matmul_node, torch.ops.npu.npu_quant_matmul.default)
                or len(quant_matmul_node.users) != 1
                or len(quant_matmul_node.args) < 3
            ):
                continue
            pertoken_scale = quant_matmul_node.kwargs.get("pertoken_scale")
            output_dtype = quant_matmul_node.kwargs.get("output_dtype")
            expected_output_dtypes = {
                torch.float16: (torch.float16, 5),
                torch.bfloat16: (torch.bfloat16, 15),
            }[self.dtype]
            if (
                not isinstance(pertoken_scale, torch.fx.Node)
                or quant_matmul_node.kwargs.get("bias") is not None
                or output_dtype not in expected_output_dtypes
            ):
                continue

            weight_scale = quant_matmul_node.args[2]
            with graph.inserting_before(reduce_node):
                int32_kwargs = dict(quant_matmul_node.kwargs)
                int32_kwargs["pertoken_scale"] = None
                int32_kwargs["output_dtype"] = torch.int32
                gate_up_int32 = graph.call_function(
                    torch.ops.npu.npu_quant_matmul.default,
                    args=quant_matmul_node.args,
                    kwargs=int32_kwargs,
                )
                weight_scale_fp32 = graph.call_method("to", args=(weight_scale, torch.float32))
                fused = graph.call_function(
                    torch.ops._C_ascend.npu_dequant_swiglu_quant.default,
                    kwargs={
                        "x": gate_up_int32,
                        "weight_scale": weight_scale_fp32,
                        "activation_scale": pertoken_scale,
                        "bias": None,
                        "quant_scale": None,
                        "quant_offset": None,
                        "group_index": None,
                        "activate_left": True,
                        "quant_mode": 1,
                        "swiglu_mode": 1,
                        "clamp_limit": 0.0,
                        "glu_alpha": 1.0,
                        "glu_bias": 0.0,
                    },
                )
                activated = graph.call_function(operator.getitem, args=(fused, 0))
                activated_scale = graph.call_function(operator.getitem, args=(fused, 1))
                replacement = graph.call_function(
                    torch.ops.vllm.quantized_matmul_and_reduce.default,
                    args=(activated, activated_scale, reduce_node.args[1], self.dtype),
                )

            if "val" in quant_matmul_node.meta:
                gate_up_int32.meta["val"] = quant_matmul_node.meta["val"].to(torch.int32)
            if isinstance(weight_scale, torch.fx.Node) and "val" in weight_scale.meta:
                weight_scale_fp32.meta["val"] = weight_scale.meta["val"].to(torch.float32)
            if "val" in swiglu_node.meta and "val" in pertoken_scale.meta:
                activated.meta["val"] = swiglu_node.meta["val"].to(torch.int8)
                activated_scale.meta["val"] = pertoken_scale.meta["val"]
                fused.meta["val"] = (activated.meta["val"], activated_scale.meta["val"])
            replacement.meta = dict(reduce_node.meta)
            reduce_node.replace_all_uses_with(replacement)
            graph.erase_node(reduce_node)
            graph.erase_node(swiglu_node)
            graph.erase_node(quant_matmul_node)
            matched_count += 1
        if matched_count:
            graph.lint()
        return matched_count

    def __call__(self, graph: torch.fx.Graph):
        self.begin()
        self.matched_count = 0
        if self.enabled:
            self.matched_count = self.pattern_match_passes.apply(graph)
            self.matched_count += self._fuse_matmul_reduce(graph)
        logger.debug("Replaced %s SwiGLU quant patterns", self.matched_count)
        self.end_and_log()

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        return True
