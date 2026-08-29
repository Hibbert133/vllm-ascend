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

from pathlib import Path

import torch
import torch.nn as nn
import vllm.config
from vllm.config import ModelConfig, VllmConfig

import vllm_ascend.ops.register_custom_ops  # noqa: F401
from vllm_ascend.compilation.passes.swiglu_quant_fusion_pass import SwiGLUQuantFusionPass

from .backend import TestBackend


class DenseW8A8MLPEpilogue(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randint(-8, 8, (hidden_size, intermediate_size * 2), device="npu", dtype=torch.int8),
            requires_grad=False,
        )
        self.weight_scale = nn.Parameter(
            torch.rand(intermediate_size * 2, device="npu", dtype=dtype),
            requires_grad=False,
        )

    def forward(self, quantized_x: torch.Tensor, per_token_scale: torch.Tensor):
        gate_up = torch.ops.npu.npu_quant_matmul(
            quantized_x,
            self.weight,
            self.weight_scale,
            pertoken_scale=per_token_scale,
            bias=None,
            output_dtype=torch.bfloat16,
        )
        activated = torch.ops.npu.npu_swiglu(gate_up)
        return torch.ops.npu.npu_dynamic_quant(activated)


def test_dense_w8a8_swiglu_dynamic_quant_fusion():
    dtype = torch.bfloat16
    fake_model_path = Path(__file__).resolve().parents[5] / "tests/ut/_fake_weight"
    vllm_config = VllmConfig(
        model_config=ModelConfig(
            model=str(fake_model_path),
            dtype=dtype,
        ),
        additional_config={"ascend_log_path": "/tmp/vllm_ascend_test_logs"},
    )

    with vllm.config.set_current_vllm_config(vllm_config):
        backend = TestBackend(custom_passes=[SwiGLUQuantFusionPass(vllm_config)])
        model = DenseW8A8MLPEpilogue(64, 64, dtype).to("npu")
        quantized_x = torch.randint(-8, 8, (4, 64), device="npu", dtype=torch.int8)
        per_token_scale = torch.rand(4, device="npu", dtype=torch.float32)

        expected = model(quantized_x, per_token_scale)
        actual = torch.compile(model, backend=backend)(quantized_x, per_token_scale)

        torch.testing.assert_close(actual[0], expected[0], atol=1, rtol=0.1)
        torch.testing.assert_close(actual[1], expected[1], atol=1e-4, rtol=5e-3)
        backend.check_before_ops(
            [torch.ops.npu.npu_quant_matmul.default], fully_replaced=False
        )
        backend.check_before_ops(
            [
                torch.ops.npu.npu_swiglu.default,
                torch.ops.npu.npu_dynamic_quant.default,
            ]
        )
        backend.check_after_ops(
            [
                torch.ops.npu.npu_quant_matmul.default,
                torch.ops._C_ascend.npu_dequant_swiglu_quant.default,
            ]
        )
