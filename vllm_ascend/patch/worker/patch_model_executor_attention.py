from vllm.config import VllmConfig
from vllm.v1.attention.backend import AttentionType
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    SlidingWindowSpec,
    get_kv_quant_mode,
)
from vllm.model_executor.layers.attention.attention import Attention


def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
    # Block size may get updated after model loading, refresh it
    block_size = vllm_config.cache_config.block_size
    # Should not be called for enc-dec or encoder-only attention.
    assert self.attn_type == AttentionType.DECODER
    quant_mode = get_kv_quant_mode(self.kv_cache_dtype)
    if self.sliding_window is not None:
        assert not vllm_config.model_config.use_mla, (
            "MLA is not supported for slidingwindow"
        )
        return SlidingWindowSpec(
            block_size=block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            head_size_v=self.head_size_v,
            dtype=self.kv_cache_torch_dtype,
            kv_quant_mode=quant_mode,
            sliding_window=self.sliding_window,
        )
    elif self.kv_cache_dtype.startswith("turboquant_"):
        from vllm.model_executor.layers.quantization.turboquant.config import (
            TurboQuantConfig,
        )
        from vllm.v1.kv_cache_interface import TQFullAttentionSpec

        tq_config = TurboQuantConfig.from_cache_dtype(
            self.kv_cache_dtype, self.head_size
        )
        return TQFullAttentionSpec(
            block_size=block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            head_size_v=self.head_size,
            dtype=self.kv_cache_torch_dtype,
            tq_slot_size=tq_config.slot_size_aligned,
        )
    else:
        # num_spec_tokens is used to optimize the benefit conflict
        # between APC and MTP for full attention find_longest_cache_hit.
        if (
            vllm_config.speculative_config is not None
            and vllm_config.speculative_config.num_speculative_tokens
        ):
            num_spec_tokens = vllm_config.speculative_config.num_speculative_tokens
        else:
            num_spec_tokens = 0
        return FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            head_size_v=self.head_size_v,
            dtype=self.kv_cache_torch_dtype,
            kv_quant_mode=quant_mode,
            # Propagate MTP lookahead for prefix-cache hit selection.
            num_spec_tokens=num_spec_tokens,
        )


Attention.get_kv_cache_spec = get_kv_cache_spec
