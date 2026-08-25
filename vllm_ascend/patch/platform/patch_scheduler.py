from vllm.logger import logger
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request

from vllm_ascend import envs as ascend_envs


def _mamba_block_aligned_split(
    self,
    request: Request,
    num_new_tokens: int,
    num_new_local_computed_tokens: int = 0,
    num_external_computed_tokens: int = 0,
) -> int:
    num_computed_tokens = (
        request.num_computed_tokens
        + num_new_local_computed_tokens
        + num_external_computed_tokens
    )
    if num_computed_tokens < max(request.num_prompt_tokens, request.num_tokens - 1):
        # To enable block-aligned caching of the Mamba state, `num_new_tokens`
        # must be a multiple of `block_size`.
        block_size = self.cache_config.block_size
        last_cache_position = request.num_tokens - request.num_tokens % block_size
        align_mamba_prefix_caching_length = ascend_envs.ALIGN_MAMBA_PREFIX_CACHING_LENGTH
        if align_mamba_prefix_caching_length > 0:
            # Cap the cacheable prefix at the configured length (block-aligned).
            # Not applied under Eagle: the block subtracted below must remain
            # available so the last chunk stays >= block_size, otherwise no
            # Mamba state snapshot is ever taken and prefix caching silently
            # degrades to 0% hit rate.
            custom_prefix_caching_length = (
                align_mamba_prefix_caching_length
                - align_mamba_prefix_caching_length % block_size
            )

            if num_new_tokens > custom_prefix_caching_length:
                if (
                    request.num_tokens > custom_prefix_caching_length
                    and num_computed_tokens == 0
                ):
                    last_cache_position = custom_prefix_caching_length
                else:
                    last_cache_position = num_computed_tokens
            else:
                last_cache_position = num_computed_tokens
            if self.use_eagle:
                last_cache_position += block_size
        # adapt end
        # eagle prune
        if self.use_eagle:
            last_cache_position = max(last_cache_position - block_size, 0)
        num_computed_tokens_after_sched = num_computed_tokens + num_new_tokens
        if num_computed_tokens_after_sched < last_cache_position:
            # align to block_size
            num_new_tokens = num_new_tokens // block_size * block_size
        elif (
            num_computed_tokens
            < last_cache_position
            < num_computed_tokens_after_sched
        ):
            # force to cache the last chunk
            num_new_tokens = last_cache_position - num_computed_tokens
        else:
            # prefill the last few tokens
            pass
    return num_new_tokens


Scheduler._mamba_block_aligned_split = _mamba_block_aligned_split

logger.info(
    "Mamba align APC patch applied: Scheduler._mamba_block_aligned_split "
    "(ALIGN_MAMBA_PREFIX_CACHING_LENGTH=%d)",
    ascend_envs.ALIGN_MAMBA_PREFIX_CACHING_LENGTH,
)
