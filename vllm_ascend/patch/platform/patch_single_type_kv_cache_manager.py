import itertools

from vllm.logger import logger
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    BlockHashList,
    KVCacheBlock,
)
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    KVCacheSpec,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
)


@classmethod
def find_longest_cache_hit(
    cls,
    block_hashes: BlockHashList,
    max_length: int,
    kv_cache_group_ids: list[int],
    block_pool: BlockPool,
    kv_cache_spec: KVCacheSpec,
    drop_eagle_block: bool,
    alignment_tokens: int,
    dcp_world_size: int = 1,
    pcp_world_size: int = 1,
) -> tuple[list[KVCacheBlock], ...]:
    assert isinstance(
        kv_cache_spec, FullAttentionSpec | ChunkedLocalAttentionSpec
    ), (
        "FullAttentionManager can only be used for full attention "
        "and chunked local attention groups"
    )
    computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(
        [] for _ in range(len(kv_cache_group_ids))
    )
    block_size = kv_cache_spec.block_size
    if dcp_world_size * pcp_world_size > 1:
        block_size *= dcp_world_size * pcp_world_size
    max_num_blocks = max_length // block_size
    for block_hash in itertools.islice(block_hashes, max_num_blocks):
        # block_hashes is a chain of block hashes. If a block hash is not
        # in the cached_block_hash_to_id, the following block hashes are
        # not computed yet for sure.
        if cached_block := block_pool.get_cached_block(
            block_hash, kv_cache_group_ids
        ):
            for computed, cached in zip(computed_blocks, cached_block):
                computed.append(cached)
        else:
            break
    # adapt begin
    if (
        drop_eagle_block
        and computed_blocks[0]
        and len(computed_blocks[0]) == max_num_blocks
    ):
        # When num_spec_tokens is set, only drop the last matched block if
        # the remainder is small enough that the spec tokens would not
        # overlap a new block. This optimizes the conflict between APC
        # (Automatic Prefix Caching) and MTP (Multi-Token Prediction) by
        # preserving the cache hit when beneficial.
        num_spec_tokens = getattr(kv_cache_spec, "num_spec_tokens", None)
        if (
            num_spec_tokens is None
            or num_spec_tokens == 0
            or max_length % block_size < num_spec_tokens
        ):
            for computed in computed_blocks:
                computed.pop()
    # adapt end
    while (
        block_size != alignment_tokens  # Faster for common case.
        and len(computed_blocks[0]) * block_size % alignment_tokens != 0
    ):
        for computed in computed_blocks:
            computed.pop()
    return computed_blocks


FullAttentionManager.find_longest_cache_hit = find_longest_cache_hit

logger.info(
    "Mamba align APC patch applied: FullAttentionManager.find_longest_cache_hit"
)
