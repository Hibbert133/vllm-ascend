from vllm.v1.kv_cache_interface import FullAttentionSpec

origin_init = FullAttentionSpec.__init__


def _full_attn_init(self, **kwargs):
    num_spec_tokens = kwargs.pop("num_spec_tokens", 0)
    origin_init(self, **kwargs)
    object.__setattr__(self, "num_spec_tokens", num_spec_tokens)

original_repr = FullAttentionSpec.__repr__


def _full_attn_repr(self):
    base = original_repr(self)
    if hasattr(self, "num_spec_tokens"):
        base = base[:-1] + f", num_spec_tokens={self.num_spec_tokens}"
    return base

original_merge = FullAttentionSpec.merge


def _full_attn_merge(cls, specs):
    merged = original_merge(specs)
    num_spec_tokens = getattr(specs[0], "num_spec_tokens", 0)
    object.__setattr__(merged, "num_spec_tokens", num_spec_tokens)
    return merged


FullAttentionSpec.__init__ = _full_attn_init
FullAttentionSpec.__repr__ = _full_attn_repr
FullAttentionSpec.merge = _full_attn_merge
