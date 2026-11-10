"""QC-19 bounded-liveness integer kernel for independent MPO witness checking.

Cache entries are released immediately after the corresponding pre-update
core leaves the verified MPO chain. The exact QC-16 arithmetic is unchanged;
all results remain exact dyadic fractions. No NumPy or simulator dependencies.
"""
from __future__ import annotations

from qc16_dyadic import Kernel


class LiveKernel(Kernel):
    """Cache only still-live tensor versions and the reusable gate/Kraus matrices.

    The verifier calls ``retire()`` only after all residual and context checks
    involving the pre-update tensors have completed. Source identity is checked
    before release, so id reuse cannot invalidate a newer tensor.
    """

    def __init__(self, cache_limit: int = 96):
        if type(cache_limit) is not int or not 1 <= cache_limit <= 256:
            raise ValueError("cache limit must be between 1 and 256")
        super().__init__(cache_limit=cache_limit)

    def retire(self, tensor):
        entry = self.cache.get(id(tensor))
        if entry is not None and entry[0] is tensor:
            self.cache.pop(id(tensor))
