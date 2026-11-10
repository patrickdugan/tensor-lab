"""QC-21 exact summary-cache kernel for MPO witness replay.

Cache scalar squared norms (dyadic Fractions), which are sufficient to validate
all subsequent context multipliers for an unchanged MPO core. Keep just a
small, bounded set of the bulky integer cores used by residual contractions.
Cached summaries are tied to tensor *identity* and retired together with that
version, preventing cross-event reuse after a core changes.

The original residual, Gram, norm, channel and readout arithmetic stays in
QC-20's independently tested exact-integer kernel. This module changes only
its caching decisions.
"""
from __future__ import annotations

from fractions import Fraction as F
from qc15_checker import sqrt_up
from qc20_packed import StreamFastKernel


class SummaryKernel(StreamFastKernel):
    """Live norm summaries plus a bounded converted-integer operand cache.

    `cache_limit` controls the number of large converted operands; the separate
    small rational summary cache holds only still-live tensor versions.
    """

    def __init__(self, cache_limit: int = 6):
        super().__init__(cache_limit=cache_limit)
        self.norm_summaries = {}
        self.summary_hits = 0
        self.summary_misses = 0

    def _memo(self, tensor, key, calculation):
        entry = self.norm_summaries.get(id(tensor))
        if entry is not None and entry[0] is tensor and key in entry[1]:
            self.summary_hits += 1
            return entry[1][key]
        answer = calculation()
        # A nested memoization call may have installed this tensor already.
        entry = self.norm_summaries.get(id(tensor))
        if entry is None or entry[0] is not tensor:
            self.norm_summaries[id(tensor)] = (tensor, {key: answer})
        else:
            entry[1][key] = answer
        self.summary_misses += 1
        return answer

    def frob_norm2(self, tensor):
        return self._memo(tensor, 'frob', lambda: super(SummaryKernel, self).frob_norm2(tensor))

    def gram_defect2(self, tensor, orientation):
        # Always validate the input orientation even on cache hits.
        if orientation not in ('L', 'R'):
            raise ValueError('bad isometry orientation')
        return self._memo(tensor, orientation,
                          lambda: super(SummaryKernel, self).gram_defect2(tensor, orientation))

    def context_bound(self, cores, excluded, *, gate=False):
        # Cache the *outward-rounded* roots as well as their exact squared
        # norms. Repeat the same index order and Fraction multiplies as QC-16.
        lo = min(excluded)
        hi = max(excluded)
        product = F(1)
        for idx, core in enumerate(cores):
            if idx in excluded:
                continue
            if gate and idx < lo:
                factor = self._memo(core, 'root_L',
                                    lambda core=core: 1 + sqrt_up(self.gram_defect2(core, 'L')))
            elif gate and idx > hi:
                factor = self._memo(core, 'root_R',
                                    lambda core=core: 1 + sqrt_up(self.gram_defect2(core, 'R')))
            else:
                factor = self._memo(core, 'root_frob',
                                    lambda core=core: sqrt_up(self.frob_norm2(core)))
            product *= factor
        return product

    def retire(self, tensor):
        super().retire(tensor)
        entry = self.norm_summaries.get(id(tensor))
        if entry is not None and entry[0] is tensor:
            del self.norm_summaries[id(tensor)]
