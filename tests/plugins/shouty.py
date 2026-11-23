"""Example plugin used by the test-suite."""

from llmverdict.scorers import Scorer, register_scorer
from llmverdict.types import Sample, Score


@register_scorer
class Shouty(Scorer):
    type_name = "shouty"

    async def score(self, sample: Sample, output: str) -> Score:
        return self._bool(output.isupper())
