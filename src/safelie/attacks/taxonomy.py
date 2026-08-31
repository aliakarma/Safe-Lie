"""The four-axis corruption taxonomy (Table 2, main_iclr.tex).

Report reference: PROJECT_REPORT.md §1.3, Table `tab:taxonomy`.

An attack is a point in a 2x2x2x2 product space, not a row in a flat
list — this makes the axes independently sweepable in a factorial config,
rather than producing overlapping conditions that cannot be varied one at
a time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Direction = Literal["negative", "positive", "mixed"]
Support = Literal["persistent", "selective"]
Adaptivity = Literal["static", "adaptive"]
Consistency = Literal["consistent", "byzantine"]


@dataclass(frozen=True)
class TaxonomyPoint:
    direction: Direction
    support: Support
    adaptivity: Adaptivity
    consistency: Consistency

    def __str__(self) -> str:
        return f"({self.direction}, {self.support}, {self.adaptivity}, {self.consistency})"


# The two named attack instances the paper actually specifies (App. B).
PRIMARY_ATTACK = TaxonomyPoint("negative", "persistent", "static", "consistent")
STEALTH_ATTACK = TaxonomyPoint("negative", "selective", "adaptive", "consistent")

# Named for the gating-defense discussion (§5.1): over-reporting that
# defeats a naive update-gating defense (Proposition 3).
OVER_REPORT_ATTACK = TaxonomyPoint("positive", "persistent", "static", "consistent")
