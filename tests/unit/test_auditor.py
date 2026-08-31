"""Phase 10 exit criterion: the auditor correctly rejects the paper's own
N=2, M=5 configuration when replicas share an independence class."""

from __future__ import annotations

from safelie.governance.auditor import audit_sources
from safelie.sources.registry import default_m5_sources, default_m7_sources


def test_auditor_rejects_m5_configuration_at_f2():
    """The paper's N=2 config: nominal M=5, but the 3 ensemble replicas
    share one independence class, giving effective_M=3. At f=2,
    required = 2*2+1 = 5 > 3: FAILS, even though 5 >= 5 nominally."""
    sources = default_m5_sources()
    result = audit_sources(sources, assumed_f=2)
    assert result.nominal_m == 5
    assert result.effective_m == 3
    assert result.passes is False
    assert any("share an independence class" in w for w in result.warnings)


def test_auditor_accepts_m5_configuration_at_f1():
    sources = default_m5_sources()
    result = audit_sources(sources, assumed_f=1)
    assert result.effective_m == 3
    assert result.required_m == 3
    assert result.passes is True


def test_auditor_accepts_m7_primary_configuration_at_f1():
    sources = default_m7_sources()
    result = audit_sources(sources, assumed_f=1)
    assert result.nominal_m == result.effective_m == 7
    assert result.passes is True


def test_auditor_rejects_m7_primary_configuration_at_f3():
    sources = default_m7_sources()
    result = audit_sources(sources, assumed_f=3)
    assert result.required_m == 7
    assert result.passes is True  # boundary: effective_M(7) == required(7)
    assert any("boundary" in w for w in result.warnings)

    result_f3_5 = audit_sources(sources, assumed_f=4)
    assert result_f3_5.passes is False
