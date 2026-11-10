"""Hemdutt Rao: the three Wave 0 orderings must hold under fair IV (document 3.1)."""

import pytest

from optspread.gates import wave0_gate


@pytest.mark.invariant
@pytest.mark.slow
def test_wave0_orderings_hold() -> None:
    report = wave0_gate(n_episodes=100, seed=1000)
    assert report.passed, "\n" + report.render()
