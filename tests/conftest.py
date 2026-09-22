"""Test-wide fixtures.

``NSREngine.save_front`` defaults to ``True`` and writes to a *working
directory*-relative ``nsr_pareto_front/``, so without this every fitting test
would drop a CSV in the repo root.  Running each test in its own directory
keeps that side effect where it can be inspected -- and exercises the real
default rather than an opted-out one.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def run_in_tmp_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
