#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused compatibility test for Batch 15 Hotfix 01."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "TRB/set/src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ra_ild_trb.linear_svm_visualization import _boxplot_with_tick_labels


class _LegacyAxis:
    def __init__(self) -> None:
        self.calls = []

    def boxplot(self, values, **kwargs):
        self.calls.append(dict(kwargs))
        if "tick_labels" in kwargs:
            raise TypeError("boxplot() got an unexpected keyword argument 'tick_labels'")
        assert kwargs["labels"] == ["RA", "ILD"]
        assert kwargs["showmeans"] is True
        return "legacy-ok"


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="batch15_hotfix01_") as temp:
        output = Path(temp) / "boxplot.pdf"
        figure, axis = plt.subplots()
        _boxplot_with_tick_labels(
            axis,
            [np.asarray([1.0, 2.0]), np.asarray([2.0, 3.0])],
            ["RA", "ILD"],
            showmeans=True,
        )
        figure.savefig(output)
        plt.close(figure)
        assert output.read_bytes().startswith(b"%PDF")

    legacy = _LegacyAxis()
    assert (
        _boxplot_with_tick_labels(
            legacy,
            [[1.0], [2.0]],
            ["RA", "ILD"],
            showmeans=True,
        )
        == "legacy-ok"
    )
    assert len(legacy.calls) == 2
    assert "tick_labels" in legacy.calls[0]
    assert "labels" in legacy.calls[1]
    print(f"PASS current Matplotlib boxplot API ({matplotlib.__version__})")
    print("PASS legacy Matplotlib labels fallback")
    print("TRB_BATCH15_HOTFIX01_MATPLOTLIB_ACCEPTANCE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
