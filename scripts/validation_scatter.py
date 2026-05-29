#!/usr/bin/env python3
"""Regenerate the predicted-vs-measured TTFT/ITL scatter plot.

For each experiment under experiments/<expN>/data, runs the joint
Nelder-Mead fit by invoking demos/guidellm-multiple with all sweep files
concatenated by the FileNameSeparator '$'. Parses the optimizer's
predicted/measured table from stdout and overlays both models in a
two-panel scatter, saved as PDF (vector text).

The fit produced here is identical to the one whose alpha/beta/gamma and
mean errors appear in tabs/validation-fit.tex of the MASCOTS 2026 paper.

Usage:
    python scripts/validation_scatter.py [-o OUTPUT_PDF]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_PKG = "./demos/guidellm-multiple"
FILE_SEP = "$"

ROW_RE = re.compile(
    r"^\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+"
    r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$"
)


@dataclass
class FitResult:
    label: str
    color: str
    marker: str
    alpha: float
    beta: float
    gamma: float
    err_ttft: float
    err_itl: float
    ttft_meas: np.ndarray
    ttft_pred: np.ndarray
    itl_meas: np.ndarray
    itl_pred: np.ndarray


def joint_fit(data_dir: Path, label: str, color: str, marker: str) -> FitResult:
    files = sorted(data_dir.glob("sweep-i*-o*.*"))
    if not files:
        raise FileNotFoundError(f"no sweep files under {data_dir}")
    arg = FILE_SEP.join(str(f) for f in files)

    proc = subprocess.run(
        ["go", "run", DEMO_PKG, arg],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return _parse(proc.stdout, label, color, marker)


def _parse(stdout: str, label: str, color: str, marker: str) -> FitResult:
    lines = stdout.splitlines()

    parms_idx = next(
        (i for i, l in enumerate(lines) if l.strip() == "Estimated parameters:"),
        None,
    )
    if parms_idx is None or parms_idx + 1 >= len(lines):
        raise RuntimeError("did not find 'Estimated parameters:' in trainer output")
    parms = json.loads(lines[parms_idx + 1])
    p = parms["OptimizedParms"]

    rows = []
    for line in lines:
        if "TTFTMeas" in line or "TTFTPred" in line:
            continue
        m = ROW_RE.match(line)
        if m:
            rows.append([float(g) for g in m.groups()])
    if not rows:
        raise RuntimeError("no predicted/measured rows parsed from trainer output")
    arr = np.asarray(rows)
    ttft_meas, ttft_pred = arr[:, 3], arr[:, 4]
    itl_meas, itl_pred = arr[:, 5], arr[:, 6]

    err_ttft = float(np.mean(np.abs(ttft_pred - ttft_meas)) / np.mean(ttft_meas))
    err_itl = float(np.mean(np.abs(itl_pred - itl_meas)) / np.mean(itl_meas))

    return FitResult(
        label=label,
        color=color,
        marker=marker,
        alpha=p["alpha"],
        beta=p["beta"],
        gamma=p["gamma"],
        err_ttft=err_ttft,
        err_itl=err_itl,
        ttft_meas=ttft_meas,
        ttft_pred=ttft_pred,
        itl_meas=itl_meas,
        itl_pred=itl_pred,
    )


def plot(fits: list[FitResult], out_path: Path) -> None:
    fig, (ax_ttft, ax_itl) = plt.subplots(1, 2, figsize=(10, 4.5))

    for f in fits:
        ax_ttft.scatter(
            f.ttft_meas, f.ttft_pred,
            s=36, marker=f.marker, color=f.color, alpha=0.7, label=f.label,
        )
        ax_itl.scatter(
            f.itl_meas, f.itl_pred,
            s=36, marker=f.marker, color=f.color, alpha=0.7, label=f.label,
        )

    for ax, xlabel, ylabel in (
        (ax_ttft, "Measured TTFT (ms)", "Predicted TTFT (ms)"),
        (ax_itl, "Measured ITL (ms)", "Predicted ITL (ms)"),
    ):
        lo = 0.0
        hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    ax_ttft.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-o", "--output",
        default=str(REPO_ROOT / "experiments" / "validation-scatter.pdf"),
        help="output PDF path",
    )
    args = ap.parse_args()

    fits = [
        joint_fit(REPO_ROOT / "experiments" / "exp1" / "data",
                  "Llama-3.1-8B", "tab:blue", "o"),
        joint_fit(REPO_ROOT / "experiments" / "exp2" / "data",
                  "Qwen2.5-14B", "tab:red", "^"),
    ]

    print(f"\n{'model':<16}{'alpha':>10}{'beta':>14}{'gamma':>14}"
          f"{'err_ttft':>12}{'err_itl':>12}{'N':>6}")
    for f in fits:
        print(f"{f.label:<16}{f.alpha:>10.3f}{f.beta:>14.4e}{f.gamma:>14.4e}"
              f"{f.err_ttft:>12.4f}{f.err_itl:>12.4f}{len(f.ttft_meas):>6d}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plot(fits, out)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
