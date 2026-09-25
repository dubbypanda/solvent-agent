"""
calibration.py — teach the margin gate what jobs actually cost.

`pricing.py` estimates fulfilment cost from a static table of vendor rates. The
stage machine already records what each job *really* cost once it was
fulfilled (`job_metrics.actual_cost_cents`), but nothing fed that back into the
next quote. A shop whose cost model drifts sells below its floor without ever
noticing.

The feedback is deliberately one-sided. When realized costs run **hotter** than
the model, quotes are marked up by the observed ratio, so the margin floor
keeps meaning what it says. When they run **cooler**, nothing happens
automatically: an optimistic sample is not a reason to quote closer to the
bone, and cutting prices is a decision for the operator, who sees the same
number in `solvent costs` and can lower `.solvent/pricing_overrides.json` by
hand.

The factor is clamped and needs a minimum sample count, so one freak job cannot
move pricing.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from .pricing import RESOURCE_COSTS_CENTS, get_resource_costs
from .treasury import Treasury, fmt

#: Jobs to look back over. Recent behaviour beats ancient history.
DEFAULT_WINDOW = 20

#: Below this many usable samples the static model is left alone.
MIN_SAMPLES = 5

#: The factor never quotes below the static model (1.0) and never more than
#: doubles it, whatever the sample says.
FACTOR_FLOOR = 1.0
FACTOR_CEILING = 2.0


def _usable_rows(treasury: Treasury, window: int) -> list[dict[str, Any]]:
    """Metrics rows with both an estimate and a realized cost, newest `window`."""
    rows = [
        row
        for row in treasury.list_metrics()
        if (row.get("est_cost_cents") or 0) > 0
        and (row.get("actual_cost_cents") or 0) > 0
        and not row.get("refunded")
    ]
    return rows[-window:]


def cost_drift(treasury: Treasury | None = None, *, window: int = DEFAULT_WINDOW) -> dict[str, Any]:
    """Compare estimated against realized fulfilment cost over recent jobs."""
    t = treasury or Treasury()
    rows = _usable_rows(t, window)
    est_total = sum(row["est_cost_cents"] for row in rows)
    actual_total = sum(row["actual_cost_cents"] for row in rows)
    ratio = round(actual_total / est_total, 3) if est_total else 1.0

    worst = sorted(
        (
            {
                "job_id": row.get("job_id"),
                "est_cost_cents": row["est_cost_cents"],
                "actual_cost_cents": row["actual_cost_cents"],
                "drift_cents": row["actual_cost_cents"] - row["est_cost_cents"],
                "ratio": round(row["actual_cost_cents"] / row["est_cost_cents"], 3),
            }
            for row in rows
        ),
        key=lambda r: -r["ratio"],
    )[:5]

    return {
        "samples": len(rows),
        "window": window,
        "est_total_cents": est_total,
        "actual_total_cents": actual_total,
        "drift_cents": actual_total - est_total,
        "ratio": ratio,
        "worst_jobs": worst,
    }


def calibration_factor(
    treasury: Treasury | None = None,
    *,
    window: int = DEFAULT_WINDOW,
    min_samples: int = MIN_SAMPLES,
    drift: dict[str, Any] | None = None,
) -> float:
    """The multiplier the margin gate should apply to its cost estimates.

    Returns exactly ``1.0`` — the static model, untouched — when there is not
    enough evidence, or when realized costs came in at or below the estimate.
    """
    observed = drift if drift is not None else cost_drift(treasury, window=window)
    if observed["samples"] < min_samples:
        return 1.0
    return round(min(max(observed["ratio"], FACTOR_FLOOR), FACTOR_CEILING), 3)


def recommendation(drift: dict[str, Any], factor: float) -> str:
    """One line of plain advice about the cost model."""
    samples = drift["samples"]
    if samples < MIN_SAMPLES:
        return (
            f"Only {samples} fulfilled job(s) on record — "
            f"{MIN_SAMPLES} are needed before quotes are calibrated."
        )
    ratio = drift["ratio"]
    if factor > 1.0:
        return (
            f"Realized costs are running {round((ratio - 1) * 100, 1)}% above the model; "
            f"quotes are marked up ×{factor} to protect the margin floor."
        )
    if ratio < 0.9:
        return (
            f"Realized costs are {round((1 - ratio) * 100, 1)}% below the model. Quotes are "
            "left alone; lower the rates in .solvent/pricing_overrides.json to price keener."
        )
    return "The cost model matches what jobs actually cost. No adjustment applied."


def report(treasury: Treasury | None = None, *, window: int = DEFAULT_WINDOW) -> dict[str, Any]:
    """Everything `solvent costs` prints, as plain data."""
    t = treasury or Treasury()
    drift = cost_drift(t, window=window)
    factor = calibration_factor(drift=drift)
    effective = get_resource_costs()
    return {
        "drift": drift,
        "factor": factor,
        "recommendation": recommendation(drift, factor),
        "resource_costs": [
            {
                "resource": name,
                "default_cents": RESOURCE_COSTS_CENTS[name],
                "effective_cents": effective.get(name, default),
                "calibrated_cents": round(effective.get(name, default) * factor),
            }
            for name, default in RESOURCE_COSTS_CENTS.items()
        ],
    }


def format_report(data: dict[str, Any]) -> str:
    """Render the cost-model report for a terminal."""
    drift = data["drift"]
    lines = [
        "",
        "  COST MODEL",
        f"  {'─' * 70}",
        f"  Samples              {drift['samples']} fulfilled job(s) (window {drift['window']})",
        f"  Estimated cost       {fmt(drift['est_total_cents'])}",
        f"  Realized cost        {fmt(drift['actual_total_cents'])}  "
        f"(drift {fmt(drift['drift_cents'])}, ×{drift['ratio']})",
        f"  Calibration in force ×{data['factor']}",
        "",
        f"  {'RESOURCE':<26}{'RATE':>10}{'QUOTED AT':>12}",
    ]
    for row in data["resource_costs"]:
        lines.append(
            f"  {row['resource']:<26}{fmt(row['effective_cents']):>10}"
            f"{fmt(row['calibrated_cents']):>12}"
        )

    if drift["worst_jobs"]:
        lines += ["", "  Worst drift"]
        for job in drift["worst_jobs"]:
            lines.append(
                f"    {str(job['job_id'])[:12]:<14}est {fmt(job['est_cost_cents']):>8}"
                f"  actual {fmt(job['actual_cost_cents']):>8}  ×{job['ratio']}"
            )
    lines += ["", f"  → {data['recommendation']}", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="solvent costs",
        description="Estimated vs realized fulfilment cost, and the calibration it implies.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW,
        help=f"how many recent fulfilled jobs to look at (default {DEFAULT_WINDOW})",
    )
    parser.add_argument("--json", action="store_true", dest="as_json", help="output as JSON")
    args = parser.parse_args()

    data = report(window=args.window)
    if args.as_json:
        print(json.dumps(data, indent=2, default=str))
    else:
        print(format_report(data))


if __name__ == "__main__":
    main()
