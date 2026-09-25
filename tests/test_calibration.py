"""Tests for the self-calibrating cost model (solvent.calibration + pricing)."""

from __future__ import annotations

import pytest

from solvent.calibration import (
    FACTOR_CEILING,
    MIN_SAMPLES,
    calibration_factor,
    cost_drift,
    format_report,
    recommendation,
    report,
)
from solvent.pricing import PricingPolicy, calibrated_cost, estimate_cost, quote
from solvent.treasury import Treasury

JOB = {
    "id": "J1",
    "topic": "topic",
    "budget_cents": 4_900,
    "est_tokens": 9_000,
    "market_data_calls": 2,
    "web_search_calls": 8,
}


@pytest.fixture
def treasury(tmp_path):
    return Treasury(path=tmp_path / "ledger.db")


def _metrics(treasury, count, est, actual, *, refunded=0, prefix="J"):
    for i in range(count):
        treasury.upsert_metrics(
            f"{prefix}{i}",
            est_cost_cents=est,
            actual_cost_cents=actual,
            refunded=refunded,
        )


def test_no_history_leaves_the_static_model_alone(treasury):
    drift = cost_drift(treasury)
    assert drift["samples"] == 0
    assert drift["ratio"] == 1.0
    assert calibration_factor(treasury) == 1.0


def test_too_few_samples_leaves_the_model_alone(treasury):
    _metrics(treasury, MIN_SAMPLES - 1, 1_000, 2_000)
    assert calibration_factor(treasury) == 1.0
    assert "are needed before" in recommendation(cost_drift(treasury), 1.0)


def test_hot_costs_mark_quotes_up(treasury):
    _metrics(treasury, MIN_SAMPLES, 1_000, 1_400)
    drift = cost_drift(treasury)
    assert drift["ratio"] == pytest.approx(1.4)
    assert calibration_factor(treasury) == pytest.approx(1.4)
    assert "above the model" in recommendation(drift, 1.4)


def test_cool_costs_never_cut_the_price(treasury):
    """One-sided by design: an optimistic sample must not erode the floor."""
    _metrics(treasury, MIN_SAMPLES, 1_000, 200)
    assert calibration_factor(treasury) == 1.0
    assert "below the model" in recommendation(cost_drift(treasury), 1.0)


def test_factor_is_clamped_at_the_ceiling(treasury):
    _metrics(treasury, MIN_SAMPLES, 100, 10_000)
    assert calibration_factor(treasury) == FACTOR_CEILING


def test_refunded_jobs_are_excluded_from_the_sample(treasury):
    _metrics(treasury, MIN_SAMPLES, 1_000, 5_000, refunded=1, prefix="R")
    assert cost_drift(treasury)["samples"] == 0
    assert calibration_factor(treasury) == 1.0


def test_window_keeps_only_recent_jobs(treasury):
    _metrics(treasury, 6, 1_000, 3_000, prefix="old")
    _metrics(treasury, 6, 1_000, 1_000, prefix="new")
    drift = cost_drift(treasury, window=6)
    assert drift["samples"] == 6
    assert drift["ratio"] == pytest.approx(1.0)


def test_worst_drift_is_ranked_first(treasury):
    treasury.upsert_metrics("mild", est_cost_cents=1_000, actual_cost_cents=1_100)
    treasury.upsert_metrics("wild", est_cost_cents=1_000, actual_cost_cents=4_000)
    assert cost_drift(treasury)["worst_jobs"][0]["job_id"] == "wild"


# --- pricing integration ----------------------------------------------------


def test_calibration_scales_the_cost_breakdown_and_its_total():
    raw_total, raw_breakdown = estimate_cost(JOB)
    total, breakdown = calibrated_cost(JOB, PricingPolicy(cost_calibration=1.5))

    assert total == sum(breakdown.values())  # the lines always add up
    assert total > raw_total
    for line, cents in breakdown.items():
        assert cents == round(raw_breakdown[line] * 1.5)


def test_calibration_of_one_is_the_untouched_static_model():
    assert calibrated_cost(JOB, PricingPolicy()) == estimate_cost(JOB)


def test_a_marked_up_quote_can_fall_below_the_margin_floor():
    lean = {**JOB, "budget_cents": 1_600}
    assert quote(lean).accept
    marked_up = quote(lean, PricingPolicy(cost_calibration=2.0))
    assert not marked_up.accept
    assert "below floor" in marked_up.reason


def test_counter_offer_prices_off_the_calibrated_cost():
    policy = PricingPolicy(cost_calibration=2.0)
    declined = quote({**JOB, "budget_cents": 1_600}, policy)
    offer = declined.counter_offer
    assert offer is not None
    # The offered deal must clear the floor against calibrated, not raw, costs.
    assert quote(
        {**JOB, **(offer["scope"] or {}), "budget_cents": offer["price_cents"]}, policy
    ).accept


def test_agent_starts_with_the_calibration_its_history_implies(tmp_path):
    from solvent.agent import Solvent

    agent = Solvent(seed_cents=10_000, fresh=True)
    db = tmp_path / "t.db"
    agent.t.path = db
    agent.t.lock_path = db.with_suffix(".lock")
    agent.t._init_db()
    assert agent.pricing.cost_calibration == 1.0  # fresh treasury, no history

    _metrics(agent.t, MIN_SAMPLES, 1_000, 1_500)
    assert calibration_factor(agent.t) == pytest.approx(1.5)


def test_report_renders(treasury):
    _metrics(treasury, MIN_SAMPLES, 1_000, 1_200)
    data = report(treasury)
    assert data["factor"] == pytest.approx(1.2)
    rendered = format_report(data)
    assert "COST MODEL" in rendered
    assert "Calibration in force" in rendered
    assert "nemotron_tokens_per_1k" in rendered
