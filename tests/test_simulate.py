"""Tests for the policy simulator (solvent.simulate)."""

from __future__ import annotations

import random

import pytest

from solvent.guardrails import SpendPolicy
from solvent.pricing import PricingPolicy
from solvent.simulate import JobMix, format_simulation, run_trial, simulate
from solvent.treasury import REFUND_VENDOR, Treasury


def _trial(**kwargs):
    defaults = dict(
        jobs=10,
        capital_cents=10_000,
        pricing=PricingPolicy(),
        spend_policy=SpendPolicy(),
        mix=JobMix(),
        cost_spread=0.0,
    )
    defaults.update(kwargs)
    return run_trial(random.Random(1), **defaults)


def test_same_seed_gives_the_same_run():
    first = simulate(trials=5, jobs=6, seed=42)
    second = simulate(trials=5, jobs=6, seed=42)
    assert first == second


def test_different_seeds_diverge():
    assert simulate(trials=5, jobs=6, seed=1) != simulate(trials=5, jobs=6, seed=2)


def test_every_offered_job_is_accounted_for():
    data = simulate(trials=8, jobs=7, seed=3)
    assert data["jobs_offered"] == 56
    assert data["jobs_accepted"] + data["jobs_declined"] + data["jobs_blocked"] == 56


def test_the_simulation_never_touches_the_real_treasury(tmp_path, monkeypatch):
    monkeypatch.setenv("SOLVENT_HOME", str(tmp_path))
    before = Treasury(path=tmp_path / "ledger.db")
    before.seed(5_000)

    simulate(trials=3, jobs=5, seed=9)

    after = Treasury(path=tmp_path / "ledger.db")
    assert after.balance_cents() == 5_000
    assert after.list_jobs() == []


def test_a_punishing_margin_floor_declines_everything():
    data = simulate(trials=3, jobs=8, seed=5, pricing=PricingPolicy(margin_floor_pct=99.9))
    assert data["jobs_accepted"] == 0
    assert data["acceptance_rate_pct"] == 0.0
    assert "below margin floor" in data["decline_reasons"]


def test_profitable_policy_grows_the_paper_treasury():
    data = simulate(trials=20, jobs=8, capital_cents=10_000, seed=11)
    assert data["balance_p50_cents"] > 10_000
    assert data["mean_net_cents"] > 0


def test_a_vendor_price_shock_erodes_the_margin():
    calm = simulate(trials=20, jobs=8, seed=11)
    shock = simulate(trials=20, jobs=8, seed=11, cost_multiplier=8.0)
    assert shock["mean_net_cents"] < calm["mean_net_cents"]
    assert shock["balance_p50_cents"] < calm["balance_p50_cents"]


def test_a_tight_transaction_cap_blocks_and_refunds():
    data = simulate(
        trials=5,
        jobs=8,
        seed=13,
        spend_policy=SpendPolicy(max_txn_cents=1),
        cost_multiplier=5.0,
    )
    assert data["jobs_blocked"] > 0
    assert "max_txn_cap" in data["block_rules"]


def test_blocked_jobs_are_refunded_not_kept():
    trial = _trial(spend_policy=SpendPolicy(max_txn_cents=1), cost_spread=0.0)
    assert trial.blocked > 0
    assert trial.refunded_cents == trial.revenue_cents


def test_simulated_refunds_do_not_eat_the_spend_budget():
    """Matches the live ledger: a refund is tagged, not counted as vendor spend."""
    trial = _trial(spend_policy=SpendPolicy(max_txn_cents=1))
    assert trial.blocked > 0
    assert "daily_budget" not in trial.block_rules


def test_refund_entries_carry_the_refund_vendor_tag():
    assert REFUND_VENDOR == "customer-refund"


def test_zero_cost_spread_is_deterministic_per_job():
    assert _trial(cost_spread=0.0) == _trial(cost_spread=0.0)


def test_job_mix_respects_its_bounds():
    mix = JobMix(min_budget_cents=2_000, max_budget_cents=3_000)
    rng = random.Random(0)
    for i in range(50):
        job = mix.draw(rng, i)
        assert 2_000 <= job["budget_cents"] <= 3_000
        assert job["est_tokens"] > 0
        assert job["market_data_calls"] >= 0
        assert job["web_search_calls"] >= 1


def test_percentiles_are_ordered():
    data = simulate(trials=25, jobs=6, seed=17)
    assert data["balance_p10_cents"] <= data["balance_p50_cents"] <= data["balance_p90_cents"]


def test_format_simulation_reports_the_policy_and_the_outcome():
    rendered = format_simulation(simulate(trials=5, jobs=5, seed=21))
    assert "POLICY SIMULATION" in rendered
    assert "Ending balance" in rendered
    assert "Margin floor" in rendered


@pytest.mark.parametrize("trials,jobs", [(1, 1), (3, 2)])
def test_small_runs_do_not_divide_by_zero(trials, jobs):
    data = simulate(trials=trials, jobs=jobs, seed=1)
    assert data["jobs_offered"] == trials * jobs
    assert isinstance(data["acceptance_rate_pct"], float)
