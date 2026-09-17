"""Tests for capital-aware backlog prioritisation (solvent.backlog)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from solvent.backlog import (
    BacklogItem,
    format_plan,
    job_payload,
    plan,
    prioritise,
    rank,
    score_job,
    spend_capacity,
)
from solvent.guardrails import Guardrails, SpendPolicy
from solvent.treasury import Treasury


def _row(job_id, budget, *, status="awaiting_payment", tokens=8_000, market=2, search=6):
    job = {
        "id": job_id,
        "topic": f"topic {job_id}",
        "budget_cents": budget,
        "est_tokens": tokens,
        "market_data_calls": market,
        "web_search_calls": search,
    }
    return {**job, "status": status, "job_payload_json": json.dumps(job)}


@pytest.fixture
def treasury(tmp_path):
    t = Treasury(path=tmp_path / "ledger.db")
    t.seed(100_000)
    return t


@pytest.fixture
def guard(treasury):
    return Guardrails(treasury, SpendPolicy())


def test_job_payload_prefers_the_stored_payload():
    row = _row("J1", 4_900)
    assert job_payload(row)["budget_cents"] == 4_900


def test_job_payload_falls_back_to_columns_when_payload_is_unusable():
    row = {**_row("J1", 4_900), "job_payload_json": "{not json"}
    assert job_payload(row)["id"] == "J1"

    row.pop("job_payload_json")
    assert job_payload(row)["budget_cents"] == 4_900


def test_score_job_computes_return_on_capital(treasury):
    item = score_job(_row("J1", 4_900), treasury)
    assert item.price_cents == 4_900
    assert item.est_cost_cents > 0
    assert item.roi == pytest.approx(item.margin_cents / item.est_cost_cents, rel=1e-3)
    assert item.scored


def test_score_job_survives_an_unquotable_row(treasury):
    item = score_job({"id": "bad", "status": "pending_quote", "budget_cents": "twenty"}, treasury)
    assert item.job_id == "bad"
    assert not item.scored


def test_best_return_on_capital_ranks_first(treasury, guard):
    rows = [
        _row("big-fat", 9_000, tokens=60_000, market=10, search=20),  # big margin, expensive
        _row("lean", 4_000, tokens=5_000, market=0, search=3),  # smaller margin, cheap
    ]
    ranked = rank(rows, treasury, guard)
    assert [i.job_id for i in ranked] == ["lean", "big-fat"]
    assert ranked[0].roi > ranked[1].roi


def test_paid_work_outranks_every_unpaid_job(treasury, guard):
    treasury.earn(4_000, "paid up front", job_id="paid-job", stripe_ref="pi_1")
    rows = [
        _row("lean", 4_000, tokens=5_000, market=0, search=3),
        _row("paid-job", 4_000, status="paid_pending_fulfill"),
    ]
    ranked = rank(rows, treasury, guard)
    assert ranked[0].job_id == "paid-job"
    assert ranked[0].paid


def test_unfundable_unpaid_work_is_deferred_not_dropped(tmp_path):
    treasury = Treasury(path=tmp_path / "thin.db")
    treasury.seed(2_200)  # $2 of spend capacity above the $20 reserve
    guard = Guardrails(treasury, SpendPolicy())

    ranked = rank([_row("J1", 9_900)], treasury, guard)
    assert len(ranked) == 1
    assert not ranked[0].fundable
    assert "spend capacity" in ranked[0].defer_reason


def test_a_cheaper_job_still_fits_after_an_expensive_one_is_deferred(tmp_path):
    treasury = Treasury(path=tmp_path / "tight.db")
    treasury.seed(2_700)  # $7 of capacity above the reserve
    guard = Guardrails(treasury, SpendPolicy())

    rows = [
        _row("expensive", 30_000, tokens=40_000, market=8, search=20),
        _row("cheap", 3_000, tokens=4_000, market=0, search=3),
    ]
    by_id = {i.job_id: i for i in rank(rows, treasury, guard)}
    assert not by_id["expensive"].fundable
    assert by_id["cheap"].fundable


def test_paid_work_is_funded_even_when_capacity_is_gone(tmp_path):
    treasury = Treasury(path=tmp_path / "paid.db")
    treasury.seed(2_000)  # exactly the reserve floor: no capacity at all
    treasury.earn(9_900, "paid", job_id="paid-job", stripe_ref="pi_2")
    guard = Guardrails(treasury, SpendPolicy())

    ranked = rank([_row("paid-job", 9_900, status="paid_pending_fulfill")], treasury, guard)
    assert ranked[0].fundable  # refunding the customer would be worse


def test_spend_capacity_is_the_tighter_of_budget_and_reserve(treasury, guard):
    treasury.spend(24_000, "burn", job_id="J0", vendor="nvidia-nemotron")
    # 24h budget left ($10) is now tighter than the cash above reserve.
    assert spend_capacity(guard) == guard.policy.daily_budget_cents - 24_000


def test_spend_capacity_is_unconstrained_for_a_stubbed_treasury():
    assert spend_capacity(Guardrails(MagicMock())) is None


def test_prioritise_returns_runnable_rows_in_ranked_order(treasury, guard):
    rows = [
        _row("big-fat", 9_000, tokens=60_000, market=10, search=20),
        _row("lean", 4_000, tokens=5_000, market=0, search=3),
    ]
    assert [r["id"] for r in prioritise(rows, treasury, guard)] == ["lean", "big-fat"]


def test_prioritise_holds_back_deferred_work(tmp_path):
    treasury = Treasury(path=tmp_path / "thin.db")
    treasury.seed(2_100)
    guard = Guardrails(treasury, SpendPolicy())
    assert prioritise([_row("J1", 9_900)], treasury, guard) == []


def test_plan_totals_the_backlog(treasury, guard):
    for row in (_row("J1", 4_900), _row("J2", 7_500)):
        treasury.upsert_job(
            row["id"],
            row["status"],
            topic=row["topic"],
            budget_cents=row["budget_cents"],
            est_tokens=row["est_tokens"],
            market_data_calls=row["market_data_calls"],
            web_search_calls=row["web_search_calls"],
            job_payload_json=json.loads(row["job_payload_json"]),
        )

    data = plan(treasury, guard)
    assert data["open_jobs"] == 2
    assert data["fundable_jobs"] == 2
    assert data["deferred_jobs"] == 0
    assert data["projected_margin_cents"] == sum(i["margin_cents"] for i in data["items"])
    assert "BACKLOG" in format_plan(data)


def test_format_plan_handles_an_empty_backlog(treasury, guard):
    assert "Nothing open" in format_plan(plan(treasury, guard))


def test_backlog_item_sort_key_puts_paid_first():
    paid = BacklogItem(job_id="p", topic="", status="", roi=0.1, paid=True)
    unpaid = BacklogItem(job_id="u", topic="", status="", roi=9.0, paid=False)
    assert sorted([unpaid, paid], key=lambda i: i.sort_key)[0] is paid


def test_worker_processes_the_backlog_in_ranked_order():
    """run_worker drives the claim loop through the backlog ranking."""
    from unittest import mock

    from solvent.worker import run_worker

    claimable = [{"id": "J-low"}, {"id": "J-high"}]
    with (
        mock.patch("solvent.worker.Solvent") as solvent_cls,
        mock.patch("solvent.worker.resume_incomplete_jobs", return_value=[]),
        mock.patch("solvent.worker.list_claimable", return_value=claimable),
        mock.patch("solvent.worker.prioritise", return_value=[{"id": "J-high"}]) as mock_prioritise,
    ):
        agent = MagicMock()
        solvent_cls.return_value = agent
        agent.t.claim_job.return_value = True

        run_worker(once=True)

    mock_prioritise.assert_called_once_with(claimable, agent.t, agent.guard)
    agent.advance_job.assert_called_once_with("J-high")
