"""Tests for the counter-offer side of the margin gate (solvent.pricing + quote_cmd).

A decline now carries the deal the agent *would* accept: a narrower scope that
fits the customer's budget, or the lowest price that clears the margin floor.
"""

from __future__ import annotations

import pytest

from solvent.pricing import (
    PricingPolicy,
    counter_offer,
    job_scope,
    min_viable_price_cents,
    quote,
    scoped_alternative,
)
from solvent.quote_cmd import build_job, format_quote

CHEAP_JOB = {
    "id": "C1",
    "topic": "One-line definition of EBITDA",
    "budget_cents": 600,
    "est_tokens": 7_000,
    "market_data_calls": 2,
    "web_search_calls": 6,
}

# Budget is well over the minimum order size, but the scope is far too rich for it.
OVERSCOPED_JOB = {
    "id": "O1",
    "topic": "Exhaustive teardown of the memory market",
    "budget_cents": 2_500,
    "est_tokens": 60_000,
    "market_data_calls": 10,
    "web_search_calls": 20,
}


def test_accepted_quote_has_no_counter_offer():
    accepted = quote(
        {"id": "A", "topic": "t", "budget_cents": 9_900, "est_tokens": 9_000},
    )
    assert accepted.accept
    assert accepted.counter_offer is None


def test_min_viable_price_clears_the_floor_and_the_minimum_order():
    policy = PricingPolicy()
    price = min_viable_price_cents(1_000, policy)
    assert price is not None
    # Clears the margin floor...
    assert 100 * (price - 1_000) / price >= policy.margin_floor_pct
    # ...and never undercuts the minimum order size.
    assert min_viable_price_cents(10, policy) >= policy.min_price_cents
    # Whole dollars only.
    assert price % 100 == 0


def test_min_viable_price_is_none_when_no_price_can_clear_the_floor():
    assert min_viable_price_cents(1_000, PricingPolicy(margin_floor_pct=100.0)) is None


def test_tiny_budget_gets_a_price_counter_offer():
    declined = quote(CHEAP_JOB)
    assert not declined.accept
    offer = declined.counter_offer
    assert offer is not None
    assert offer["kind"] == "price"
    assert offer["price_cents"] > CHEAP_JOB["budget_cents"]
    assert offer["margin_pct"] >= PricingPolicy().margin_floor_pct
    # The offered price is one the gate would genuinely accept.
    assert quote({**CHEAP_JOB, "budget_cents": offer["price_cents"]}).accept


def test_overscoped_job_gets_a_scope_counter_offer_at_the_same_budget():
    declined = quote(OVERSCOPED_JOB)
    assert not declined.accept
    offer = declined.counter_offer
    assert offer is not None
    assert offer["kind"] == "scope"
    # The customer keeps their budget; the brief gets narrower.
    assert offer["price_cents"] == OVERSCOPED_JOB["budget_cents"]
    assert offer["scope_changes"]
    for dimension, value in offer["scope_changes"].items():
        assert value < OVERSCOPED_JOB[dimension]
    assert quote({**OVERSCOPED_JOB, **offer["scope"]}).accept


def test_scope_offer_is_the_richest_one_that_fits():
    scope, scoped_quote = scoped_alternative(OVERSCOPED_JOB)
    assert scoped_quote.accept
    # One step richer on the last-cut dimension would breach the floor.
    richer = {**scope, "est_tokens": scope["est_tokens"] + 1_000}
    assert not quote({**OVERSCOPED_JOB, **richer}).accept


def test_scope_offer_respects_the_policy_scope_floor():
    scope, _ = scoped_alternative(OVERSCOPED_JOB)
    floor = PricingPolicy().min_scope
    for dimension, minimum in floor.items():
        assert scope[dimension] >= minimum


def test_no_scope_offer_below_the_minimum_order_size():
    # Under the minimum order size, narrowing the brief cannot help.
    assert scoped_alternative(CHEAP_JOB) is None


def test_no_counter_offer_when_nothing_is_viable():
    impossible = PricingPolicy(margin_floor_pct=100.0)
    assert counter_offer(CHEAP_JOB, impossible) is None


def test_job_scope_fills_in_defaults():
    assert job_scope({}) == {
        "est_tokens": 8_000,
        "market_data_calls": 2,
        "web_search_calls": 6,
    }


def test_counter_offer_can_be_disabled_for_internal_evaluation():
    assert quote(CHEAP_JOB, with_counter_offer=False).counter_offer is None


@pytest.mark.parametrize("job", [CHEAP_JOB, OVERSCOPED_JOB])
def test_format_quote_shows_verdict_and_counter_offer(job):
    q = quote(job)
    rendered = format_quote(job, q, PricingPolicy())
    assert "DECLINE" in rendered
    assert "Counter-offer" in rendered
    assert q.reason in rendered


def test_build_job_carries_cli_arguments():
    job = build_job("topic", 4_900, est_tokens=12_000, market_data_calls=3, web_search_calls=9)
    assert job["budget_cents"] == 4_900
    assert job["est_tokens"] == 12_000
    assert job["market_data_calls"] == 3
    assert job["web_search_calls"] == 9


def test_declined_job_emits_a_counter_offer_event(tmp_path):
    """The stage machine publishes the counter-offer alongside the decline."""
    from solvent.agent import Solvent

    agent = Solvent(seed_cents=10_000, fresh=True)
    db = tmp_path / "t.db"
    agent.t.path = db
    agent.t.lock_path = db.with_suffix(".lock")
    agent.t._init_db()
    agent.t.seed(10_000)

    agent.handle_job({**CHEAP_JOB, "id": "J-decline"})

    stages = [e.get("stage") for e in agent.log]
    assert "declined" in stages
    offer = next(e for e in agent.log if e.get("stage") == "counter_offer")
    assert offer["job_id"] == "J-decline"
    assert offer["price_cents"] > CHEAP_JOB["budget_cents"]
    assert offer["message"]
    # The decline still comes first: the counter-offer is the follow-up.
    assert stages.index("declined") < stages.index("counter_offer")


def test_dashboard_shows_the_counter_offer_on_a_declined_card():
    from solvent.dashboard import build_status_data

    snapshot = {
        "balance_cents": 10_000,
        "capital_cents": 10_000,
        "revenue_cents": 0,
        "expense_cents": 0,
        "net_profit_cents": 0,
        "margin_pct": 0.0,
        "entries": [],
    }
    log = [
        {
            "stage": "quote",
            "job_id": "J3",
            "title": CHEAP_JOB["topic"],
            "price": 600,
            "est_cost": 543,
            "margin_pct": 9.5,
            "accept": False,
            "reason": "order $6 below minimum order size $15",
            "ts": 0,
        },
        {"stage": "declined", "job_id": "J3", "reason": "below minimum order size", "ts": 0},
        {
            "stage": "counter_offer",
            "job_id": "J3",
            "kind": "price",
            "price_cents": 1_500,
            "margin_pct": 63.8,
            "message": "can deliver this brief as specified for $15.00",
            "ts": 0,
        },
    ]

    data = build_status_data(snapshot, log)
    assert data["jobs_data"]["J3"]["counter_offer"].startswith("can deliver")
    assert "Counter-offer" in data["job_cards_html"]
    assert "Counter-offer" in data["console_log_html"]
