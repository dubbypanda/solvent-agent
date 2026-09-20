"""Tests for the customer book (solvent.customers)."""

from __future__ import annotations

import time

import pytest

from solvent.customers import (
    book_summary,
    customer_jobs,
    customer_stats,
    format_customer_detail,
    format_customers,
)
from solvent.treasury import Treasury


@pytest.fixture
def treasury(tmp_path):
    return Treasury(path=tmp_path / "ledger.db")


def _job(treasury, job_id, email, *, status="completed", budget=4_900, topic="topic"):
    treasury.upsert_job(job_id, status, topic=topic, budget_cents=budget, customer_email=email)


def test_empty_book(treasury):
    assert customer_stats(treasury) == []
    summary = book_summary([])
    assert summary["customers"] == 0
    assert summary["repeat_rate_pct"] == 0.0
    assert "No customers yet" in format_customers([], summary)


def test_revenue_and_cogs_come_from_the_ledger(treasury):
    _job(treasury, "J1", "a@x.example")
    treasury.earn(4_900, "paid", job_id="J1", stripe_ref="pi_1")
    treasury.spend(600, "vendors", job_id="J1", vendor="nvidia-nemotron")

    row = customer_stats(treasury)[0]
    assert row["email"] == "a@x.example"
    assert row["revenue_cents"] == 4_900
    assert row["cogs_cents"] == 600
    assert row["net_cents"] == 4_300
    assert row["margin_pct"] == pytest.approx(87.8, abs=0.1)


def test_repeat_customers_are_flagged_and_counted(treasury):
    _job(treasury, "J1", "repeat@x.example")
    _job(treasury, "J2", "repeat@x.example")
    _job(treasury, "J3", "once@x.example")
    for job_id in ("J1", "J2", "J3"):
        treasury.earn(1_000, "paid", job_id=job_id, stripe_ref=f"pi_{job_id}")

    rows = customer_stats(treasury)
    by_email = {r["email"]: r for r in rows}
    assert by_email["repeat@x.example"]["repeat"] is True
    assert by_email["repeat@x.example"]["jobs"] == 2
    assert by_email["once@x.example"]["repeat"] is False

    summary = book_summary(rows)
    assert summary["customers"] == 2
    assert summary["repeat_customers"] == 1
    assert summary["repeat_rate_pct"] == 50.0


def test_email_case_and_whitespace_fold_into_one_customer(treasury):
    _job(treasury, "J1", "Mixed@X.example")
    _job(treasury, "J2", "  mixed@x.example ")
    rows = customer_stats(treasury)
    assert len(rows) == 1
    assert rows[0]["jobs"] == 2


def test_declined_jobs_do_not_dilute_the_average_order(treasury):
    _job(treasury, "J1", "a@x.example", budget=5_000)
    _job(treasury, "J2", "a@x.example", status="failed", budget=600)
    treasury.earn(5_000, "paid", job_id="J1", stripe_ref="pi_1")

    row = customer_stats(treasury)[0]
    assert row["declined"] == 1
    assert row["avg_order_cents"] == 5_000  # one billed job, not two


def test_book_is_ranked_by_net_contribution(treasury):
    _job(treasury, "J1", "small@x.example")
    _job(treasury, "J2", "big@x.example")
    treasury.earn(1_000, "paid", job_id="J1", stripe_ref="pi_1")
    treasury.earn(9_000, "paid", job_id="J2", stripe_ref="pi_2")

    assert [r["email"] for r in customer_stats(treasury)] == [
        "big@x.example",
        "small@x.example",
    ]


def test_summary_reports_concentration_risk(treasury):
    _job(treasury, "J1", "whale@x.example")
    _job(treasury, "J2", "minnow@x.example")
    treasury.earn(9_000, "paid", job_id="J1", stripe_ref="pi_1")
    treasury.earn(1_000, "paid", job_id="J2", stripe_ref="pi_2")

    rows = customer_stats(treasury)
    summary = book_summary(rows)
    assert summary["top_customer_share_pct"] == 90.0
    assert "concentration risk" in format_customers(rows, summary)


def test_customer_detail_lists_jobs_newest_first(treasury):
    now = time.time()
    treasury.upsert_job("old", "completed", topic="old job", customer_email="a@x.example")
    treasury.upsert_job("new", "completed", topic="new job", customer_email="a@x.example")
    # upsert_job stamps created_at itself; force a known order.
    with treasury._conn() as conn, conn:
        conn.execute("UPDATE jobs SET created_at = ? WHERE id = 'old'", (now - 1_000,))
        conn.execute("UPDATE jobs SET created_at = ? WHERE id = 'new'", (now,))

    rows = customer_jobs("A@X.example", treasury)
    assert [j["id"] for j in rows] == ["new", "old"]
    assert "new job" in format_customer_detail("a@x.example", rows)


def test_customer_detail_for_an_unknown_email_is_empty(treasury):
    assert customer_jobs("nobody@x.example", treasury) == []
    assert "No jobs" in format_customer_detail("nobody@x.example", [])
