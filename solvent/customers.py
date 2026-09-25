"""
customers.py — who actually pays the agent.

The treasury knows what the business earned; it did not know *from whom*. Every
job already carries a customer email, and every ledger entry carries a job id,
so the two join into the view a business steers by: lifetime value per
customer, how much of it was profit, and who came back.

Repeat business is the number to watch. A job won twice costs nothing to
acquire the second time, so the repeat rate says more about whether the shop
works than a single month's revenue does.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from typing import Any

from .treasury import Treasury, fmt

#: Job statuses that mean the customer never got billed.
UNBILLED_STATUSES = ("failed", "declined", "cancelled")


def _blank(email: str) -> dict[str, Any]:
    return {
        "email": email,
        "jobs": 0,
        "completed": 0,
        "declined": 0,
        "revenue_cents": 0,
        "cogs_cents": 0,
        "net_cents": 0,
        "margin_pct": 0.0,
        "avg_order_cents": 0,
        "first_seen": None,
        "last_seen": None,
        "repeat": False,
    }


def customer_stats(treasury: Treasury | None = None) -> list[dict[str, Any]]:
    """Per-customer lifetime value, ranked by net contribution.

    Revenue and COGS come from the ledger (the money that actually moved), not
    from quoted prices, so a refunded job nets out the way it should.
    """
    t = treasury or Treasury()
    jobs = t.list_jobs()
    revenue_by_job: dict[str, int] = defaultdict(int)
    cogs_by_job: dict[str, int] = defaultdict(int)
    for entry in t.entries:
        if not entry.job_id:
            continue
        if entry.kind == "revenue":
            revenue_by_job[entry.job_id] += entry.amount_cents
        elif entry.kind == "expense":
            cogs_by_job[entry.job_id] += entry.amount_cents

    stats: dict[str, dict[str, Any]] = {}
    for job in jobs:
        email = (job.get("customer_email") or "unknown").strip().lower()
        row = stats.setdefault(email, _blank(email))
        job_id = job.get("id", "")
        status = job.get("status", "")

        row["jobs"] += 1
        if status == "completed":
            row["completed"] += 1
        elif status in UNBILLED_STATUSES:
            row["declined"] += 1
        row["revenue_cents"] += revenue_by_job.get(job_id, 0)
        row["cogs_cents"] += cogs_by_job.get(job_id, 0)

        started = job.get("created_at") or job.get("updated_at")
        touched = job.get("updated_at") or job.get("created_at")
        if started is not None:
            row["first_seen"] = (
                started if row["first_seen"] is None else min(row["first_seen"], started)
            )
        if touched is not None:
            row["last_seen"] = (
                touched if row["last_seen"] is None else max(row["last_seen"], touched)
            )

    for row in stats.values():
        row["net_cents"] = row["revenue_cents"] - row["cogs_cents"]
        row["margin_pct"] = (
            round(100 * row["net_cents"] / row["revenue_cents"], 1) if row["revenue_cents"] else 0.0
        )
        paid_jobs = max(row["jobs"] - row["declined"], 0)
        row["avg_order_cents"] = round(row["revenue_cents"] / paid_jobs) if paid_jobs else 0
        row["repeat"] = row["jobs"] > 1

    return sorted(stats.values(), key=lambda r: (-r["net_cents"], r["email"]))


def book_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Totals across the customer book: concentration, repeat rate, ARPU."""
    revenue = sum(r["revenue_cents"] for r in rows)
    net = sum(r["net_cents"] for r in rows)
    repeats = [r for r in rows if r["repeat"]]
    top_share = round(100 * rows[0]["revenue_cents"] / revenue, 1) if rows and revenue else 0.0
    return {
        "customers": len(rows),
        "repeat_customers": len(repeats),
        "repeat_rate_pct": round(100 * len(repeats) / len(rows), 1) if rows else 0.0,
        "revenue_cents": revenue,
        "net_cents": net,
        "revenue_per_customer_cents": round(revenue / len(rows)) if rows else 0,
        # Share of revenue from the single best customer: the concentration risk.
        "top_customer_share_pct": top_share,
    }


def customer_jobs(email: str, treasury: Treasury | None = None) -> list[dict[str, Any]]:
    """Every job belonging to one customer, newest first."""
    t = treasury or Treasury()
    target = email.strip().lower()
    rows = [j for j in t.list_jobs() if (j.get("customer_email") or "").strip().lower() == target]
    rows.sort(key=lambda j: j.get("created_at") or 0, reverse=True)
    for job in rows:
        job["pnl_cents"] = t.job_pnl_cents(job.get("id", ""))
    return rows


def _ago(ts: float | None) -> str:
    if not ts:
        return "-"
    delta = max(time.time() - float(ts), 0)
    for unit, seconds in (("d", 86_400), ("h", 3_600), ("m", 60)):
        if delta >= seconds:
            return f"{int(delta // seconds)}{unit} ago"
    return "just now"


def format_customers(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    """Render the customer book for a terminal."""
    lines = ["", "  CUSTOMER BOOK", f"  {'─' * 74}"]
    if not rows:
        lines += ["  No customers yet — no job has carried an email address.", ""]
        return "\n".join(lines)

    lines.append(f"  {'CUSTOMER':<34}{'JOBS':>5}{'REVENUE':>11}{'NET':>11}{'MARGIN':>8}  LAST")
    for row in rows:
        marker = "↻" if row["repeat"] else " "
        lines.append(
            f"  {marker}{row['email'][:32]:<33}{row['jobs']:>5}"
            f"{fmt(row['revenue_cents']):>11}{fmt(row['net_cents']):>11}"
            f"{row['margin_pct']:>7}%  {_ago(row['last_seen'])}"
        )
    lines += [
        "",
        f"  Customers {summary['customers']}  ·  repeat {summary['repeat_customers']} "
        f"({summary['repeat_rate_pct']}%)  ·  revenue per customer "
        f"{fmt(summary['revenue_per_customer_cents'])}",
        f"  Top customer is {summary['top_customer_share_pct']}% of revenue"
        f"{'  ⚠ concentration risk' if summary['top_customer_share_pct'] >= 50 else ''}",
        "  (↻ = repeat customer)",
        "",
    ]
    return "\n".join(lines)


def format_customer_detail(email: str, rows: list[dict[str, Any]]) -> str:
    """Render one customer's job history."""
    lines = ["", f"  {email}", f"  {'─' * 74}"]
    if not rows:
        lines += ["  No jobs for this customer.", ""]
        return "\n".join(lines)
    lines.append(f"  {'JOB':<14}{'STATUS':<22}{'BUDGET':>10}{'P&L':>10}  TOPIC")
    for job in rows:
        lines.append(
            f"  {str(job.get('id', ''))[:13]:<14}{str(job.get('status', ''))[:21]:<22}"
            f"{fmt(job.get('budget_cents') or 0):>10}{fmt(job.get('pnl_cents') or 0):>10}"
            f"  {(job.get('topic') or '')[:30]}"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="solvent customers",
        description="Lifetime value, repeat rate, and margin by customer.",
    )
    parser.add_argument("--email", help="show one customer's job history instead of the book")
    parser.add_argument("--limit", type=int, default=20, help="how many customers to list")
    parser.add_argument("--json", action="store_true", dest="as_json", help="output as JSON")
    args = parser.parse_args()

    treasury = Treasury()

    if args.email:
        rows = customer_jobs(args.email, treasury)
        if args.as_json:
            print(json.dumps(rows, indent=2, default=str))
        else:
            print(format_customer_detail(args.email, rows))
        sys.exit(0 if rows else 1)

    rows = customer_stats(treasury)
    summary = book_summary(rows)
    if args.as_json:
        print(json.dumps({"summary": summary, "customers": rows}, indent=2, default=str))
    else:
        print(format_customers(rows[: args.limit], summary))


if __name__ == "__main__":
    main()
