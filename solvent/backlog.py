"""
backlog.py — which job the agent should work on next.

A queue is not a plan. When several jobs are waiting and the treasury has a
finite amount of cash and a finite 24h spend budget, the order the agent works
in decides how much money it makes. This module ranks the open backlog the way
a business would:

1. **Finish what is already paid for.** Revenue is collected before cost is
   incurred, so a paid job left unfinished is a refund waiting to happen.
2. **Then the best return on capital** — projected margin per cent of
   fulfilment cost, not raw margin, so a $20 job that costs $5 outranks a $90
   job that costs $60.
3. **Never start work the treasury cannot fund.** A job whose fulfilment would
   breach the rolling spend budget or the cash reserve is deferred rather than
   started and refunded half way through.

The worker consumes this ordering; `solvent backlog` prints it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from .guardrails import Guardrails
from .pricing import PricingPolicy, quote
from .queue import WORKER_STATUSES
from .treasury import Treasury


@dataclass
class BacklogItem:
    """One open job, scored and funding-checked."""

    job_id: str
    topic: str
    status: str
    price_cents: int = 0
    est_cost_cents: int = 0
    margin_cents: int = 0
    margin_pct: float = 0.0
    roi: float = 0.0
    paid: bool = False
    fundable: bool = True
    defer_reason: str | None = None
    scored: bool = True
    job: dict = field(default_factory=dict, repr=False)

    @property
    def sort_key(self) -> tuple:
        # Paid work first, then best return on capital, then bigger margin.
        return (0 if self.paid else 1, -self.roi, -self.margin_cents)

    def as_dict(self) -> dict:
        data = asdict(self)
        data.pop("job", None)
        return data


def job_payload(row: dict) -> dict:
    """Rebuild a job dict from its treasury row (payload JSON if present)."""
    payload = row.get("job_payload_json")
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            pass
    elif isinstance(payload, dict):
        return payload
    return {
        "id": row.get("id"),
        "topic": row.get("topic"),
        "budget_cents": row.get("budget_cents"),
        "customer_email": row.get("customer_email"),
        "est_tokens": row.get("est_tokens"),
        "market_data_calls": row.get("market_data_calls"),
        "web_search_calls": row.get("web_search_calls"),
    }


def score_job(
    row: dict,
    treasury: Treasury,
    policy: PricingPolicy | None = None,
) -> BacklogItem:
    """Score one backlog row. Never raises: an unscorable row sorts last-but-runnable."""
    job_id = str(row.get("id", ""))
    item = BacklogItem(
        job_id=job_id,
        topic=row.get("topic") or "",
        status=row.get("status") or "",
        job=job_payload(row),
    )
    try:
        item.paid = bool(treasury.job_has_revenue(job_id))
    except Exception:
        item.paid = False
    try:
        q = quote(item.job, policy, with_counter_offer=False)
        item.price_cents = q.price_cents
        item.est_cost_cents = q.est_cost_cents
        item.margin_cents = q.margin_cents
        item.margin_pct = q.margin_pct
        item.roi = round(q.margin_cents / max(q.est_cost_cents, 1), 3)
    except Exception:
        item.scored = False
    return item


def spend_capacity(guard: Guardrails) -> int | None:
    """Cents the agent may still spend: the tighter of 24h budget and cash reserve.

    Returns ``None`` when the capacity cannot be read (no treasury yet, or a
    stubbed one), meaning "do not constrain".
    """
    try:
        budget = guard.policy.daily_budget_cents
        spent = guard._spent_last_24h()
        balance = guard.t.balance_cents()
        floor = guard.policy.min_reserve_cents
    except (TypeError, ValueError, AttributeError):
        return None
    numbers = (budget, spent, balance, floor)
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in numbers):
        return None  # a stubbed or mocked treasury — do not constrain
    return max(min(int(budget - spent), int(balance - floor)), 0)


def rank(
    rows: list[dict],
    treasury: Treasury,
    guard: Guardrails,
    policy: PricingPolicy | None = None,
) -> list[BacklogItem]:
    """Rank open jobs by priority and mark each one fundable or deferred.

    Paid jobs are always fundable: the customer's money is already in the
    treasury and abandoning the work would mean refunding it. Unpaid jobs are
    committed against the remaining spend capacity in ranked order; one that
    does not fit is *deferred*, not skipped over permanently, and a cheaper job
    behind it can still take the remaining capacity.

    Deferring matters because the quote stage declines — permanently — any job
    whose fulfilment would breach the spend budget or the cash reserve. Holding
    such a job back until the treasury can fund it turns a lost customer into a
    later one.
    """
    items = sorted((score_job(row, treasury, policy) for row in rows), key=lambda i: i.sort_key)
    capacity = spend_capacity(guard)
    if capacity is None:
        return items

    committed = 0
    for item in items:
        if item.paid or not item.scored:
            # The customer's money is already in the treasury (or the job could
            # not be scored) — finish it rather than hold it back.
            continue
        if committed + item.est_cost_cents > capacity:
            item.fundable = False
            item.defer_reason = (
                f"fulfilment needs {item.est_cost_cents}c; only {max(capacity - committed, 0)}c "
                "of spend capacity left (24h budget / cash reserve)"
            )
            continue
        committed += item.est_cost_cents
    return items


def prioritise(rows: list[dict], treasury: Treasury, guard: Guardrails) -> list[dict]:
    """The worker's view: runnable job rows, best first, deferred ones held back."""
    by_id = {str(row.get("id", "")): row for row in rows}
    return [by_id[item.job_id] for item in rank(rows, treasury, guard) if item.fundable]


def open_rows(treasury: Treasury) -> list[dict]:
    """Every job the worker could still act on."""
    return treasury.list_jobs_by_status(list(WORKER_STATUSES))


def plan(treasury: Treasury | None = None, guard: Guardrails | None = None) -> dict:
    """The whole backlog, ranked, with the capital it commits and the margin it buys."""
    t = treasury or Treasury()
    g = guard or Guardrails(t)
    items = rank(open_rows(t), t, g)
    fundable = [i for i in items if i.fundable]
    return {
        "items": [i.as_dict() for i in items],
        "open_jobs": len(items),
        "fundable_jobs": len(fundable),
        "deferred_jobs": len(items) - len(fundable),
        "spend_capacity_cents": spend_capacity(g),
        "committed_cost_cents": sum(i.est_cost_cents for i in fundable if not i.paid),
        "projected_margin_cents": sum(i.margin_cents for i in fundable),
    }


def format_plan(data: dict) -> str:
    """Render a ranked backlog for a terminal."""
    from .treasury import fmt

    lines = [
        "",
        "  BACKLOG — what the agent works on next",
        f"  {'─' * 74}",
    ]
    if not data["items"]:
        lines += ["  Nothing open. Every job is completed, failed, or declined.", ""]
        return "\n".join(lines)

    lines.append(f"  {'#':<3}{'JOB':<12}{'STATUS':<22}{'PRICE':>9}{'COST':>9}{'ROI':>7}  TOPIC")
    for rank_no, item in enumerate(data["items"], start=1):
        flag = " " if item["fundable"] else "⏸"
        paid = "$" if item["paid"] else " "
        lines.append(
            f"  {rank_no:<3}{item['job_id'][:11]:<12}{item['status'][:21]:<22}"
            f"{fmt(item['price_cents']):>9}{fmt(item['est_cost_cents']):>9}"
            f"{item['roi']:>7.2f}{paid}{flag} {item['topic'][:28]}"
        )
    for item in data["items"]:
        if item["defer_reason"]:
            lines.append(f"    ⏸ {item['job_id']}: {item['defer_reason']}")

    capacity = data["spend_capacity_cents"]
    lines += [
        "",
        f"  Open {data['open_jobs']}  ·  runnable now {data['fundable_jobs']}  ·  "
        f"deferred {data['deferred_jobs']}",
        f"  Spend capacity      {fmt(capacity) if capacity is not None else 'unbounded'}",
        f"  Committed to unpaid work  {fmt(data['committed_cost_cents'])}",
        f"  Projected margin    {fmt(data['projected_margin_cents'])}",
        "  ($ = already paid for, ⏸ = deferred until the treasury can fund it)",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="solvent backlog",
        description="Rank the open backlog by return on capital and fundability.",
    )
    parser.add_argument("--json", action="store_true", dest="as_json", help="output as JSON")
    args = parser.parse_args()

    data = plan()
    if args.as_json:
        print(json.dumps(data, indent=2, default=str))
    else:
        print(format_plan(data))


if __name__ == "__main__":
    main()
