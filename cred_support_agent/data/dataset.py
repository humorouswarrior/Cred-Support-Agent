"""Task 1 - deterministic loan-application dataset for Cred lending ops.

Design choices (reproducibility contract, mirrored in README.md):

* ``seed = 20240917`` fed to a single ``random.Random`` instance; every draw in
  this module comes from that instance in a fixed order, so the dataset is
  byte-identical on every machine and every run.
* ``48`` records.
* Category weights ``{Personal 0.30, Personal-adjacent ...}`` - see
  ``CATEGORY_WEIGHTS`` below; weights mirror a retail-lending book where
  unsecured personal loans dominate volume and business loans are rarest.
* Status weights - see ``STATUS_WEIGHTS``; most applications sit in the
  Submitted/Under Review part of the funnel, Disbursed is the smallest bucket.
* Amount range ``INR 50,000 - INR 5,000,000``, drawn from per-category bands,
  because Cred's retail book runs from small unsecured personal loans (tens of
  thousands of rupees) up to metro home loans in the tens of lakhs, and a single
  flat range would produce ``INR 50,000`` home loans that no reviewer would
  believe.
* ``flagged_for_fraud_review`` is an independent Bernoulli draw with
  ``p = 0.20`` - the centre of the required 10-30% band - so the realised
  percentage falls out of the seed rather than being hand-edited.

Coverage of every required category (>= 3 records) and every required status
(>= 1 record) is guaranteed *structurally* by seeding a coverage block first,
then filling the remainder by weighted sampling. No row is edited after the
fact.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Any, Dict, List

from cred_support_agent.config import (
    DATASET_SEED,
    DATASET_SIZE,
    ESCALATION_PERCENTILE,
    FRAUD_FLAG_PROBABILITY,
    MAX_DAYS_SINCE_CREATED,
)

REQUIRED_CATEGORIES = [
    "Personal Loan",
    "Home Loan",
    "Auto Loan",
    "Education Loan",
    "Business Loan",
]

REQUIRED_STATUSES = [
    "Submitted",
    "Under Review",
    "Approved",
    "Rejected",
    "Disbursed",
]

CATEGORY_WEIGHTS: Dict[str, float] = {
    "Personal Loan": 0.34,
    "Auto Loan": 0.22,
    "Home Loan": 0.18,
    "Education Loan": 0.14,
    "Business Loan": 0.12,
}

STATUS_WEIGHTS: Dict[str, float] = {
    "Under Review": 0.30,
    "Submitted": 0.24,
    "Approved": 0.20,
    "Rejected": 0.14,
    "Disbursed": 0.12,
}

# Per-category INR bands (inclusive), rounded to the nearest INR 1,000 on draw.
AMOUNT_BANDS_INR: Dict[str, tuple[int, int]] = {
    "Personal Loan": (50_000, 1_500_000),
    "Auto Loan": (150_000, 2_000_000),
    "Home Loan": (1_200_000, 5_000_000),
    "Education Loan": (100_000, 2_500_000),
    "Business Loan": (300_000, 5_000_000),
}

GLOBAL_AMOUNT_RANGE_INR = (50_000, 5_000_000)

RECORD_ID_PREFIX = "CRED-LN-"


def _record_id(index: int) -> str:
    return f"{RECORD_ID_PREFIX}{index:04d}"


def _weighted_pick(rng: random.Random, weights: Dict[str, float]) -> str:
    keys = list(weights)
    return rng.choices(keys, weights=[weights[k] for k in keys], k=1)[0]


def _draw_amount(rng: random.Random, category: str) -> int:
    low, high = AMOUNT_BANDS_INR[category]
    raw = rng.randint(low, high)
    return int(round(raw / 1000.0) * 1000)


def _coverage_plan() -> List[tuple[str, str | None]]:
    """Structural guarantee: 3 records per required category, and the first five
    of those carry the five required statuses, one each."""
    plan: List[tuple[str, str | None]] = []
    for position, category in enumerate(REQUIRED_CATEGORIES):
        plan.append((category, REQUIRED_STATUSES[position]))
        plan.append((category, None))
        plan.append((category, None))
    return plan


def generate_loan_applications(
    seed: int = DATASET_SEED, size: int = DATASET_SIZE
) -> List[Dict[str, Any]]:
    """Generate the deterministic loan-application dataset."""
    if size < len(REQUIRED_CATEGORIES) * 3:
        raise ValueError("size must leave room for 3 records per required category")

    rng = random.Random(seed)
    plan = _coverage_plan()
    plan += [(None, None)] * (size - len(plan))

    records: List[Dict[str, Any]] = []
    for index, (forced_category, forced_status) in enumerate(plan, start=1):
        category = forced_category or _weighted_pick(rng, CATEGORY_WEIGHTS)
        status = forced_status or _weighted_pick(rng, STATUS_WEIGHTS)
        records.append(
            {
                "record_id": _record_id(index),
                "category": category,
                "status": status,
                "loan_amount_inr": _draw_amount(rng, category),
                "days_since_created": rng.randint(0, MAX_DAYS_SINCE_CREATED),
                "flagged_for_fraud_review": rng.random() < FRAUD_FLAG_PROBABILITY,
            }
        )
    return records


LOAN_APPLICATIONS: List[Dict[str, Any]] = generate_loan_applications()

_INDEX: Dict[str, Dict[str, Any]] = {r["record_id"]: r for r in LOAN_APPLICATIONS}


def get_application(record_id: str) -> Dict[str, Any] | None:
    """Case/whitespace tolerant lookup by record id."""
    if not record_id:
        return None
    return _INDEX.get(record_id.strip().upper())


def percentile(values: List[float], pct: float) -> float:
    """Linear-interpolation percentile (no numpy dependency in the data layer)."""
    if not values:
        raise ValueError("percentile of empty sequence")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return float(ordered[low] + (ordered[high] - ordered[low]) * (rank - low))


def dataset_profile(records: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    """Per-category counts, per-status counts, fraud-review % and age stats."""
    rows = records if records is not None else LOAN_APPLICATIONS
    ages = [float(r["days_since_created"]) for r in rows]
    flagged = sum(1 for r in rows if r["flagged_for_fraud_review"])
    amounts = [r["loan_amount_inr"] for r in rows]
    return {
        "seed": DATASET_SEED,
        "total_records": len(rows),
        "category_counts": dict(sorted(Counter(r["category"] for r in rows).items())),
        "status_counts": dict(sorted(Counter(r["status"] for r in rows).items())),
        "flagged_count": flagged,
        "fraud_review_pct": round(100.0 * flagged / len(rows), 2),
        "amount_min_inr": min(amounts),
        "amount_max_inr": max(amounts),
        "days_since_created_percentile": {
            "p50": round(percentile(ages, 50), 2),
            "p80": round(percentile(ages, ESCALATION_PERCENTILE), 2),
            "p90": round(percentile(ages, 90), 2),
        },
    }


def validate_dataset(records: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    """Assert every structural threshold from the brief; returns the check table."""
    rows = records if records is not None else LOAN_APPLICATIONS
    profile = dataset_profile(rows)
    checks = {
        "at_least_40_records": len(rows) >= 40,
        "all_required_categories_present": all(
            c in profile["category_counts"] for c in REQUIRED_CATEGORIES
        ),
        "every_category_at_least_3": all(
            profile["category_counts"].get(c, 0) >= 3 for c in REQUIRED_CATEGORIES
        ),
        "all_required_statuses_present": all(
            s in profile["status_counts"] for s in REQUIRED_STATUSES
        ),
        "every_status_at_least_1": all(
            profile["status_counts"].get(s, 0) >= 1 for s in REQUIRED_STATUSES
        ),
        "fraud_pct_within_10_to_30": 10.0 <= profile["fraud_review_pct"] <= 30.0,
        "amounts_within_declared_range": all(
            GLOBAL_AMOUNT_RANGE_INR[0] <= r["loan_amount_inr"] <= GLOBAL_AMOUNT_RANGE_INR[1]
            for r in rows
        ),
        "days_within_0_to_30": all(
            0 <= r["days_since_created"] <= MAX_DAYS_SINCE_CREATED for r in rows
        ),
        "record_ids_unique": len({r["record_id"] for r in rows}) == len(rows),
    }
    return {"profile": profile, "checks": checks, "all_passed": all(checks.values())}


def report(records: List[Dict[str, Any]] | None = None) -> str:
    """Human-readable dataset report (Task 1 requires printing these counts)."""
    result = validate_dataset(records)
    profile = result["profile"]
    lines = [
        "=" * 72,
        "CRED LOAN APPLICATION DATASET - deterministic profile",
        "=" * 72,
        f"seed                 : {profile['seed']}",
        f"total records        : {profile['total_records']}",
        f"amount range (INR)   : {profile['amount_min_inr']:,} - {profile['amount_max_inr']:,}",
        "",
        "per-category counts:",
    ]
    for key, value in profile["category_counts"].items():
        lines.append(f"  {key:<18} {value:>3}")
    lines.append("")
    lines.append("per-status counts:")
    for key, value in profile["status_counts"].items():
        lines.append(f"  {key:<18} {value:>3}")
    lines.append("")
    lines.append(
        f"flagged for fraud review: {profile['flagged_count']}/{profile['total_records']} "
        f"= {profile['fraud_review_pct']}%  (required band: 10%-30%)"
    )
    pct = profile["days_since_created_percentile"]
    lines.append(
        f"days_since_created   : p50={pct['p50']}  p80={pct['p80']}  p90={pct['p90']}"
    )
    lines.append("")
    lines.append("structural checks:")
    for key, value in result["checks"].items():
        lines.append(f"  [{'PASS' if value else 'FAIL'}] {key}")
    lines.append("")
    lines.append(f"ALL CHECKS PASSED: {result['all_passed']}")
    lines.append("=" * 72)
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - manual inspection entry point
    print(report())
