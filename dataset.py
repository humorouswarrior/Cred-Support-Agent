"""Task 1 - the seeded, deterministic loan-application dataset generator.

The brief names this file ``dataset.py``. The generator itself lives in
``cred_support_agent/data/dataset.py`` so that the rest of the package can import
it; this top-level module re-exports it and prints the required report.

    python dataset.py

prints the per-category counts, per-status counts and the fraud-review percentage,
then validates every structural threshold from the brief.
"""

from __future__ import annotations

from cred_support_agent.data.dataset import (
    AMOUNT_BANDS_INR,
    CATEGORY_WEIGHTS,
    GLOBAL_AMOUNT_RANGE_INR,
    LOAN_APPLICATIONS,
    REQUIRED_CATEGORIES,
    REQUIRED_STATUSES,
    STATUS_WEIGHTS,
    dataset_profile,
    generate_loan_applications,
    get_application,
    report,
    validate_dataset,
)

__all__ = [
    "AMOUNT_BANDS_INR",
    "CATEGORY_WEIGHTS",
    "GLOBAL_AMOUNT_RANGE_INR",
    "LOAN_APPLICATIONS",
    "REQUIRED_CATEGORIES",
    "REQUIRED_STATUSES",
    "STATUS_WEIGHTS",
    "dataset_profile",
    "generate_loan_applications",
    "get_application",
    "report",
    "validate_dataset",
]

if __name__ == "__main__":
    print(report())
