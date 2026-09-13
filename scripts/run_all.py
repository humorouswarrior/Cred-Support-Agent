"""Run the whole project end to end under MOCK_LLM and regenerate every transcript.

    python scripts/run_all.py            # full run
    python scripts/run_all.py --reset    # rebuild the vector store from scratch first
"""

from __future__ import annotations

import shutil
import sys
import time

from _common import PROJECT_ROOT, banner

import acceptance_check
import run_part1
import run_part2
import run_part3
import run_part4


def reset_artifacts() -> None:
    from cred_support_agent.config import CHROMA_DIR, LOG_PATH, CALIBRATION_PATH

    shutil.rmtree(CHROMA_DIR, ignore_errors=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.unlink(missing_ok=True)
    CALIBRATION_PATH.unlink(missing_ok=True)
    print(f"[reset] cleared vector store, request log and calibration under {PROJECT_ROOT / 'artifacts'}")


def main() -> int:
    if "--reset" in sys.argv:
        reset_artifacts()

    started = time.perf_counter()
    for label, module in (
        ("PART 1 - DATASET DESIGN & RAG CORE", run_part1),
        ("PART 2 - CREWAI ORCHESTRATION, TOOLS, MEMORY, GUARDRAILS", run_part2),
        ("PART 3 - EVALUATION, OBSERVABILITY & FASTAPI DEPLOYMENT", run_part3),
        ("PART 4 - RESILIENCE & GOVERNANCE", run_part4),
    ):
        banner(label)
        module.main()

    banner("ACCEPTANCE CHECKLIST")
    status = acceptance_check.main()

    elapsed = time.perf_counter() - started
    print(f"\nfull end-to-end run completed in {elapsed:.1f}s under MOCK_LLM "
          f"(no API keys, no network calls)")
    return status


if __name__ == "__main__":
    sys.exit(main())
