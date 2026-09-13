"""Central configuration for the Cred domain support agent.

Everything in this project is designed to run with zero API keys and zero
network access at grading time.  The only switch that matters is
``CRED_LLM_BACKEND``: it defaults to ``mock`` (deterministic MOCK_LLM) and a
real backend is opt-in behind the env flag.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Telemetry must be disabled *before* crewai is imported anywhere, otherwise
# kickoff() can attempt outbound OTEL traffic.  Importing this module is the
# first thing every entry point in the project does.
# --------------------------------------------------------------------------
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_TELEMETRY_OPT_OUT", "true")
# Also disables CrewAI's first-run tracing prompt, which otherwise prints a
# banner on every kickoff() and, on the opt-in path, opens a browser and uploads
# an execution trace. Neither is acceptable for a zero-network run.
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
# CrewAI keeps a per-user "first execution" flag in the home directory. On a
# machine that has never run CrewAI, the first kickoff() would print a "Tracing
# Preference Saved" banner (or, in an interactive terminal, wait 20 seconds on a
# prompt), so output would depend on the machine. In CrewAI 1.x this variable
# only switches that first-run flow off, which makes every machine behave alike.
os.environ.setdefault("CREWAI_TESTING", "true")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")          # chromadb
os.environ.setdefault("CHROMA_TELEMETRY_IMPL", "chromadb.telemetry.posthog.NoopTelemetry")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
# Zero network at run time. bootstrap.py downloads the embedding model once; after
# that, Hugging Face libraries must not contact huggingface.co. Without these,
# every model load sends HEAD requests to check for updates, and with no network
# each one retries with back-off, stalling a run for minutes.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# The embedding model lives inside the project (downloaded once by bootstrap.py),
# not in the user's home directory, so a run does not depend on which account runs
# it, where $HOME is, or any HF_HOME/HF_HUB_CACHE the machine already has set.
# These are assigned, not defaulted, and before any Hugging Face library is
# imported. CRED_MODEL_DIR is the one supported override.
MODEL_DIR = Path(os.environ.get("CRED_MODEL_DIR", Path(__file__).resolve().parent.parent / "models"))
os.environ["HF_HUB_CACHE"] = str(MODEL_DIR)
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(MODEL_DIR)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Runtime state (vector store, request log, calibration). CRED_ARTIFACT_DIR lets the
# test suite use a throwaway directory, so tests never write to the project's store.
ARTIFACT_DIR = Path(os.environ.get("CRED_ARTIFACT_DIR", PROJECT_ROOT / "artifacts"))
TRANSCRIPT_DIR = PROJECT_ROOT / "transcripts"
CHROMA_DIR = ARTIFACT_DIR / "chroma"
LOG_PATH = ARTIFACT_DIR / "requests.jsonl"
CALIBRATION_PATH = ARTIFACT_DIR / "threshold_calibration.json"
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "data" / "knowledge_base"

for _d in (ARTIFACT_DIR, TRANSCRIPT_DIR, CHROMA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# CrewAI (a small SQLite file) and ChromaDB (an anonymous id file) write under the
# user's home directory. Some machines have none, or a read-only one (containers run
# with --user, CI sandboxes, service accounts), and both libraries then crash on
# import. In that case, give them a private home inside the artifact directory.
def _writable_dir(path: Path) -> bool:
    try:
        return path.is_dir() and os.access(path, os.W_OK)
    except (OSError, RuntimeError):
        return False


try:
    _home = Path.home()
except (KeyError, RuntimeError):
    _home = Path("/nonexistent")
if not _writable_dir(_home):
    FALLBACK_HOME = ARTIFACT_DIR / "home"
    FALLBACK_HOME.mkdir(parents=True, exist_ok=True)
    for _var in ("HOME", "USERPROFILE", "LOCALAPPDATA", "APPDATA"):
        os.environ[_var] = str(FALLBACK_HOME)
    os.environ["XDG_DATA_HOME"] = str(FALLBACK_HOME / ".local" / "share")
    os.environ["XDG_CACHE_HOME"] = str(FALLBACK_HOME / ".cache")

# --------------------------------------------------------------------------
# Dataset generation design choices (Part 1 / Task 1).  These constants are
# the reproducibility contract quoted verbatim in README.md.
# --------------------------------------------------------------------------
DATASET_SEED = 20240917
DATASET_SIZE = 48
FRAUD_FLAG_PROBABILITY = 0.20          # target centre of the required 10-30% band
MAX_DAYS_SINCE_CREATED = 30

# --------------------------------------------------------------------------
# Retrieval / generation
# --------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = os.environ.get("CRED_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
FIXED_CHUNK_COLLECTION = "cred_kb_fixed_overlap"
SENTENCE_CHUNK_COLLECTION = "cred_kb_sentence"
FIXED_CHUNK_SIZE = 240                  # characters
FIXED_CHUNK_OVERLAP = 60                # characters
DEFAULT_TOP_K = 3

# Runtime retrieval strategy. Set to the strategy recommended by the Task 5
# measurement (fixed_overlap won on mean precision/F1); see retrieval/evaluation.py.
DEFAULT_RETRIEVAL_STRATEGY = os.environ.get("CRED_RETRIEVAL_STRATEGY", "fixed_overlap")

# Fallback only.  The value actually used at runtime is measured by
# ``calibration.calibrate_threshold`` and persisted to CALIBRATION_PATH.
FALLBACK_SIMILARITY_THRESHOLD = 0.35

# --------------------------------------------------------------------------
# Escalation scoring (Part 2 / Task 6)
# --------------------------------------------------------------------------
ESCALATION_WEIGHT_FRAUD = 0.5
ESCALATION_WEIGHT_AGE = 0.5
ESCALATION_PERCENTILE = 80              # cutoff anchored on this percentile of age

# --------------------------------------------------------------------------
# Runtime governance (Part 4 / Task 15)
# --------------------------------------------------------------------------
# Caps are set from measured behaviour, not guessed. Every language-model stage
# charges the per-request ledger: the three crew agents, the grounded-generation
# model and both Autogen reviewers. Across the 15 evaluation queries, the Task 4/5
# queries and multi-part status questions (with the cache cold, the worst case), a
# normal request used 2,700-7,753 estimated tokens over 5-8 model calls, median
# 5,160. The running cap sits at 12,000 tokens (about 1.55x the busiest observed
# request). The cost cap is INR 2.00; at INR 0.15 per 1k tokens the token cap binds
# first (12,000 tokens = INR 1.80), and the cost cap still holds if the unit price
# changes. The intake cap rejects any single customer turn over ~2,400 characters,
# far beyond any genuine support question.
MAX_REQUEST_TOKENS = 600                # per-request prompt budget at intake
MAX_RESPONSE_TOKENS = 800
MAX_TOTAL_TOKENS_PER_REQUEST = 12000    # running total across every model call
COST_PER_1K_TOKENS_INR = 0.15           # notional unit cost for the budget ledger
MAX_COST_PER_REQUEST_INR = 2.00         # running cost, enforced on every model call

# --------------------------------------------------------------------------
# Guardrails (Part 2 / Task 10)
# --------------------------------------------------------------------------
GROUNDEDNESS_MIN_SUPPORT = 0.45         # share of answer content words seen in context

LLM_BACKEND = os.environ.get("CRED_LLM_BACKEND", "mock").strip().lower()


def is_mock_backend() -> bool:
    """MOCK_LLM is the default and the only mode required for acceptance."""
    return LLM_BACKEND == "mock"
