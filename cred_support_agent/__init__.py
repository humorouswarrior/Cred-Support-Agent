"""Cred Domain Support Agent (Banking & FinTech track).

Importing any part of this package runs this file first, and this file imports
``config`` first. ``config`` sets ``CREWAI_DISABLE_TELEMETRY=true``,
``OTEL_SDK_DISABLED=true`` and the related switches, so telemetry is off before
CrewAI, ChromaDB or Hugging Face are loaded, whichever module is imported first.
"""

from cred_support_agent import config as _config  # noqa: F401  (sets telemetry env vars)

assert _config.PROJECT_ROOT.exists()
