#!/usr/bin/env python3
"""One-command, cross-platform setup for the Cred Domain Support Agent.

Run it with any Python you have, on Linux, macOS or Windows:

    python bootstrap.py            # Linux / macOS (or python3)
    py bootstrap.py                # Windows

It needs only the standard library. It then:

  1. finds a supported interpreter (Python 3.12 or 3.13), even if the Python
     running this script is a different version;
  2. creates a virtual environment in .venv;
  3. installs the CPU-only PyTorch build (the default build bundles several GB of
     GPU libraries this project never uses);
  4. installs the pinned dependencies, with every transitive version locked by
     constraints.txt so the install is the same on every machine and every day;
  5. downloads the embedding model into ./models (inside the project, not the
     user's home directory);
  6. verifies the install fully offline and prints the next commands for this OS.

This is the only step that uses the network. Options:

    --python PATH      use this Python 3.12/3.13 interpreter for the environment
    --system           install into the running interpreter instead of a .venv
                       (used by the Dockerfile)
    --default-torch    install PyTorch from PyPI instead of the CPU-only index
    --skip-model       do not download the embedding model
    --model-only       only download and verify the embedding model (packages
                       already installed; used by the Dockerfile)
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
SUPPORTED = ((3, 12), (3, 13))
CPU_TORCH_INDEX = "https://download.pytorch.org/whl/cpu"
IS_WINDOWS = os.name == "nt"
IS_MACOS = sys.platform == "darwin"


def say(message: str) -> None:
    print(message, flush=True)


def fail(message: str) -> None:
    say("")
    say("SETUP FAILED: " + message)
    sys.exit(1)


def run(cmd: list[str], env: dict[str, str] | None = None, check: bool = True) -> int:
    say("      $ " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
    result = subprocess.run(cmd, cwd=ROOT, env=env)
    if check and result.returncode != 0:
        fail(f"command exited with status {result.returncode}")
    return result.returncode


def version_of(interpreter: list[str]) -> tuple[int, int] | None:
    """Ask an interpreter for its major.minor version; None if it can't run."""
    try:
        out = subprocess.run(
            interpreter + ["-c", "import sys; print(sys.version_info[0], sys.version_info[1])"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    parts = out.stdout.split()
    if out.returncode != 0 or len(parts) != 2:
        return None
    return int(parts[0]), int(parts[1])


def find_interpreter(explicit: str | None) -> list[str]:
    """Locate a Python 3.12 or 3.13 interpreter on this machine."""
    if explicit:
        candidate = [explicit]
        found = version_of(candidate)
        if found not in SUPPORTED:
            fail(f"--python {explicit} is Python {found}; this project needs 3.12 or 3.13.")
        return candidate

    candidates: list[list[str]] = [[sys.executable]]
    if IS_WINDOWS:
        candidates += [["py", "-3.12"], ["py", "-3.13"]]
    for name in ("python3.12", "python3.13"):
        path = shutil.which(name)
        if path:
            candidates.append([path])
    uv = shutil.which("uv")
    if uv:
        for minor in ("3.12", "3.13"):
            out = subprocess.run([uv, "python", "find", minor], capture_output=True, text=True)
            if out.returncode == 0 and out.stdout.strip():
                candidates.append([out.stdout.strip()])

    for candidate in candidates:
        if version_of(candidate) in SUPPORTED:
            return candidate

    fail(
        "no Python 3.12 or 3.13 found.\n"
        "  Install one, then rerun this script:\n"
        "    - any OS, easiest:  pip install uv   then   uv python install 3.12\n"
        "    - Windows:          https://www.python.org/downloads/  (3.12.x installer)\n"
        "    - macOS:            brew install python@3.12\n"
        "    - Debian/Ubuntu:    sudo apt install python3.12 python3.12-venv\n"
        "  Or point at one directly:  python bootstrap.py --python /path/to/python3.12"
    )
    raise AssertionError  # unreachable


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")


def pinned_torch_version() -> str:
    """Single source of truth for the PyTorch version: constraints.txt."""
    match = re.search(r"^torch==([0-9.]+)", (ROOT / "constraints.txt").read_text(encoding="utf-8"), re.M)
    if not match:
        fail("constraints.txt does not pin torch")
    return match.group(1)


def install_packages(python: list[str], args: argparse.Namespace) -> None:
    """Steps 2-3: CPU-only PyTorch where available, then the locked dependency set."""
    run(python + ["-m", "pip", "install", "--quiet", "--upgrade", "pip"])
    torch_version = pinned_torch_version()
    if args.default_torch or IS_MACOS:
        say(f"[2/5] PyTorch {torch_version} will come from PyPI with the other packages"
            + (" (macOS wheels are already CPU-only)" if IS_MACOS else ""))
    else:
        say(f"[2/5] Installing CPU-only PyTorch {torch_version} (about 200 MB instead of several GB)")
        code = run(python + ["-m", "pip", "install", "--quiet", f"torch=={torch_version}",
                             "--index-url", CPU_TORCH_INDEX], check=False)
        if code != 0:
            say("      CPU-only wheel unavailable for this machine; the PyPI build will be used instead.")

    # 3. everything else, fully locked
    say("[3/5] Installing pinned dependencies (requirements.txt, locked by constraints.txt)")
    run(python + ["-m", "pip", "install", "--quiet", "-r", "requirements.txt", "-c", "constraints.txt"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up the Cred Domain Support Agent.")
    parser.add_argument("--python", help="path to a Python 3.12/3.13 interpreter")
    parser.add_argument("--system", action="store_true", help="install into the running interpreter")
    parser.add_argument("--default-torch", action="store_true", help="install PyTorch from PyPI")
    parser.add_argument("--skip-model", action="store_true", help="skip the embedding model download")
    parser.add_argument("--model-only", action="store_true", help="only download and verify the model")
    args = parser.parse_args()

    say(f"Cred Domain Support Agent setup  ({platform.system()} {platform.machine()}, project at {ROOT})")

    # 1. interpreter + environment
    if args.model_only:
        python = [sys.executable] if args.system else [str(venv_python())]
        say("[1-3/5] Skipping package installation (--model-only)")
    elif args.system:
        if (sys.version_info[0], sys.version_info[1]) not in SUPPORTED:
            fail(f"--system needs this script to run on Python 3.12 or 3.13, not {platform.python_version()}.")
        python = [sys.executable]
        say(f"[1/5] Using the running interpreter, Python {platform.python_version()} (no virtual environment)")
    else:
        interpreter = find_interpreter(args.python)
        major, minor = version_of(interpreter)  # type: ignore[misc]
        say(f"[1/5] Creating .venv with Python {major}.{minor}  ({' '.join(interpreter)})")
        if VENV.exists() and version_of([str(venv_python())]) not in SUPPORTED:
            say("      removing an existing .venv built with an unsupported Python")
            shutil.rmtree(VENV)
        if not venv_python().exists():
            run(interpreter + ["-m", "venv", str(VENV)])
        python = [str(venv_python())]
    if not args.model_only:
        install_packages(python, args)

    # 4. embedding model into ./models
    online = dict(os.environ, HF_HUB_OFFLINE="0", TRANSFORMERS_OFFLINE="0")
    if args.skip_model:
        say("[4/5] Skipping the embedding model download (--skip-model)")
    else:
        say("[4/5] Downloading the embedding model into the project's models/ folder (one time, ~90 MB)")
        run(python + ["-c", (
            "from cred_support_agent.config import EMBEDDING_MODEL_NAME, MODEL_DIR; "
            "from sentence_transformers import SentenceTransformer; "
            "SentenceTransformer(EMBEDDING_MODEL_NAME).encode(['warm up']); "
            "print('      model saved to', MODEL_DIR)"
        )], env=online)

    # 5. verify offline, exactly as the project will run
    offline = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    if args.skip_model:
        say("[5/5] Verifying the installed packages import cleanly")
        run(python + ["-c", (
            "import crewai, chromadb, sentence_transformers, fastapi, autogen_agentchat, langchain_core; "
            "print('      core packages import cleanly')"
        )], env=offline)
    else:
        say("[5/5] Verifying the installation with model-hub network access disabled")
        run(python + ["-c", (
            "import os, sys, cred_support_agent.config as c; "
            "from cred_support_agent.retrieval.embeddings import backend_name; "
            "name = backend_name(); "
            "print('      Python', sys.version.split()[0], '| LLM backend:', c.LLM_BACKEND, "
            "'| telemetry disabled:', os.environ['CREWAI_DISABLE_TELEMETRY'], '| embeddings:', name); "
            "sys.exit(0 if name.startswith('sentence-transformers') else 3)"
        )], env=offline)

    exe = " ".join(python) if args.system else (r".venv\Scripts\python" if IS_WINDOWS else "./.venv/bin/python")
    say("")
    say("Setup complete. Everything from here runs offline. Next:")
    say(f"  {exe} scripts/run_all.py --reset     # every task + acceptance check")
    say(f"  {exe} -m pytest -q                   # automated tests")
    say(f"  {exe} scripts/chat.py --debug        # talk to the agent")


if __name__ == "__main__":
    main()
