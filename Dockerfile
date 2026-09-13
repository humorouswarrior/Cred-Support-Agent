# Runs the Cred Domain Support Agent identically on any machine with Docker:
# Linux, macOS (Intel or Apple Silicon) and Windows.
#
#   docker build -t cred-support-agent .
#   docker run --rm --network none cred-support-agent                          # every task + acceptance check, no network
#   docker run --rm --network none cred-support-agent python -m pytest -q      # automated tests
#   docker run --rm -p 8000:8000 cred-support-agent \
#       uvicorn cred_support_agent.api:app --host 0.0.0.0 --port 8000         # the API on http://localhost:8000
#   docker run --rm -it cred-support-agent python scripts/chat.py --debug      # terminal chat
#
# Keep the regenerated transcripts on the host (Linux: add --user so files are yours):
#   docker run --rm --network none --user "$(id -u):$(id -g)" \
#       -v "$PWD/transcripts:/app/transcripts" cred-support-agent
#
# Python 3.13 instead of 3.12:  docker build --build-arg PYTHON_VERSION=3.13 -t cred-support-agent .

ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first, so they are cached as a layer and only rebuilt when the
# requirement files change. bootstrap.py installs CPU-only PyTorch and the locked
# dependency set into the image's own Python (--system: no virtualenv needed).
COPY requirements.txt constraints.txt bootstrap.py ./
RUN python bootstrap.py --system --skip-model

# The project itself, then the embedding model baked into the image, so the
# container runs with no network at all.
COPY . .
RUN python bootstrap.py --system --model-only

# Let the container run as any user, so on Linux
#   docker run --user "$(id -u):$(id -g)" -v "$PWD/transcripts:/app/transcripts" ...
# writes files owned by you rather than by root. An arbitrary user has no home
# directory and cannot write to root-owned /app, so give it a writable HOME and make
# the directories the project writes to writable by all users.
ENV HOME=/tmp/home
RUN mkdir -p /tmp/home /app/artifacts \
    && chmod -R a+rwX /tmp/home /app/artifacts /app/transcripts

EXPOSE 8000
CMD ["python", "scripts/run_all.py", "--reset"]
