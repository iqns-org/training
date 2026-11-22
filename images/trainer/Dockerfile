# Generic trainer image (SPEC §3/§10): CUDA + pinned axolotl + in-pod agent.
# Canonical copy lands in the cluster Forgejo registry; a public mirror backs
# cloud pulls (see scripts/sync-image.sh). Nothing is installed at pod boot.
#
# Base = official axolotl image (Docker Hub, verified publisher "axolotlai").
# py3.12 + CUDA 13.0 + torch 2.13 chosen for a conservative ML-ecosystem fit
# (bitsandbytes/deepspeed wheels); py3.14 tags exist but are too fresh.
# Pinned by digest for reproducibility; bump deliberately, never on rebuild.
ARG AXOLOTL_IMAGE=axolotlai/axolotl:main-py3.12-cu130-2.13.0@sha256:dd4a17340e15ef380f5969be22737c0603c2aa116282dbf0c4978df2bf9d88dc
FROM ${AXOLOTL_IMAGE}

# Agent deps live in their OWN venv: the axolotl base is a uv-managed venv
# without pip, and fastapi/pydantic must never perturb the training env.
RUN set -eux; \
    if command -v uv >/dev/null 2>&1; then \
        uv venv /opt/agent-venv && \
        uv pip install --python /opt/agent-venv/bin/python \
            "fastapi>=0.115" "uvicorn[standard]>=0.30" "httpx>=0.27" "pydantic>=2.7"; \
    else \
        python -m venv /opt/agent-venv && \
        /opt/agent-venv/bin/pip install --no-cache-dir \
            "fastapi>=0.115" "uvicorn[standard]>=0.30" "httpx>=0.27" "pydantic>=2.7"; \
    fi

COPY agent/trainer_agent /opt/trainer_agent/trainer_agent
ENV PYTHONPATH=/opt/trainer_agent \
    PYTHONUNBUFFERED=1 \
    AGENT_PORT=8000 \
    AGENT_WORKDIR=/work \
    # BLAS thread caps proven necessary on thread-constrained GPU vendors
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

# HF cache: mount a RunPod network volume / provider disk here (P2) so
# re-runs skip base-model downloads.
ENV HF_HOME=/work/hf-cache

WORKDIR /work
EXPOSE 8000

# Single-GPU training fits one agent process supervising one job (SPEC §5).
# Interpreter is the agent venv; PYTHONPATH resolves trainer_agent.
ENTRYPOINT ["/opt/agent-venv/bin/python", "-m", "trainer_agent.app"]
