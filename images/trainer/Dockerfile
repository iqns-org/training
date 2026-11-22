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

# Agent deps live in an ISOLATED lib dir on PYTHONPATH. The axolotl base is
# uv-managed (venv ships WITHOUT pip; first build proved `python -m pip`
# dead), and fastapi/uvicorn must never touch the training venv — so install
# into /opt/agent-libs with uv when present (VIRTUAL_ENV unset for the call),
# else plain pip, else fail loudly.
RUN set -u; \
    UVBIN="$(command -v uv || true)"; \
    [ -n "$UVBIN" ] || { [ -x /usr/local/bin/uv ] && UVBIN=/usr/local/bin/uv; }; \
    if [ -n "$UVBIN" ]; then \
        echo "agent-deps: uv at $UVBIN"; \
        env -u VIRTUAL_ENV "$UVBIN" pip install --target /opt/agent-libs \
            --python "$(command -v python)" \
            "fastapi>=0.115" "uvicorn[standard]>=0.30" "httpx>=0.27"; \
    else \
        echo "agent-deps: no uv, trying pip"; \
        python -m pip install --target /opt/agent-libs --ignore-installed \
            "fastapi>=0.115" "uvicorn[standard]>=0.30" "httpx>=0.27"; \
    fi

COPY agent/trainer_agent /opt/trainer_agent/trainer_agent
ENV PYTHONPATH=/opt/agent-libs:/opt/trainer_agent \
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
# Base interpreter; PYTHONPATH carries agent code + isolated deps.
ENTRYPOINT ["python", "-m", "trainer_agent.app"]
