# mcp-trainer — cluster MCP server image (SPEC §4 mcp layer).
# Pure-python control plane: orchestrator + corpora expansion tooling.
# Built for linux/arm64 by .github/workflows/mcp-trainer-image.yml.
FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY orchestrator/ ./orchestrator/
COPY agent/ ./agent/
COPY src/ ./src/
COPY bin/ ./bin/
COPY axolotl/ ./axolotl/
COPY mcp_trainer/ ./mcp_trainer/

RUN pip install --no-cache-dir "httpx>=0.27" "pydantic>=2.7" "PyYAML>=6" \
    "fastmcp>=2.0" "uvicorn>=0.30"

ENV PYTHONPATH=/app \
    MCP_TRAINER_PORT=3022 \
    CORPORA_DIR=/data/corpora \
    PYTHONUNBUFFERED=1

EXPOSE 3022
CMD ["python", "-m", "mcp_trainer"]
