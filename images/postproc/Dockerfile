# postproc — quantize/export/merge tooling with the trainer agent inside.
# Same boot contract as the trainer image (JOB_PAYLOAD_PATH /
# JOB_PAYLOAD_URL, volume+gate semantics), so stage jobs ride the SAME
# lanes. arm64-native (local K8s Jobs on pi00/nano0), pure-python deps.
FROM python:3.12-slim

WORKDIR /work
COPY postproc/ /opt/postproc/
COPY agent/trainer_agent/ /opt/agent/trainer_agent/

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu 'torch==2.6.*' \
 && pip install --no-cache-dir \
    "optimum[exporters]<2" onnx onnxruntime \
    "transformers>=4.51,<4.56" peft accelerate \
    fastapi uvicorn httpx

ENV PYTHONPATH=/opt/agent \
    AGENT_WORKDIR=/work \
    MCP_TRAINER_PORT_UNUSED=1

EXPOSE 8000
CMD ["python", "-m", "trainer_agent.app"]
