# postproc-gpu — CUDA side of post-processing (AWQ/GPTQ calibration).
# amd64: cloud stage pods (RunPod). Orin-native builds deferred (the
# PyPI cu124 wheels are server-GPU; Jetson needs NVIDIA's own index).
FROM python:3.12-slim

WORKDIR /work
COPY postproc/ /opt/postproc/
COPY agent/trainer_agent/ /opt/agent/trainer_agent/

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124 \
 && pip install --no-cache-dir \
    "transformers<5" accelerate datasets safetensors sentencepiece protobuf \
    llmcompressor fastapi uvicorn httpx

ENV PYTHONPATH=/opt/agent \
    AGENT_WORKDIR=/work

EXPOSE 8000
CMD ["python", "-m", "trainer_agent.app"]
