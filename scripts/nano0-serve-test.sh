#!/usr/bin/env bash
# nano0 one-shot serve validation for a REGISTERED model (SPEC 13v).
# The Orin hosts one vLLM engine at a time (~6 GiB of 7.7 GiB unified
# memory); the standing engine is the quadra/vllm-qwen3-05 deployment.
# This pauses it, boots YOUR artifact with the SAME image the cluster
# already caches (vllm/vllm-openai:latest — just changed config: model
# path, served name, Qwen-specific parsers dropped), curls a completion,
# then restores qwen regardless of the outcome.
#
#   MODEL=queer-360m-awq ./scripts/nano0-serve-test.sh
#   MODEL=foo SERVED=foo MAXLEN=8192 ./scripts/nano0-serve-test.sh
#   PROMPT="..." ./scripts/nano0-serve-test.sh
#
# Prereqs: ssh me@pi00.lan kubectl works; the artifact is registered on
# nano0 at /home/me/models/<MODEL> (qtrain register).
set -euo pipefail

MODEL="${MODEL:?set MODEL to the /home/me/models/<dir> name}"
SERVED="${SERVED:-$MODEL}"
MAXLEN="${MAXLEN:-4096}"
GPUMEM="${GPUMEM:-0.5}"
PORT="${PORT:-8000}"
PROMPT="${PROMPT:-Describe in one sentence what makes a community kitchen welcoming.}"
K="ssh me@pi00.lan k3s kubectl"

restore() {
  $K -n quadra delete pod serve-test --ignore-not-found --wait=false || true
  $K -n quadra scale deploy vllm-qwen3-05 --replicas=1 || true
  echo "[serve-test] qwen restore issued (verify: $K -n quadra get pods)"
}
trap restore EXIT

echo "[serve-test] pausing qwen..."
$K -n quadra scale deploy vllm-qwen3-05 --replicas=0
for _ in $(seq 1 20); do
  rdy=$($K -n quadra get deploy vllm-qwen3-05 -o jsonpath='{.status.readyReplicas}' 2>/dev/null || true)
  [ -z "${rdy:-}" ] && break; sleep 6
done
sleep 8
ssh me@nano0.lan 'free -m | head -2'

echo "[serve-test] booting $SERVED on the cluster-cached image..."
$K apply -f - <<YAML
apiVersion: v1
kind: Pod
metadata:
  name: serve-test
  namespace: quadra
spec:
  nodeName: nano0
  restartPolicy: Never
  runtimeClassName: nvidia
  hostNetwork: true
  dnsPolicy: ClusterFirstWithHostNet
  terminationGracePeriodSeconds: 0
  automountServiceAccountToken: false
  tolerations:
  - {key: nano-worker, operator: Exists, effect: NoSchedule}
  containers:
  - name: vllm
    image: vllm/vllm-openai:latest
    imagePullPolicy: IfNotPresent
    args:
    - /models/$MODEL
    - --served-model-name=$SERVED
    - --max-model-len=$MAXLEN
    - --gpu-memory-utilization=$GPUMEM
    - --max-num-seqs=8
    - --allowed-origins=["*"]
    env: [{name: VLLM_PORT, value: "$PORT"}]
    volumeMounts:
    - {name: models, mountPath: /models}
    - {name: hf-cache, mountPath: /root/.cache/huggingface}
    - {name: vllm-cache, mountPath: /root/.cache/vllm}
    resources:
      requests: {cpu: 500m, memory: 1Gi}
      limits: {cpu: "4", memory: 5Gi}
    startupProbe:
      httpGet: {path: /health, port: $PORT}
      failureThreshold: 120
      periodSeconds: 5
    readinessProbe:
      httpGet: {path: /health, port: $PORT}
      periodSeconds: 5
      timeoutSeconds: 3
  volumes:
  - name: models
    hostPath: {path: /home/me/models, type: Directory}
  - name: hf-cache
    hostPath: {path: /home/me/.cache/huggingface, type: Directory}
  - name: vllm-cache
    hostPath: {path: /home/me/.cache/vllm, type: DirectoryOrCreate}
YAML

$K -n quadra wait --for=condition=Ready pod/serve-test --timeout=660s
curl -s -m 8 "http://nano0.lan:$PORT/v1/models" | head -c 300; echo
echo "=== completion ==="
curl -s -m 60 "http://nano0.lan:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\": \"$SERVED\", \"messages\": [{\"role\": \"user\", \"content\": \"$PROMPT\"}], \"max_tokens\": 60, \"temperature\": 0.6}" \
  | head -c 900; echo

# restore happens via trap (delete pod + scale back)
echo "[serve-test] validation done; restoring qwen"
