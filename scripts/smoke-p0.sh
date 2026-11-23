#!/usr/bin/env bash
# P0 smoke: trainer image on a live RunPod GPU, agent verified via ssh.direct
# (RunPod v2 gives no port mappings for pods). Pod is ALWAYS terminated.
set -u
cd /media/me/developer/cogs/training
set -a; . ./.env; set +a
TOK=$(openssl rand -hex 12)
PID=""
cleanup() {
  if [ -n "$PID" ]; then
    echo "** terminate $PID"
    .venv/bin/qtrain terminate "$PID" || echo "MANUAL KILL NEEDED: $PID"
  fi
  .venv/bin/qtrain doctor && echo "doctor clean"
}
trap cleanup EXIT

echo "** launch"
out=$(.venv/bin/qtrain launch --ssh --record smoke-p0 \
  --image "ghcr.io/iqns-org/trainer@sha256:d1f766d24d62274e770fd8913c3ffe612b87c115d0cd59c318ca60d5c5d8bd36" \
  --gpu "NVIDIA GeForce RTX 3070" --cloud COMMUNITY --name trainer-smoke \
  --env AGENT_TOKEN="$TOK" --env IMAGE_TAG=sha-a27a4f8) || { echo "$out"; exit 1; }
echo "$out"
PID=$(echo "$out" | python3 -c "import json,sys;print(json.load(sys.stdin)['instance'])")

echo "** wait for RUNNING (image pull ~8 GB)"
for i in $(seq 1 60); do
  st=$(.venv/bin/qtrain status "$PID" 2>/dev/null | python3 -c "import json,sys;print(json.load(sys.stdin)['status'])" 2>/dev/null || echo unknown)
  echo "  [$i] $st"
  [ "$st" = "running" ] && break
  case "$st" in terminated|failed|error|stopped) echo "DIED: $st"; exit 1;; esac
  sleep 15
done
[ "$st" = "running" ] || { echo "timeout waiting"; exit 1; }

echo "** ssh.direct + in-pod agent checks"
conn=$(.venv/bin/python -c "
from orchestrator.providers.runpod import RunpodProvider
p = RunpodProvider(); c = p.ssh_direct('$PID')
print(f\"{c['user']}|{c['host']}|{c['port']}\")" 2>&1) || { echo "ssh_direct failed: $conn"; exit 1; }
IFS='|' read -r U H P <<<"$conn"
echo "  direct: $U@$H:$P"
ssh -o LogLevel=ERROR -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 -p "$P" "$U@$H" \
    "echo '### healthz:'; curl -sf localhost:8000/healthz; echo;
     echo '### info:'; curl -sf -H 'Authorization: Bearer $TOK' localhost:8000/v1/info; echo;
     echo '### gpu:'; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"
rc=$?
echo "** in-pod checks rc=$rc"
exit $rc
