#!/usr/bin/env python3
"""P0 smoke with live evidence lane: cloudflared quick-tunnel -> local
capture server <- pod self-report (PUT). Proves: public ghcr pull, CUDA
image boot, agent serves /healthz + /v1/info, pod can push to an internet
URL (the lane shape P1's S3 return path will use). ALWAYS terminates pods."""
import json, os, re, subprocess, sys, threading, time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, os.getcwd())
from orchestrator.config import load_dotenv
from orchestrator.providers.runpod import RunpodProvider, normalize_pod

load_dotenv()
CAP = Path("/tmp/captures"); CAP.mkdir(exist_ok=True)
for old in CAP.glob("*"):
    old.unlink()
BIN, CF = "/tmp/cloudflared", "/tmp/cf.log"
prov = RunpodProvider()
PIDS = []
IMAGE = os.environ.get("TRAINER_IMAGE") or ("ghcr.io/iqns-org/trainer"
         "@sha256:d1f766d24d62274e770fd8913c3ffe612b87c115d0cd59c318ca60d5c5d8bd36")

SCRIPT = """python -m trainer_agent.app &
AP=$!
sleep 8
python - <<'PY'
import json,os,time,httpx
h={'ok':False}
for _ in range(20):
    try:
        h=httpx.get('http://localhost:8000/healthz',timeout=2).json()
        break
    except Exception:
        time.sleep(1)
info={}
try:
    info=httpx.get('http://localhost:8000/v1/info',
                   headers={'Authorization':'Bearer smoke'},timeout=15).json()
except Exception:
    pass
rep={'healthz':h,'info':info,'ts':time.time(),'node':os.uname().nodename}
print(json.dumps(rep))
try:
    httpx.put(os.environ['REPORT_URL'],json=rep,timeout=25)
    print('REPORTED')
except Exception as e:
    print('REPORT-FAIL',e)
PY
kill $AP 2>/dev/null
sleep 3
"""

def cleanup():
    for pid in PIDS:
        try:
            print(f"** terminate {pid}", flush=True)
            prov.terminate(pid)
        except Exception as e:
            print(f"MANUAL KILL NEEDED {pid}: {e}", flush=True)

def capture_server():
    class H(BaseHTTPRequestHandler):
        def _store(self):
            n = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(n) if n else b""
            (CAP / f"{int(time.time()*1000)}-put.json").write_bytes(body or b"{}")
            self.send_response(200); self.end_headers(); self.wfile.write(b'{"ok":true}')
        do_PUT = do_POST = do_GET = _store
        def log_message(self, *a): pass
    HTTPServer(("127.0.0.1", 8080), H).serve_forever()

def main():
    threading.Thread(target=capture_server, daemon=True).start()
    if not os.path.exists(BIN):
        subprocess.run(["curl", "-sL", "-o", BIN,
                        "https://github.com/cloudflare/cloudflared/releases/"
                        "latest/download/cloudflared-linux-amd64"], check=True)
        os.chmod(BIN, 0o755)
    cf = subprocess.Popen([BIN, "tunnel", "--url", "http://localhost:8080",
                           "--no-autoupdate"], stderr=open(CF, "w"),
                          stdout=subprocess.DEVNULL)
    pub = ""
    for _ in range(60):
        m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", open(CF).read())
        if m:
            pub = m.group(0); break
        time.sleep(1)
    if not pub:
        print("NO TUNNEL URL"); cleanup(); cf.terminate(); return 1
    print(f"** tunnel: {pub}", flush=True)

    # ladder: cheapest across both tiers, up to 10 attempts (community flags
    # lie under churn; create is the only ground truth)
    offers = prov.catalog()
    inst = None
    for o in offers[:10]:
        body = {"name": "trainer-smoke", "image": IMAGE,
                "gpu": {"id": o.gpu_id, "count": 1},
                "cloud": "SECURE" if o.secure else "COMMUNITY",
                "ports": ["8000/http"], "disk": 30,
                "env": {"REPORT_URL": f"{pub}/boot", "AGENT_TOKEN": "smoke",
                        "IMAGE_TAG": "sha-a27a4f8"},
                "entrypoint": ["sh", "-c", SCRIPT]}
        try:
            inst = normalize_pod(prov._req("POST", "/v2/pods", json=body))
            tier = "secure" if o.secure else "community"
            print(f"** launched {inst.id} on {o.gpu_id} ${o.price_hr}/hr "
                  f"({tier})", flush=True)
            break
        except Exception as e:
            print(f"   {o.gpu_id}: {str(e)[:110]}", flush=True)
    if not inst:
        print("LAUNCH FAILED"); cf.terminate(); return 1
    PIDS.append(inst.id)

    deadline = time.time() + 600
    while time.time() < deadline:
        got = [f for f in CAP.glob("*-put.json") if f.stat().st_size > 2]
        if got:
            rep = json.loads(sorted(got)[-1].read_text())
            health, info = rep.get("healthz", {}), rep.get("info", {})
            print("** REPORT:", json.dumps(rep, indent=1)[:900], flush=True)
            print(f"** VERDICT healthz.ok={health.get('ok')} "
                  f"agent={info.get('version')} py={info.get('python')} "
                  f"gpu={info.get('gpu')!r}", flush=True)
            cleanup(); cf.terminate()
            return 0 if health.get("ok") is True else 2
        try:
            st = prov.poll(inst.id).status
            if st in {"terminated", "failed", "error", "stopped"}:
                print(f"pod died: {st}", flush=True)
                break
        except Exception:
            pass
        time.sleep(12)
    print("TIMEOUT waiting for report", flush=True)
    cleanup(); cf.terminate(); return 1

if __name__ == "__main__":
    sys.exit(main())
