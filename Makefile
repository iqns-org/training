# quadra/trainer image lane — SPEC §3/§10.
# Trainer CUDA image builds in GitHub Actions (on tag v*, reached via the
# Forgejo→GitHub repo mirror) — NOT here: a ~20 GB CUDA base does not fit on
# this workstation's disks. podman locally is for small images only.
PY := .venv/bin/python
PIP := .venv/bin/pip

CANONICAL  ?= pi00.lan:30080/quadra/trainer
TAG        ?= dev

.venv/bin/python:
	python3 -m venv .venv || uv venv .venv
	$(PIP) install -q -e ".[dev]"

.PHONY: setup test live catalog doctor tag sync agent-run

test: setup
	$(PY) -m pytest -q

live: setup            ## real-API tests (needs RUNPOD_API_KEY in .env)
	RUNPOD_LIVE_TESTS=1 $(PY) -m pytest -q -k live

catalog: setup
	.venv/bin/qtrain catalog

doctor: setup
	.venv/bin/qtrain doctor

## Cut a release: tag + push to Forgejo. The Forgejo→GitHub mirror feeds the
## commit to Actions, which builds linux/amd64 and pushes ghcr.io (cloud
## mirror). Watch: gh run list -R iqns-org/training
tag:
	@git tag -a "$(TAG)" -m "trainer $(TAG)" 2>/dev/null || echo "tag $(TAG) exists"
	@git push origin "$(TAG)"

## Inside the cluster/workstation: pull CI-built tags back into the canonical
## Forgejo registry at the SAME digest (needs skopeo + forgejo push creds).
sync:
	scripts/sync-image.sh $(TAG)

agent-run: setup       ## run the in-pod agent locally (dev, no GPU)
	AGENT_WORKDIR=./tmp/work $(PY) -m trainer_agent.app
