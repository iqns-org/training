"""Provider-agnostic Axolotl fine-tune workflow.

Reimplements the bin/gpuai-deploy + bin/gpuai-job orchestration on top of
any :class:`Provider`: prepare data → provision → upload → remote train →
artifact gate → download → terminate. Remote steps mirror
src/remote_axolotl_run.sh (conda env, BLAS thread caps, preprocess+train,
success marker).
"""

from pathlib import Path

from .common import (
    GpuError, find_config_file, is_active, load_job, next_sequential_dir,
    require_final_artifacts, save_job,
)


class TrainingError(GpuError):
    """A failed training run that knows whether fallback is safe.

    ``keep_alive=True`` means a live instance (possibly with a trained
    model on it) must NOT be torn down or retried elsewhere — the error
    message already tells the user how to resume. ``keep_alive=False``
    means nothing valuable exists yet, so the caller may terminate the
    instance best-effort and try the next provider.
    """

    def __init__(self, message, instance_id=None, keep_alive=False):
        super().__init__(message)
        self.instance_id = instance_id
        self.keep_alive = keep_alive

REMOTE_RUN_SCRIPT = """\
set -euo pipefail
cd ~
rm -f ~/axolotl-job-success
source /root/miniconda3/etc/profile.d/conda.sh
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$OPENBLAS_NUM_THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OPENBLAS_NUM_THREADS}"
. ./fix-axolotl.sh
axolotl preprocess ./${REMOTE_CFG_NAME}
axolotl train ./${REMOTE_CFG_NAME}
touch ~/axolotl-job-success
"""

APP = "axolotl-finetuning"


def prepare_domain(root, domain):
    """Expand datasets/{domain} → tmp/{domain}.jsonl (bin/prepare-domain idea)."""
    import subprocess
    import sys

    root = Path(root)
    proc = subprocess.run(
        [sys.executable, str(root / "bin" / "prepare-domain"), domain],
        capture_output=True, text=True, cwd=root,
    )
    if proc.returncode != 0:
        raise GpuError(f"prepare-domain {domain} failed: "
                       f"{(proc.stderr or proc.stdout).strip()[:1000]}")
    final = root / "tmp" / f"{domain}.jsonl"
    if not final.is_file():
        raise GpuError(f"prepare-domain produced no {final}")
    return final


def run_training(provider, root, domain, model="SmolLM2-360M-Instruct",
                 strategy="qlora", quick=False, job_file=None, timeout=86400,
                 resume=False):
    """Full schedule-and-monitor cycle. Returns paths + instance record.

    On download/artifact failure the instance is deliberately KEPT ALIVE:
    tearing it down would destroy the trained model still on the remote.
    The raised error names the instance and the exact resume commands.

    With ``resume=True``, a live instance recorded in ``job_file`` is
    reused (no new provision, no dataset re-upload); if none is live the
    run starts fresh. Raises :class:`TrainingError` on operational failure.
    """
    root = Path(root)
    job_file = Path(job_file) if job_file else None

    cfg_name = find_config_file(root / "axolotl" / domain, model, strategy)
    if not quick:
        dataset = prepare_domain(root, domain)
    else:
        dataset = root / "tmp" / f"{domain}.jsonl"

    if job_file and job_file.is_file():
        saved = load_job(job_file) or {}
    else:
        saved = {}

    inst_id, resumed = None, False
    if resume and job_file and job_file.is_file():
        cand = (saved.get("instance", {}) or {}).get("id", "")
        if cand:
            try:
                if is_active(provider.status(cand).get("status")):
                    inst_id, resumed = cand, True
            except GpuError:
                inst_id = None
    try:
        if inst_id is None:
            if resume:
                print("no live saved instance; starting a fresh run")
            provisioned = provider.provision(app=APP, name=f"{APP}-{domain}",
                                             job_file=job_file)
            inst_id = provisioned["instance"]["id"]
        conn = provider.wait_ssh(inst_id)
        target = f"{conn['user']}@{conn['host']}"

        provider.run_remote(inst_id, "mkdir -p ~/datasets ~/datasets/prepared",
                            timeout=600)
        if not quick and not resumed:
            # On resume the dataset is already on the instance from the
            # previous attempt; re-uploading gigabytes each retry is pure waste.
            provider.upload(inst_id, str(dataset), "~/datasets/")
        provider.upload(inst_id, str(root / "axolotl" / domain / cfg_name),
                        f"~/{cfg_name}")
        for extra in ("axolotl/fix-axolotl.sh", ".env"):
            src = root / extra
            if src.is_file():
                provider.upload(inst_id, str(src), f"~/{Path(extra).name}")
        provider.run_remote(inst_id, REMOTE_RUN_SCRIPT,
                            env={"REMOTE_CFG_NAME": cfg_name}, timeout=timeout)

        marker = provider.run_remote(
            inst_id, "test -f ~/axolotl-job-success && test -d ~/outputs "
                     "&& find ~/outputs -mindepth 1 -maxdepth 3 -type f "
                     "| head -n 1 | grep -q .", timeout=600)
        if marker.get("rc") != 0:
            raise TrainingError("remote training failed or produced no "
                                "artifacts (instance kept alive for "
                                "inspection)",
                                instance_id=inst_id, keep_alive=True)

        local_dir = next_sequential_dir(root / "tmp", f"{domain}-model")
        local_dir.mkdir(parents=True, exist_ok=True)
        try:
            provider.download(inst_id, "~/outputs/", str(local_dir) + "/")
            require_final_artifacts(local_dir)
        except GpuError as e:
            raise TrainingError(
                f"{e}. Instance {inst_id} KEPT ALIVE with the trained model "
                f"still on it — resume with `bin/gpu download {domain}` then "
                f"`bin/gpu terminate {inst_id}`",
                instance_id=inst_id, keep_alive=True) from e
    except TrainingError:
        raise
    except GpuError as e:
        # Pre-training failure: nothing valuable exists yet, so fallback
        # to another provider (with best-effort teardown) is safe.
        raise TrainingError(str(e), instance_id=inst_id,
                            keep_alive=False) from e
    provider.terminate(inst_id)
    if job_file and job_file.is_file():
        job_file.unlink(missing_ok=True)
    _ = saved, target  # (kept for future resume metadata)
    return {"instance": inst_id, "config": cfg_name, "model_dir": str(local_dir)}


def run_with_fallback(named, root, domain, **kwargs):
    """Try providers in order; fall back on safe failures.

    ``named`` is a list of ``(provider_name, provider, job_path_or_None)``.
    A provider that fails *before* anything valuable exists gets its
    instance terminated best-effort and the next provider is tried. A
    failure flagged ``keep_alive`` (live instance, possibly with a trained
    model) aborts immediately — retraining elsewhere while billing two
    instances would be worse than asking the user to resume manually.
    """
    errors = {}
    for pname, provider, job in named:
        try:
            res = run_training(provider, root, domain, job_file=job, **kwargs)
            res["provider"] = pname
            return res
        except TrainingError as e:
            errors[pname] = str(e)
            if e.keep_alive:
                raise
            if e.instance_id:
                try:
                    provider.terminate(e.instance_id)
                except GpuError:
                    pass
        except GpuError as e:
            errors[pname] = str(e)
    raise GpuError("all providers failed: "
                   + "; ".join(f"{k}: {v}" for k, v in errors.items()))


def download_artifacts(provider, root, domain, instance_id=None, out=None):
    """Fetch ~/outputs → outputs/<domain> (bin/gpuai-download idea)."""
    root = Path(root)
    if instance_id is None:
        infos = sorted((root / "tmp").glob(f"gpu_*_{APP}_{domain}.job"))
        if not infos:
            raise GpuError(f"no saved job for domain {domain!r}")
        saved = load_job(infos[-1]) or {}
        instance_id = saved.get("instance", {}).get("id", "")
        if not instance_id:
            raise GpuError(f"saved job for {domain!r} has no instance id")
    out = Path(out) if out else root / "outputs" / domain
    out.mkdir(parents=True, exist_ok=True)
    try:
        provider.download(instance_id, "~/outputs/", str(out) + "/")
        require_final_artifacts(out)
    except GpuError as e:
        raise GpuError(
            f"{e}. Instance {instance_id} KEPT ALIVE — retry with "
            f"`bin/gpu download {domain}`") from e
    return {"instance": instance_id, "out": str(out)}
