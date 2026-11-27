"""Job finalization: artifact gate + manifest + presigned uploads (SPEC §5).

Runs after the training command exits. The gate refuses known-bad output
states (leftover checkpoints as a final set, missing adapter files) so the
orchestrator never has to trust a bare exit code.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

from . import __version__
from .transfer import download, sha256_file, upload

REQUIRED_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
)
EXCLUDE_DIRS = {"checkpoint-", "runs", "logs", "prepared", "hf-cache"}


def _excluded(name: str, exclude: tuple | list | None = None) -> bool:
    return any(part.startswith(p) for part in Path(name).parts
               for p in (exclude if exclude is not None else EXCLUDE_DIRS))


def gate(outputs_dir: Path, gate_spec: dict | None = None) -> tuple[bool, list[str]]:
    """(ok, problems). Required final files present; checkpoint-* leftovers
    are NORMAL (axolotl saves_per_epoch) — excluded from pack, never fatal.
    Stage jobs override the required set (ONNX dirs have no adapter files).
    Entries may be PATTERNS ("model*.safetensors" — full FT shards;
    "*/config.json" — one required file per co-tenant subdir): matched
    with fnmatch against every relative path under outputs_dir. A plain
    name stays an exact root-file check (back-compat with schema-2 law)."""
    import fnmatch
    required = (gate_spec or {}).get("required_files", REQUIRED_FILES)
    problems = []
    if not outputs_dir.is_dir():
        return False, [f"outputs dir missing: {outputs_dir}"]
    rels = None
    for f in required:
        if "*" in f or "?" in f:
            if rels is None:
                rels = [str(p.relative_to(outputs_dir))
                        for p in outputs_dir.rglob("*") if p.is_file()]
            if not any(fnmatch.fnmatch(r, f) for r in rels):
                problems.append(f"required artifact missing: {f}")
        elif not (outputs_dir / f).is_file():
            problems.append(f"required artifact missing: {f}")
    if not any(p.is_file() for p in outputs_dir.rglob("*")):
        problems.append("outputs dir has no files")
    return (not problems), problems


def pack(outputs_dir: Path, dest: Path,
         exclude: tuple | list | None = None) -> Path:
    """tar.gz of the artifact set minus excluded dirs."""
    with tarfile.open(dest, "w:gz") as tf:
        for p in sorted(outputs_dir.rglob("*")):
            if p.is_file() and not _excluded(str(p.relative_to(outputs_dir)),
                                             exclude):
                tf.add(p, arcname=str(p.relative_to(outputs_dir)))
    return dest


def gpu_info() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        return {"gpus": out.splitlines()}
    except Exception:                                   # noqa: BLE001
        return {"gpus": []}


def _log_summary(log_path) -> dict:
    """Terminal metrics from the job's own log: HF Trainer emits one dict
    line per log step plus a final summary dict ({'train_loss':...,
    'train_runtime':..., 'train_samples_per_second':...}). Scraping them
    here makes manifest.training self-describing even if the live
    heartbeat missed the last cadence (orchestrator-side parse_runlog does
    the same join for historical runs — SPEC 14c)."""
    import ast
    out: dict = {}
    wanted = ("loss", "epoch", "train_loss", "train_runtime",
              "train_samples_per_second", "train_steps_per_second",
              "eval_loss", "eval_samples_per_second",
              "holdout_loss", "rows_scored", "rows_total",
              "merge_applied_pairs", "merge_expected_pairs",
              "merge_changed_tensors")
    try:
        text = Path(log_path).read_text(errors="replace")
    except OSError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if "'loss'" not in line and "'train_loss'" not in line:
            continue
        try:
            d = ast.literal_eval(line)
        except (ValueError, SyntaxError):
            continue
        if isinstance(d, dict):
            for k in wanted:
                if k in d:
                    out[k] = d[k]
    return out


def build_manifest(run_id: str, outputs_dir: Path, state: dict,
                   status: str, problems: list[str] | None = None,
                   exclude: tuple | list | None = None,
                   log_path: Path | None = None,
                   attestation: dict | None = None) -> dict:
    files = {}
    if outputs_dir.is_dir():
        for p in outputs_dir.rglob("*"):
            rel = str(p.relative_to(outputs_dir))
            if p.is_file() and not _excluded(rel, exclude):
                files[rel] = {"size": p.stat().st_size, "sha256": sha256_file(p)}
    training = {k: state.get(k) for k in
                ("state", "step", "total", "epoch", "loss", "exit_code")}
    if state.get("deadman"):
        # SPEC 13an: this terminal was forced by orchestrator silence, not
        # by the recipe finishing. Stamp survives even when the gate found
        # complete artifacts (fire between saves) — a partial round must
        # never masquerade as a clean one.
        training["deadman"] = True
    summ = _log_summary(log_path) if log_path else {}
    if summ:
        training["summary"] = summ
        # heartbeat can lag the log's final dict line; prefer the log truth
        if training.get("loss") is None and "loss" in summ:
            training["loss"] = summ["loss"]
        if training.get("epoch") is None and "epoch" in summ:
            training["epoch"] = summ["epoch"]
    man = {
        "schema": 2, "run_id": run_id, "status": status,
        "agent_version": __version__,
        "image_tag": os.environ.get("IMAGE_TAG", ""),
        "finished_at": time.time(),
        "started_at": state.get("started_at"),
        "training": training,
        "hardware": gpu_info(),
        "artifacts": files,
        "problems": problems or [],
    }
    if attestation is not None:
        man["attestation"] = attestation
    return man


def _vol_write(src: Path, volume: dict, name: str) -> None:
    """Volume lane: copy src into <mount>/<prefix>/name atomically
    (tmp+rename) so a concurrent S3-API reader never sees a partial file."""
    dest_dir = Path(volume["mount"]) / volume["prefix"]
    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp = dest_dir / (name + ".part")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dest_dir / name)


def finalize(run_id: str, outputs_dir: Path, state: dict, log_path: Path,
             urls: dict[str, str], volume: dict | None = None,
             gate_spec: dict | None = None,
             attestation: dict | None = None) -> dict:
    """Gate → pack → publish artifacts/manifest/log via the payload's lane
    (presigned URLs and/or mounted volume). Manifest LAST = completion
    signal. gate_spec overrides required files/excludes for stage jobs.
    Returns the manifest."""
    ok, problems = gate(outputs_dir, gate_spec)
    exclude = (gate_spec or {}).get("exclude_dirs", EXCLUDE_DIRS)
    manifest = build_manifest(run_id, outputs_dir, state,
                              "succeeded" if ok else "failed", problems,
                              exclude, log_path=log_path,
                              attestation=attestation)
    mpath = Path(outputs_dir).parent / "manifest.json"
    mpath.write_text(json.dumps(manifest, indent=1))

    tar = None
    if ok:
        tar = pack(outputs_dir, mpath.parent / "artifacts.tar.gz", exclude)
        manifest["artifacts_tar_sha256"] = sha256_file(tar)
        mpath.write_text(json.dumps(manifest, indent=1))
        if urls.get("artifacts_put"):
            upload(tar, urls["artifacts_put"])
        if volume:
            _vol_write(tar, volume, "artifacts.tar.gz")
    if volume:                                        # log before manifest
        if log_path.is_file():
            _vol_write(log_path, volume, "run.log")
    if urls.get("log_put") and log_path.is_file():
        upload(log_path, urls["log_put"])
    if urls.get("manifest_put"):                      # last = completion signal
        upload(mpath, urls["manifest_put"])
    if volume:
        _vol_write(mpath, volume, "manifest.json")
    (Path(outputs_dir).parent / "done.json").write_text(
        json.dumps({"run_id": run_id, "status": manifest["status"]}))
    return manifest
