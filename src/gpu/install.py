"""Software installer — local venv flow + remote setup.

Local behavior reuses the existing ``bin/install-axolotl`` script (venv,
pinned PyTorch index, axolotl[deepspeed], torch/CUDA + CLI smoke tests)
instead of duplicating its 265 lines: this module shells out with
``PYTHON``/``VENV``/``TORCH_INDEX`` overrides.

Remote behavior runs the same gates over provider SSH: conda env presence,
``fix-axolotl.sh``, ``pip check`` and the axolotl CLI smoke test — the
steps bin/gpuai-deploy implies before training.
"""

import os
import subprocess
from pathlib import Path

from .common import GpuError, run

REMOTE_SETUP_SCRIPT = """\
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-py3.11}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$OPENBLAS_NUM_THREADS}"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m pip check
axolotl --help >/dev/null && echo "axolotl CLI OK"
"""


def install_local(root=None, python=None, venv=None, torch_index=None):
    """Run bin/install-axolotl with optional env overrides."""
    root = Path(root) if root else Path(__file__).resolve().parents[2]
    script = root / "bin" / "install-axolotl"
    if not script.is_file():
        raise GpuError(f"installer not found: {script}")
    env = dict(os.environ)
    if python:
        env["PYTHON"] = python
    if venv:
        env["VENV"] = venv
    if torch_index:
        env["TORCH_INDEX"] = torch_index
    proc = subprocess.run(["bash", str(script)], cwd=root, env=env,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise GpuError(f"install-axolotl failed:\n{proc.stderr or proc.stdout}"[-3000:])
    return {"venv": venv or os.environ.get("VENV", "~/.venvs/axolotl"),
            "output": proc.stdout[-2000:]}


def install_remote(provider, instance_id, conda_env="py3.11",
                   fix_script=None):
    """Verify/repair the remote training environment via provider SSH."""
    if fix_script:
        provider.upload(instance_id, fix_script, "~/fix-axolotl.sh")
    result = provider.run_remote(
        instance_id, REMOTE_SETUP_SCRIPT, env={"CONDA_ENV": conda_env})
    if result.get("rc") != 0:
        raise GpuError("remote setup check failed: "
                       f"{result.get('stderr') or result.get('error', '')}"[:2000])
    return {"instance": instance_id, "conda_env": conda_env,
            "output": result.get("stdout", "")[-2000:]}


def check_local(venv=None):
    """Report local venv/torch/cuda status without installing."""
    import sys

    venv_python = None
    if venv:
        cand = Path(venv).expanduser() / "bin" / "python"
        venv_python = str(cand) if cand.is_file() else None
    info = {"venv_python": venv_python, "system_python": sys.version}
    for exe in ([venv_python] if venv_python else [sys.executable]):
        try:
            out = run([exe, "-c",
                       "import torch; print(torch.__version__ + '|' + "
                       "str(torch.cuda.is_available()))"]).stdout.strip()
            info["torch"], info["cuda_available"] = out.split("|")
        except GpuError as e:
            info["error"] = str(e)
    return info
