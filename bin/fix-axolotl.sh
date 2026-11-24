#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="py3.11"

echo "=================================================="
echo " Axolotl / PyTorch CUDA 13 environment repair"
echo "=================================================="

# --------------------------------------------------
# 1. Conda
# --------------------------------------------------

source /root/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_NAME"

# Guard against OpenBLAS creating 64 worker threads in a constrained GPU.ai sandbox.
# This has to be set before any Python import that touches NumPy/SciPy/transformers.
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$OPENBLAS_NUM_THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OPENBLAS_NUM_THREADS}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$OPENBLAS_NUM_THREADS}"
export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-$OPENBLAS_NUM_THREADS}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-$OPENBLAS_NUM_THREADS}"

echo "[0/7] BLAS thread budget"
echo "OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "MKL_NUM_THREADS=$MKL_NUM_THREADS"

echo "ulimit -u: $(ulimit -u 2>/dev/null || echo 'unlimited')"

CONDA_PREFIX="$(conda info --base)/envs/$ENV_NAME"
CUDA13_LIB="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cu13/lib"

# DeepSpeed expects CUDA_HOME/bin/nvcc to exist. The pip CUDA runtime may
# install the libraries under site-packages while the actual compiler lives in
# the system toolkit directories, so we must detect both places.
resolve_nvcc() {
    local candidates=(
        "$CONDA_PREFIX/bin/nvcc"
        "$CONDA_PREFIX/../bin/nvcc"
        /usr/local/cuda/bin/nvcc
        /usr/local/cuda-12.8/bin/nvcc
        /usr/local/cuda-13.0/bin/nvcc
        /usr/local/cuda-13.1/bin/nvcc
    )

    while IFS= read -r path; do
        [ -n "$path" ] && candidates+=("$path")
    done < <(find "$CONDA_PREFIX/lib/python3.11/site-packages" -type f \( -path '*/bin/nvcc' -o -path '*/nvidia/*/bin/nvcc' \) 2>/dev/null || true)

    for candidate in "${candidates[@]}"; do
        if [ -n "$candidate" ] && [ -x "$candidate" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    return 1
}

CUDA13_NVCC="$(resolve_nvcc || true)"
if [ -n "$CUDA13_NVCC" ]; then
    CUDA_HOME="$(dirname "$(dirname "$CUDA13_NVCC")")"
    export CUDA_HOME
    export PATH="${CUDA_HOME}/bin:${PATH}"
    mkdir -p "$CONDA_PREFIX/bin"
    ln -sf "$CUDA13_NVCC" "$CONDA_PREFIX/bin/nvcc" 2>/dev/null || true
    echo "Using nvcc at: $CUDA13_NVCC"
else
    CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}"
    export CUDA_HOME
    export PATH="${CUDA_HOME}/bin:${PATH}"
    echo "Warning: no nvcc binary was found in the active environment; set CUDA_HOME manually if needed."
fi

echo
echo "[1/7] Environment"
echo "CONDA_PREFIX=$CONDA_PREFIX"
echo "CUDA_HOME=$CUDA_HOME"
echo "Python=$(which python)"
python --version

# --------------------------------------------------
# 2. Verify PyTorch
# --------------------------------------------------

echo
echo "[2/7] PyTorch"

python - <<'PY'
import torch

print("torch       :", torch.__version__)
print("torch CUDA  :", torch.version.cuda)
print("CUDA usable :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU         :", torch.cuda.get_device_name(0))

if torch.version.cuda != "13.0":
    raise SystemExit(
        f"ERROR: Expected PyTorch CUDA 13.0, got {torch.version.cuda}"
    )
PY

# torchvision is not needed for a text-only Axolotl fine-tuning run and
# can be incompatible with the CUDA 13 PyTorch build in this environment.
# Remove it before importing transformers/axolotl.

echo
echo "[2.5/7] Removing incompatible torchvision"
python - <<'PY'
import importlib.util
import subprocess
import sys

spec = importlib.util.find_spec('torchvision')
if spec is not None:
    print('torchvision detected; removing it to avoid torch/vision ABI mismatches')
    subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchvision'])
else:
    print('torchvision not installed; skipping removal')
PY

# --------------------------------------------------
# 3. Locate CUDA 13 runtime
# --------------------------------------------------

echo
echo "[3/7] CUDA 13 runtime"

if [ ! -d "$CUDA13_LIB" ]; then
    echo "ERROR: CUDA 13 runtime directory does not exist:"
    echo "  $CUDA13_LIB"
    exit 1
fi

echo "CUDA 13 lib:"
echo "  $CUDA13_LIB"

NVJITLINK="$(find "$CUDA13_LIB" -maxdepth 1 \
    -name 'libnvJitLink.so.13*' \
    -type f \
    | head -1)"

if [ -z "$NVJITLINK" ]; then
    echo
    echo "ERROR: libnvJitLink.so.13 is missing from the PyTorch CUDA 13 runtime."
    echo
    echo "Installed CUDA libraries:"
    ls -1 "$CUDA13_LIB" | head -50
    exit 1
fi

echo "Found:"
echo "  $NVJITLINK"

# --------------------------------------------------
# 4. Repair conda activation environment
# --------------------------------------------------

echo
echo "[4/7] Configuring CUDA 13 runtime"

ACTIVATE_DIR="$CONDA_PREFIX/etc/conda/activate.d"
mkdir -p "$ACTIVATE_DIR"

cat > "$ACTIVATE_DIR/cuda13.sh" <<'ACTIVATE'
# PyTorch CUDA 13 runtime
CONDA_PREFIX="${CONDA_PREFIX:-$(conda info --base)/envs/py3.11}"
resolve_nvcc() {
    local candidates=(
        "$CONDA_PREFIX/bin/nvcc"
        "$CONDA_PREFIX/../bin/nvcc"
        /usr/local/cuda/bin/nvcc
        /usr/local/cuda-12.8/bin/nvcc
        /usr/local/cuda-13.0/bin/nvcc
        /usr/local/cuda-13.1/bin/nvcc
    )

    while IFS= read -r path; do
        [ -n "$path" ] && candidates+=("$path")
    done < <(find "$CONDA_PREFIX/lib/python3.11/site-packages" -type f \( -path '*/bin/nvcc' -o -path '*/nvidia/*/bin/nvcc' \) 2>/dev/null || true)

    for candidate in "${candidates[@]}"; do
        if [ -n "$candidate" ] && [ -x "$candidate" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    return 1
}

CUDA13_NVCC="$(resolve_nvcc || true)"
if [ -n "$CUDA13_NVCC" ]; then
    export CUDA_HOME="$(dirname "$(dirname "$CUDA13_NVCC")")"
else
    export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}"
fi

export PATH="${CUDA_HOME}/bin:${PATH}"

CUDA13_LIB="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cu13/lib"

# Remove system CUDA library paths that can cause CUDA 12/13
# library mixing, then prepend the CUDA 13 runtime.
if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    _CUDA13_CLEAN_LD=""
    IFS=':' read -ra _CUDA13_PATHS <<< "$LD_LIBRARY_PATH"

    for _p in "${_CUDA13_PATHS[@]}"; do
        case "$_p" in
            /usr/local/cuda/lib64)
                ;;
            /usr/local/cuda-12.*/*)
                ;;
            /usr/local/cuda-12.*)
                ;;
            *)
                if [ -n "$_CUDA13_CLEAN_LD" ]; then
                    _CUDA13_CLEAN_LD="${_CUDA13_CLEAN_LD}:$_p"
                else
                    _CUDA13_CLEAN_LD="$_p"
                fi
                ;;
        esac
    done

    export LD_LIBRARY_PATH="$CUDA13_LIB${_CUDA13_CLEAN_LD:+:$_CUDA13_CLEAN_LD}"
else
    export LD_LIBRARY_PATH="$CUDA13_LIB"
fi

unset _CUDA13_CLEAN_LD
unset _CUDA13_PATHS
unset _p
ACTIVATE

echo "Created:"
echo "  $ACTIVATE_DIR/cuda13.sh"

# --------------------------------------------------
# 5. Apply immediately
# --------------------------------------------------

echo
echo "[5/7] Applying CUDA 13 library path"

export CUDA_HOME="$CONDA_PREFIX"

# Explicitly put CUDA 13 first for this process.
export LD_LIBRARY_PATH="$CUDA13_LIB:${LD_LIBRARY_PATH:-}"

echo "LD_LIBRARY_PATH:"
echo "$LD_LIBRARY_PATH"

# --------------------------------------------------
# 6. Test bitsandbytes
# --------------------------------------------------

echo
echo "[6/7] Testing bitsandbytes"

python - <<'PY'
import os
import ctypes
import torch

print("torch:", torch.__version__)
print("CUDA :", torch.version.cuda)
print("GPU  :", torch.cuda.get_device_name(0))

try:
    ctypes.CDLL("libnvJitLink.so.13")
    print("libnvJitLink.so.13: OK")
except OSError as e:
    raise SystemExit(
        "ERROR: libnvJitLink.so.13 still cannot be loaded:\n" + str(e)
    )

import bitsandbytes as bnb
print("bitsandbytes:", bnb.__version__)
print("bitsandbytes import: OK")
PY

# --------------------------------------------------
# 7. Test Axolotl
# --------------------------------------------------

echo
echo "[7/7] Testing Axolotl"

axolotl --version

echo
echo "=================================================="
echo " FIX COMPLETE"
echo "=================================================="
echo
echo "Environment:"
echo "  $ENV_NAME"
echo
echo "PyTorch:"
python -c 'import torch; print(torch.__version__, "CUDA", torch.version.cuda)'
echo
echo "GPU:"
python -c 'import torch; print(torch.cuda.get_device_name(0))'
echo
echo "bitsandbytes:"
python -c 'import bitsandbytes as bnb; print(bnb.__version__)'
echo
echo "CUDA 13 nvJitLink:"
ldd "$(python -c 'import bitsandbytes, pathlib; import bitsandbytes.cextension as c; print(c.lib.__file__)' 2>/dev/null)" 2>/dev/null | grep -i nvJitLink || true
echo
echo "System CUDA remains:"
readlink -f /usr/local/cuda || true
echo
echo "Open a NEW shell or run:"
echo
echo "  conda activate py3.11"
echo
echo "then:"
echo
echo "  axolotl --help"
echo
