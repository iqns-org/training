#!/usr/bin/env bash
set -euo pipefail

set -a
. ~/.env 2>/dev/null || true
set +a

cd ~
rm -f ~/axolotl-job-success

source /root/miniconda3/etc/profile.d/conda.sh

# Keep BLAS/OpenMP thread creation within the remote container's process limits.
# The GPU.ai environment is heavily thread-constrained and OpenBLAS defaults to 64 threads.
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$OPENBLAS_NUM_THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OPENBLAS_NUM_THREADS}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$OPENBLAS_NUM_THREADS}"
export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-$OPENBLAS_NUM_THREADS}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-$OPENBLAS_NUM_THREADS}"

echo "BLAS thread caps: OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS} OMP_NUM_THREADS=${OMP_NUM_THREADS} MKL_NUM_THREADS=${MKL_NUM_THREADS}"
ulimit -u || true

. ./fix-axolotl.sh

python - <<'PY'
import ctypes
import os

import bitsandbytes as bnb
import torch

ctypes.CDLL('libnvJitLink.so.13')
print('CUDA', torch.version.cuda)
print('bitsandbytes', bnb.__version__)
print('LD_LIBRARY_PATH', os.environ.get('LD_LIBRARY_PATH', ''))
PY

echo "using config:"
sed -n '1,80p' ./${REMOTE_CFG_NAME}
axolotl preprocess ./${REMOTE_CFG_NAME}
axolotl train ./${REMOTE_CFG_NAME}
touch ~/axolotl-job-success
