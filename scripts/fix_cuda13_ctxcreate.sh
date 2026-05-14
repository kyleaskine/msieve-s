#!/bin/sh
set -eu

cd "$(dirname "$0")/.."
file="gnfs/poly/stage1/stage1_sieve_gpu.c"
major=$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\)\..*/\1/p' | head -n 1)

[ -n "$major" ] || { echo "could not detect CUDA version from nvcc"; exit 1; }

if [ "$major" -ge 13 ] && grep -q '^[[:space:]]*// NULL, //uncomment this on CUDA >= 13' "$file"; then
	sed -i 's#^\([[:space:]]*\)// NULL, //uncomment this on CUDA >= 13#\1NULL, // CUDA >= 13 cuCtxCreate params#' "$file"
	echo "CUDA $major: updated $file"
elif [ "$major" -lt 13 ] && grep -q '^[[:space:]]*NULL, // CUDA >= 13 cuCtxCreate params' "$file"; then
	sed -i 's#^\([[:space:]]*\)NULL, // CUDA >= 13 cuCtxCreate params#\1// NULL, //uncomment this on CUDA >= 13#' "$file"
	echo "CUDA $major: restored pre-CUDA-13 $file"
else
	echo "CUDA $major: no change needed"
fi
