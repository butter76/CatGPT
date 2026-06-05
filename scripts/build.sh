#!/usr/bin/env bash
# Sudo-less bootstrap for lks_uci + trt_benchmark on Ubuntu 22.04.
#
# Invoked by ../update.sh, which sets WORK_DIR to the directory that
# contains both the cloned CatGPT/ tree and the durable install
# artifacts (gcc-14/, TensorRT-10.16.1.11/, ${MODEL}.onnx,
# ${MODEL}.network). MODEL defaults to S4; set MODEL=S2 (etc.) to point
# at a different model stem.
#
# Idempotent: every phase has a sentinel check so re-runs only do work
# that's actually outstanding.
#
# Required on the host (cannot be installed without root):
#   - NVIDIA kernel driver matched to CUDA 12.x  (detected via nvidia-smi)
#   - A CUDA 12.x toolkit somewhere on disk      (detected by searching)
#
# Everything else (gcc-14, TensorRT, libfork/fathom/chess-library
# submodules, the .network engine bundle) is fetched/built into
# WORK_DIR.

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"

# WORK_DIR is normally set by update.sh; fall back to the repo's parent
# so this script is also runnable standalone for dev iteration.
WORK_DIR="${WORK_DIR:-$(cd "$REPO_DIR/.." && pwd -P)}"
mkdir -p "$WORK_DIR/.cache"

MODEL="${MODEL:-S4}"

GCC_PREFIX="$WORK_DIR/gcc-14"
TRT_DIR="$WORK_DIR/TensorRT-10.16.1.11"
ONNX_PATH="$WORK_DIR/${MODEL}.onnx"
NETWORK_PATH="$WORK_DIR/${MODEL}.network"
BUILD_DIR="$REPO_DIR/cpp/build"

GCC_VERSION="${GCC_VERSION:-14.2.0}"
GCC_TARBALL_URL="https://ftp.gnu.org/gnu/gcc/gcc-${GCC_VERSION}/gcc-${GCC_VERSION}.tar.xz"

TRT_URL="https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.16.1/tars/TensorRT-10.16.1.11.Linux.x86_64-gnu.cuda-12.9.tar.gz"

# TODO: replace with the real hosted ONNX URL once we have one.
ONNX_URL="${ONNX_URL:-}"

JOBS="${JOBS:-$(nproc)}"

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

log()  { printf '[build.sh] %s\n' "$*"; }
warn() { printf '[build.sh] WARN: %s\n' "$*" >&2; }
die()  { printf '[build.sh] FATAL: %s\n' "$*" >&2; exit 1; }

phase() {
    echo
    echo "============================================================"
    echo "  $*"
    echo "============================================================"
}

# Mirror stdout+stderr to a build log in WORK_DIR.
exec > >(tee -a "$WORK_DIR/build.log") 2>&1
log "build.sh starting at $(date -Is)"
log "WORK_DIR=$WORK_DIR"
log "REPO_DIR=$REPO_DIR"
log "MODEL=$MODEL"

# ---------------------------------------------------------------------------
# Phase 1: Debug scan (always runs, never fails)
# ---------------------------------------------------------------------------

phase "Phase 1: machine scan"

run_quiet() {
    # Run a probe command, swallow nonzero exits so we never abort the
    # scan just because something isn't installed.
    local desc=$1; shift
    printf -- '--- %s ---\n' "$desc"
    "$@" 2>&1 || echo "(not available: $*)"
    echo
}

run_quiet "uname"                  uname -a
run_quiet "os release"             cat /etc/os-release
run_quiet "glibc"                  ldd --version
run_quiet "system gcc"             gcc --version
run_quiet "system g++"             g++ --version
run_quiet "any g++-14 on PATH"     bash -c 'command -v g++-14 || true'
run_quiet "python3"                python3 --version
run_quiet "cmake"                  cmake --version
run_quiet "make"                   make --version
run_quiet "git"                    git --version
run_quiet "nvidia-smi -L"          nvidia-smi -L
run_quiet "nvidia-smi gpu info"    nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv
run_quiet "nvcc on PATH"           bash -c 'command -v nvcc || true'
run_quiet "nvcc --version"         nvcc --version

echo "--- CUDA toolkit candidate dirs ---"
for pat in '/usr/local/cuda*' '/opt/cuda*' '/usr/lib/cuda*' "$HOME/cuda*"; do
    # shellcheck disable=SC2086
    ls -d $pat 2>/dev/null || true
done
echo

echo "--- libcudart.so candidates (top 10) ---"
{ ldconfig -p 2>/dev/null | grep -F 'libcudart' || true; } | head -n 10
echo

run_quiet "disk free in WORK_DIR" df -h "$WORK_DIR"
run_quiet "memory" bash -c 'free -h || true'

# Fail fast if there's no GPU at all — the binary is useless without one.
if ! command -v nvidia-smi >/dev/null 2>&1; then
    die "nvidia-smi not found. NVIDIA driver must be installed by an admin (requires root)."
fi
if ! nvidia-smi -L >/dev/null 2>&1; then
    die "nvidia-smi present but cannot enumerate GPUs. Check driver health."
fi

# ---------------------------------------------------------------------------
# Phase 2: Locate CUDA 12.x toolkit
# ---------------------------------------------------------------------------

phase "Phase 2: locate CUDA 12.x toolkit"

# Parse CUDA_VERSION (in thousands: 12080 => 12.8) from <cuda.h>.
parse_cuda_version() {
    local cuda_h=$1
    awk '/^#define[[:space:]]+CUDA_VERSION[[:space:]]/ {print $3; exit}' "$cuda_h" 2>/dev/null
}

cuda_candidate_libdir() {
    # Return the libdir (lib64 or lib) inside $1 that has libcudart.so,
    # or empty if neither does.
    local root=$1
    if [[ -e "$root/lib64/libcudart.so" || -n "$(ls "$root/lib64"/libcudart.so.* 2>/dev/null | head -1)" ]]; then
        echo "$root/lib64"
    elif [[ -e "$root/lib/libcudart.so" || -n "$(ls "$root/lib"/libcudart.so.* 2>/dev/null | head -1)" ]]; then
        echo "$root/lib"
    fi
}

validate_cuda_root() {
    # Echo "ROOT|LIBDIR|VERSION" if $1 looks like a usable CUDA 12.x
    # toolkit; otherwise echo nothing.
    local root=$1
    [[ -d "$root" ]] || return 1
    local cuda_h="$root/include/cuda.h"
    [[ -f "$cuda_h" ]] || return 1
    local libdir
    libdir=$(cuda_candidate_libdir "$root")
    [[ -n "$libdir" ]] || return 1
    local ver
    ver=$(parse_cuda_version "$cuda_h")
    [[ -n "$ver" ]] || return 1
    if (( ver < 12000 )); then
        warn "skipping $root (CUDA_VERSION=$ver < 12000)"
        return 1
    fi
    echo "$root|$libdir|$ver"
}

CUDA_ROOT=""
CUDA_LIB_DIR=""
CUDA_VERSION_INT=""

# Candidate roots, in priority order.
declare -a CUDA_CANDIDATES=()
[[ -n "${CUDA_ROOT_OVERRIDE:-}" ]] && CUDA_CANDIDATES+=("$CUDA_ROOT_OVERRIDE")
[[ -n "${CUDA_HOME:-}"          ]] && CUDA_CANDIDATES+=("$CUDA_HOME")
if command -v nvcc >/dev/null 2>&1; then
    CUDA_CANDIDATES+=("$(cd "$(dirname "$(command -v nvcc)")/.." && pwd -P)")
fi
for pat in '/usr/local/cuda' '/usr/local/cuda-12'* '/opt/cuda' '/opt/cuda-12'* '/usr/lib/cuda' '/usr/lib/cuda-12'*; do
    for d in $pat; do
        [[ -d "$d" ]] && CUDA_CANDIDATES+=("$d")
    done
done
[[ -n "${CONDA_PREFIX:-}" && -f "$CONDA_PREFIX/include/cuda.h" ]] && CUDA_CANDIDATES+=("$CONDA_PREFIX")

# Last-resort wide search for libcudart.so.12*, derive root from there.
if [[ ${#CUDA_CANDIDATES[@]} -eq 0 ]] \
   || ! { for c in "${CUDA_CANDIDATES[@]}"; do validate_cuda_root "$c" >/dev/null && { echo HIT; break; }; done | grep -q HIT; }; then
    log "no obvious CUDA root yet — falling back to a wider search (may take a few seconds)"
    while IFS= read -r libpath; do
        # libcudart at .../lib64/libcudart.so.12.x  =>  root is two parents up.
        root=$(cd "$(dirname "$libpath")/.." && pwd -P)
        CUDA_CANDIDATES+=("$root")
    done < <(find /usr /opt "$HOME" -maxdepth 6 -name 'libcudart.so.12*' 2>/dev/null | head -n 8)
fi

log "CUDA candidate roots:"
for c in "${CUDA_CANDIDATES[@]}"; do echo "  - $c"; done

for c in "${CUDA_CANDIDATES[@]}"; do
    if out=$(validate_cuda_root "$c"); then
        CUDA_ROOT=${out%%|*}
        rest=${out#*|}
        CUDA_LIB_DIR=${rest%%|*}
        CUDA_VERSION_INT=${rest#*|}
        break
    fi
done

[[ -n "$CUDA_ROOT" ]] || die "could not locate a CUDA 12.x toolkit. Install one (or set CUDA_ROOT_OVERRIDE=...) and re-run."

log "CUDA_ROOT     = $CUDA_ROOT"
log "CUDA_LIB_DIR  = $CUDA_LIB_DIR"
log "CUDA_VERSION  = $CUDA_VERSION_INT  (i.e. $(( CUDA_VERSION_INT / 1000 )).$(( (CUDA_VERSION_INT / 10) % 100 )))"

# ---------------------------------------------------------------------------
# Phase 3: GCC 14 (sentinel: $WORK_DIR/gcc-14/bin/g++-14)
# ---------------------------------------------------------------------------

phase "Phase 3: GCC 14"

if [[ -x "$GCC_PREFIX/bin/g++-14" ]] \
   && "$GCC_PREFIX/bin/g++-14" --version >/dev/null 2>&1; then
    log "found existing GCC 14 at $GCC_PREFIX (skipping build)"
else
    log "no GCC 14 in $GCC_PREFIX — building from source (this takes ~30-60 min)"
    GCC_SRC_TARBALL="$WORK_DIR/.cache/gcc-${GCC_VERSION}.tar.xz"
    GCC_SRC_DIR="$WORK_DIR/.cache/gcc-${GCC_VERSION}"
    GCC_BUILD_DIR="$WORK_DIR/.cache/gcc-${GCC_VERSION}-build"

    if [[ ! -s "$GCC_SRC_TARBALL" ]]; then
        log "downloading $GCC_TARBALL_URL"
        wget -O "$GCC_SRC_TARBALL.part" "$GCC_TARBALL_URL"
        mv "$GCC_SRC_TARBALL.part" "$GCC_SRC_TARBALL"
    fi

    if [[ ! -d "$GCC_SRC_DIR" ]]; then
        log "extracting $(basename "$GCC_SRC_TARBALL")"
        tar -xf "$GCC_SRC_TARBALL" -C "$WORK_DIR/.cache"
    fi

    if [[ ! -d "$GCC_SRC_DIR/gmp" || ! -d "$GCC_SRC_DIR/mpfr" || ! -d "$GCC_SRC_DIR/mpc" ]]; then
        log "fetching GCC prerequisites (gmp, mpfr, mpc, isl)"
        ( cd "$GCC_SRC_DIR" && ./contrib/download_prerequisites )
    fi

    rm -rf "$GCC_BUILD_DIR"
    mkdir -p "$GCC_BUILD_DIR"
    (
        cd "$GCC_BUILD_DIR"
        log "configuring GCC ${GCC_VERSION}"
        "$GCC_SRC_DIR/configure" \
            --prefix="$GCC_PREFIX" \
            --disable-multilib \
            --disable-bootstrap \
            --enable-languages=c,c++ \
            --program-suffix=-14
        log "building GCC ${GCC_VERSION} with -j${JOBS}"
        make -j"$JOBS"
        log "installing GCC ${GCC_VERSION} into $GCC_PREFIX"
        make install
    )

    [[ -x "$GCC_PREFIX/bin/g++-14" ]] || die "GCC build finished but $GCC_PREFIX/bin/g++-14 is missing"
fi

CC="$GCC_PREFIX/bin/gcc-14"
CXX="$GCC_PREFIX/bin/g++-14"
GCC_LIB_DIR="$GCC_PREFIX/lib64"
[[ -d "$GCC_LIB_DIR" ]] || GCC_LIB_DIR="$GCC_PREFIX/lib"
export CC CXX
export LD_LIBRARY_PATH="${GCC_LIB_DIR}:${LD_LIBRARY_PATH:-}"

log "CC          = $CC"
log "CXX         = $CXX"
log "GCC_LIB_DIR = $GCC_LIB_DIR"
"$CXX" --version | head -1

# ---------------------------------------------------------------------------
# Phase 4: TensorRT 10.16.1.11 (sentinel: lib/libnvinfer.so present)
# ---------------------------------------------------------------------------

phase "Phase 4: TensorRT 10.16.1.11"

if [[ -f "$TRT_DIR/lib/libnvinfer.so" && -f "$TRT_DIR/include/NvInfer.h" ]]; then
    log "found TensorRT at $TRT_DIR (skipping download)"
else
    log "downloading TensorRT 10.16.1.11 tarball"
    TRT_TAR="$WORK_DIR/.cache/TensorRT-10.16.1.11.tar.gz"
    if [[ ! -s "$TRT_TAR" ]]; then
        wget -O "$TRT_TAR.part" "$TRT_URL"
        mv "$TRT_TAR.part" "$TRT_TAR"
    fi
    log "extracting into $WORK_DIR"
    tar -xzf "$TRT_TAR" -C "$WORK_DIR"
    [[ -f "$TRT_DIR/lib/libnvinfer.so" ]] || die "TensorRT extracted but $TRT_DIR/lib/libnvinfer.so missing"
fi

[[ -x "$TRT_DIR/bin/trtexec" ]] || die "trtexec missing at $TRT_DIR/bin/trtexec"
log "trtexec     = $TRT_DIR/bin/trtexec"

# ---------------------------------------------------------------------------
# Phase 5: ${MODEL}.onnx
# ---------------------------------------------------------------------------

phase "Phase 5: ${MODEL}.onnx"

if [[ -s "$ONNX_PATH" ]]; then
    log "found ONNX at $ONNX_PATH ($(stat -c %s "$ONNX_PATH") bytes)"
elif [[ -n "$ONNX_URL" ]]; then
    log "downloading ONNX from $ONNX_URL"
    wget -O "$ONNX_PATH.part" "$ONNX_URL"
    mv "$ONNX_PATH.part" "$ONNX_PATH"
else
    die "no ONNX at $ONNX_PATH and ONNX_URL is unset. Either stage ${MODEL}.onnx in WORK_DIR or set ONNX_URL and re-run."
fi

# ---------------------------------------------------------------------------
# Phase 6: configure + build (lks_uci, trt_benchmark)
# ---------------------------------------------------------------------------

phase "Phase 6: cmake configure + build"

rm -rf "$BUILD_DIR/CMakeCache.txt" "$BUILD_DIR/CMakeFiles"
mkdir -p "$BUILD_DIR"
cmake \
    -S "$REPO_DIR/cpp" \
    -B "$BUILD_DIR" \
    -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="$CC" \
    -DCMAKE_CXX_COMPILER="$CXX" \
    -DCUDA_ROOT="$CUDA_ROOT" \
    -DTENSORRT_ROOT="$TRT_DIR" \
    -DGCC14_LIB_DIR="$GCC_LIB_DIR"

cmake --build "$BUILD_DIR" --target lks_uci trt_benchmark -j"$JOBS"

LKS_UCI_BIN="$BUILD_DIR/bin/lks_uci"
TRT_BENCH_BIN="$BUILD_DIR/bin/trt_benchmark"
[[ -x "$LKS_UCI_BIN"    ]] || die "build claimed success but $LKS_UCI_BIN is missing"
[[ -x "$TRT_BENCH_BIN" ]] || die "build claimed success but $TRT_BENCH_BIN is missing"

# ---------------------------------------------------------------------------
# Phase 7: build .network bundle (sentinel: network newer than onnx)
# ---------------------------------------------------------------------------

phase "Phase 7: build ${MODEL}.network"

if [[ -s "$NETWORK_PATH" && "$NETWORK_PATH" -nt "$ONNX_PATH" ]]; then
    log "found up-to-date $NETWORK_PATH (newer than $ONNX_PATH) — skipping rebuild"
else
    log "building per-bucket TRT engines and packing into $NETWORK_PATH"
    # scripts/trt.sh writes per-bucket .trt files into CWD and then
    # rm's them after pack_network.py succeeds. Run it from WORK_DIR so
    # those scratch files land alongside the final .network.
    (
        cd "$WORK_DIR"
        TRT_ROOT="$TRT_DIR" \
        MODEL="$MODEL" \
        ONNX="$ONNX_PATH" \
        NETWORK_OUT="$NETWORK_PATH" \
        PYTHON="${PYTHON:-python3}" \
        bash "$REPO_DIR/scripts/trt.sh"
    )
fi

[[ -s "$NETWORK_PATH" ]] || die "trt.sh finished but $NETWORK_PATH is missing or empty"

CATGPT_BIN="$WORK_DIR/catgpt"
CATGPT_REAL="$WORK_DIR/catgpt.real"
log "installing engine binary -> $CATGPT_BIN (wrapper) + $CATGPT_REAL (ELF)"
# The real ELF lives next to the wrapper. The wrapper sets LD_LIBRARY_PATH
# to the GCC 14, TensorRT, and CUDA lib dirs so the loader picks up
# gcc-14/lib64/libstdc++.so.6 (which has the GLIBCXX_3.4.31/.32 + CXXABI_1.3.15
# symbols built into the binary) instead of the Ubuntu 22.04 system libstdc++.
# This is necessary because catgpt's DT_RUNPATH is not inherited by transitive
# deps (libstdc++ is pulled in via libnvinfer), so a bare ./catgpt invocation
# would otherwise resolve libstdc++ from /usr/lib and fail with GLIBCXX_3.4.32
# not found. See Phase 8's RUNTIME_LD for the same set of paths.
cp -f "$LKS_UCI_BIN" "$CATGPT_REAL"
chmod +x "$CATGPT_REAL"
cat > "$CATGPT_BIN" <<EOF
#!/usr/bin/env bash
# Auto-generated by CatGPT/scripts/build.sh. Sets LD_LIBRARY_PATH so the
# bundled gcc-14 libstdc++, TensorRT, and CUDA libs win over system ones,
# then execs the real ELF in-place (same PID, argv forwarded verbatim).
export LD_LIBRARY_PATH="${GCC_LIB_DIR}:${TRT_DIR}/lib:${CUDA_LIB_DIR}\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
exec "${CATGPT_REAL}" "\$@"
EOF
chmod +x "$CATGPT_BIN"

# ---------------------------------------------------------------------------
# Phase 8: trt_benchmark (stdout + log for the viewer)
# ---------------------------------------------------------------------------

phase "Phase 8: trt_benchmark"

RUNTIME_LD="${GCC_LIB_DIR}:${TRT_DIR}/lib:${CUDA_LIB_DIR}:${LD_LIBRARY_PATH:-}"
TRT_BENCH_LOG="$WORK_DIR/trt_benchmark.txt"

log "running trt_benchmark on $NETWORK_PATH (also teeing to $TRT_BENCH_LOG)"
set +e
env LD_LIBRARY_PATH="$RUNTIME_LD" "$TRT_BENCH_BIN" "$NETWORK_PATH" 2>&1 | tee "$TRT_BENCH_LOG"
bench_rc=${PIPESTATUS[0]}
set -e
if (( bench_rc != 0 )); then
    die "trt_benchmark exited with status $bench_rc (see $TRT_BENCH_LOG)"
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

phase "Done"
log "catgpt        = $CATGPT_BIN"
log "lks_uci       = $LKS_UCI_BIN"
log "trt_benchmark = $TRT_BENCH_BIN"
log "${MODEL}.network  = $NETWORK_PATH"
log "bench log     = $TRT_BENCH_LOG"
echo
echo "OK: $CATGPT_BIN $NETWORK_PATH"
