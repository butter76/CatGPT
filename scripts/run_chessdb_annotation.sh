#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Annotate chessDB tactical positions with lc0.
#
# Prerequisites:
#   1. Run tsv_to_fens.py on all 128 clean.*.tsv shards first:
#        for f in ~/tactical_output_filtered/clean.*.tsv; do
#            uv run python scripts/tsv_to_fens.py "$f" -o ~/chessdb_annotation/ &
#        done; wait
#
#   2. This script then runs lc0 annotate across 8 GPUs, processing
#      16 shards per GPU serially.
#
# Usage:
#   bash scripts/run_chessdb_annotation.sh
#
#   # Resume from a specific shard (skip already-done ones):
#   RESUME=1 bash scripts/run_chessdb_annotation.sh
# ---------------------------------------------------------------------------

LC0_DIR="${LC0_DIR:-$HOME/lc0-annotation/build/release}"
LC0="$LC0_DIR/lc0"
WEIGHTS="$LC0_DIR/768x15x24h-t82-swa-11264000.pb.gz"
INPUT_DIR="${CHESSDB_DIR:-$HOME/chessdb_annotation}"
NUM_GPUS=8
NUM_SHARDS=128
RESUME="${RESUME:-0}"

LC0_FLAGS=(
    --visits=400
    --parallel=8
    --cpuct=1.20
    --cpuct-at-root=2.0
    --root-has-own-cpuct-params=true
    --fpu-strategy=reduction
    --fpu-value=0.49
    --fpu-strategy-at-root=absolute
    --fpu-value-at-root=1.0
    --policy-softmax-temp=1.45
    --noise-epsilon=0.0
    --noise-alpha=0.12
    --minimum-kldgain-per-node=0.000050
    --sticky-endgames=true
    --moves-left-max-effect=0.2
    --moves-left-threshold=0.0
    --moves-left-slope=0.007
    --moves-left-quadratic-factor=0.85
    --moves-left-scaled-factor=0.15
    --moves-left-constant-factor=0.0
)

run_gpu() {
    local gpu=$1
    shift
    local shards=("$@")

    for shard in "${shards[@]}"; do
        local input="$INPUT_DIR/chessdb-${shard}-fens.txt"
        local output="$INPUT_DIR/chessdb-${shard}-annotated.jsonl"

        if [[ ! -f "$input" ]]; then
            echo "[GPU $gpu] SKIP shard $shard: $input not found"
            continue
        fi

        if [[ "$RESUME" == "1" && -f "$output" ]]; then
            echo "[GPU $gpu] SKIP shard $shard: output already exists"
            continue
        fi

        echo "[GPU $gpu] Starting shard $shard ..."

        CUDA_VISIBLE_DEVICES=$gpu "$LC0" annotate \
            --input="$input" \
            --output="$output" \
            --weights="$WEIGHTS" \
            "${LC0_FLAGS[@]}" \
            > >(while IFS= read -r line; do echo "[GPU $gpu shard $shard] $line"; done) \
            2>&1

        echo "[GPU $gpu] Finished shard $shard"
    done
}

# Build shard assignments: round-robin across GPUs
declare -a GPU_SHARDS
for ((i = 0; i < NUM_GPUS; i++)); do
    GPU_SHARDS[$i]=""
done

for ((s = 0; s < NUM_SHARDS; s++)); do
    gpu=$((s % NUM_GPUS))
    shard=$(printf "%03d" "$s")
    GPU_SHARDS[$gpu]="${GPU_SHARDS[$gpu]} $shard"
done

echo "=== chessDB annotation ==="
echo "lc0:      $LC0"
echo "weights:  $WEIGHTS"
echo "input:    $INPUT_DIR"
echo "GPUs:     $NUM_GPUS"
echo "shards:   $NUM_SHARDS"
echo "resume:   $RESUME"
echo ""

for ((i = 0; i < NUM_GPUS; i++)); do
    echo "  GPU $i: shards${GPU_SHARDS[$i]}"
done
echo ""

pids=()

for ((i = 0; i < NUM_GPUS; i++)); do
    # shellcheck disable=SC2086
    run_gpu "$i" ${GPU_SHARDS[$i]} &
    pids+=($!)
done

echo "All $NUM_GPUS GPU workers launched. PIDs: ${pids[*]}"
echo "Waiting for all to finish ..."

failed=0
for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
        echo "FAILED: GPU $i worker (PID ${pids[$i]})"
        ((failed++))
    fi
done

if [[ $failed -eq 0 ]]; then
    echo "All GPUs completed successfully."
else
    echo "$failed GPU worker(s) failed."
    exit 1
fi
