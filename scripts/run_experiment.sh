#!/usr/bin/env bash
# Run one experiment config across every algorithm in the paper, over several seeds.
#
#   scripts/run_experiment.sh configs/baseline-ring.yaml 16
#   scripts/run_experiment.sh configs/vary-noniid-roc.yaml 10 --non-iid 0.9
#
# Positional: <config> <num_clients>. Anything further is passed to swift-train,
# so per-run overrides work: --non-iid 0.9, --slowdown 4, --clusters 2, ...
#
# Environment:
#   SEEDS        seeds to sweep (default: the five below)
#   OUTPUT_DIR   where runs are written (default: outputs)
#   FORCE=1      re-run runs that already have results instead of skipping them
#
# A sweep is long enough that it will sometimes be interrupted. Finished runs
# are skipped on the next invocation, and one failing run does not abandon the
# rest -- failures are collected and reported at the end.
set -uo pipefail

CONFIG=${1:?usage: $0 <config.yaml> <num_clients> [extra swift-train flags...]}
CLIENTS=${2:?usage: $0 <config.yaml> <num_clients> [extra swift-train flags...]}
shift 2

for tool in mpirun swift-train; do
  command -v "$tool" >/dev/null || { echo "$tool not found; pip install -e ." >&2; exit 1; }
done
[ -f "$CONFIG" ] || { echo "no such config: $CONFIG" >&2; exit 1; }

# One shared seed list, so every algorithm sees the same data partition and the
# same initial model and the comparison is paired. The pre-2.0 scripts used a
# different list per algorithm (d-sgd: 1000 2500 250 225 200, ld-sgd: 115 105
# 75 25 15, pa-sgd: 2828 3789 99 122 37, swift: the list below); set SEEDS to
# reproduce a particular one of those runs.
SEEDS=${SEEDS:-"1333 1346 1345 9183 1337"}
OUTPUT_DIR=${OUTPUT_DIR:-outputs}
FORCE=${FORCE:-0}
EXPERIMENT=$(basename "$CONFIG" .yaml)

# Algorithm label -> flags. The paper's C0 rows use one local step and the C1
# rows use two; the synchronous baselines differ only in (i1, i2).
declare -a RUNS=(
  "swift|--algorithm swift --local-steps 1"
  "swift-2sgd|--algorithm swift --local-steps 2"
  "d-sgd|--algorithm d-sgd"
  "pa-sgd|--algorithm pa-sgd --i1 1"
  "ld-sgd|--algorithm ld-sgd --i1 1 --i2 2"
)

declare -a FAILED=()
skipped=0
for run in "${RUNS[@]}"; do
  label=${run%%|*}
  flags=${run#*|}
  for seed in $SEEDS; do
    name="${EXPERIMENT}/${label}/seed${seed}"
    if [ "$FORCE" != "1" ] && [ -f "${OUTPUT_DIR}/${name}/summary-rank0.json" ]; then
      skipped=$((skipped + 1))
      continue
    fi
    echo "==> ${name} (${CLIENTS} clients)"
    # shellcheck disable=SC2086
    if ! mpirun -np "$CLIENTS" swift-train \
        --config "$CONFIG" \
        $flags \
        --seed "$seed" \
        --name "$name" \
        --output-dir "$OUTPUT_DIR" \
        "$@"; then
      echo "!!! ${name} failed" >&2
      FAILED+=("$name")
    fi
  done
done

[ "$skipped" -gt 0 ] && echo "==> skipped $skipped run(s) that already had results (FORCE=1 to redo)"

if [ ${#FAILED[@]} -gt 0 ]; then
  echo "==> ${#FAILED[@]} run(s) failed:" >&2
  printf '      %s\n' "${FAILED[@]}" >&2
  exit 1
fi

echo "==> all runs complete."
echo "    python tools/summarize.py ${OUTPUT_DIR}/${EXPERIMENT}"
echo "    python tools/plot.py ${OUTPUT_DIR}/${EXPERIMENT} --metric test_loss"
