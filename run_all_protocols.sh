#!/usr/bin/env bash
# Run every classifier head on every layer for all 4 attribution protocols.
#
# train_probes.py defaults to ALL layers (no --layers) and ALL heads (no
# --heads: cnn_pool, s4, cnn_attention), so we only loop over the protocols.
# Results land in $RESULTS_DIR/<protocol>/<model>[__cnn]/<head>/layer_KK/.
#
# Usage:
#   ./run_all_protocols.sh                       # defaults below
#   FEATURES_DIR=feats_test ./run_all_protocols.sh
#   ./run_all_protocols.sh --models wavlm_base_emotion --device cuda --epochs 40
# Any extra args are forwarded verbatim to train_probes.py.
set -euo pipefail

PY="${PY:-./venv/bin/python}"
FEATURES_DIR="${FEATURES_DIR:-feats_test}"
RESULTS_DIR="${RESULTS_DIR:-results}"
PROTOCOLS=(cv5 csr1 csr2 attr2 attr17)

for PROTO in "${PROTOCOLS[@]}"; do
    echo "=================== protocol: ${PROTO} ==================="
    "$PY" train_probes.py \
        --features-dir "$FEATURES_DIR" \
        --protocol "$PROTO" \
        --results-dir "$RESULTS_DIR" \
        "$@"
done

echo "All protocols done -> ${RESULTS_DIR}/{$(IFS=,; echo "${PROTOCOLS[*]}")}/"
