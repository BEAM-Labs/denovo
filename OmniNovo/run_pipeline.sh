#!/usr/bin/env bash
# Complete workflow: prepare paired spectra -> infer both arms -> filter.
set -euo pipefail
if [[ $# -lt 3 ]]; then
  echo "Usage: bash run_pipeline.sh INPUT.mgf MODEL.ckpt NEW_OUTPUT_DIR [filter_fdr.py options]" >&2
  exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
INPUT="$("$PYTHON" -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$1")"
MODEL="$("$PYTHON" -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$2")"
WORK="$("$PYTHON" -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$3")"
shift 3
[[ -f "$INPUT" && -f "$MODEL" ]] || { echo "Input or checkpoint does not exist" >&2; exit 2; }
[[ ! -e "$WORK" ]] || { echo "Output already exists: $WORK" >&2; exit 2; }
CONFIG="${CONFIG:-$SCRIPT_DIR/OmniNovo/configs/default_rules.yaml}"
CONFIG="$("$PYTHON" -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$CONFIG")"
"$PYTHON" "$SCRIPT_DIR/prepare_decoys.py" --input "$INPUT" --output "$WORK/dataset" --rho "${RHO:-0.50}" --seed "${SEED:-7}"
cd "$SCRIPT_DIR"

for ARM in target decoy; do
  "$PYTHON" -m OmniNovo.run --mode=denovo --input_path="$WORK/dataset/$ARM.lmdb" \
    --model="$MODEL" --config="$CONFIG" --log_base="$WORK/predictions/$ARM"
done
"$PYTHON" "$SCRIPT_DIR/filter_fdr.py" --dataset "$WORK/dataset" \
  --target "$WORK/predictions/target/*rank*.tsv" --decoy "$WORK/predictions/decoy/*rank*.tsv" \
  --output "$WORK/filtered" "$@"
