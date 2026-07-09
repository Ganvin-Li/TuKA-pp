#!/usr/bin/env bash
# Stage A: convert raw R2R / REVERIE / NDH downloads -> VLN-CE-compatible
# <split>.json.gz files that eval (habitat.Env) and Stage B can both read.
#
# This is FAST and CPU-only -- no GPU, no habitat sim. A few seconds per
# split.
#
# Expected raw-data layout BEFORE running this script:
#   data/datasets/r2r_raw/{R2R_,}{train,val_seen,val_unseen}.json
#   data/datasets/reverie_raw/REVERIE_{train,val_seen,val_unseen}.json
#   data/datasets/cvdn_raw/{train,val_seen,val_unseen}.json
#
# Produces (consumed by eval AND by Stage B):
#   data/datasets/r2r/<split>/<split>.json.gz
#   data/datasets/reverie/<split>/<split>.json.gz
#   data/datasets/cvdn/<split>/<split>.json.gz
#
# Then optionally:
#   python convert/verify_vlnce.py --sweep
#
# Usage:
#   bash scripts/convert_to_vlnce.sh                # all 3 datasets, all 3 splits
#   bash scripts/convert_to_vlnce.sh r2r            # only r2r, all splits
#   bash scripts/convert_to_vlnce.sh reverie train  # only reverie/train

set -u

DATASET=${1:-all}
SPLIT=${2:-all}

run_r2r() {
    local sp="$1"
    echo "[Stage-A] R2R / ${sp}"
    python convert/convert_r2r_to_vlnce.py \
        --r2r_dir          data/datasets/r2r_raw \
        --connectivity_dir data/connectivity \
        --output_dir       data/datasets/r2r \
        --split            "${sp}"
}

run_reverie() {
    local sp="$1"
    echo "[Stage-A] REVERIE / ${sp}"
    python convert/convert_reverie_to_vlnce.py \
        --reverie_dir      data/datasets/reverie_raw \
        --connectivity_dir data/connectivity \
        --output_dir       data/datasets/reverie \
        --split            "${sp}"
}

run_cvdn() {
    local sp="$1"
    echo "[Stage-A] CVDN / ${sp}"
    python convert/convert_ndh_to_vlnce.py \
        --ndh_dir          data/datasets/cvdn_raw \
        --connectivity_dir data/connectivity \
        --output_dir       data/datasets/cvdn \
        --split            "${sp}"
}

dispatch() {
    local ds="$1"; local sp="$2"
    case "${ds}" in
        r2r)     run_r2r     "${sp}" ;;
        reverie) run_reverie "${sp}" ;;
        cvdn)    run_cvdn    "${sp}" ;;
        *) echo "[err] unknown dataset: ${ds}" >&2; return 1 ;;
    esac
}

if [[ "${DATASET}" == "all" && "${SPLIT}" == "all" ]]; then
    for ds in r2r reverie cvdn; do
        for sp in train val_seen val_unseen; do
            dispatch "${ds}" "${sp}"
        done
    done
elif [[ "${DATASET}" == "all" ]]; then
    for ds in r2r reverie cvdn; do
        dispatch "${ds}" "${SPLIT}"
    done
elif [[ "${SPLIT}" == "all" ]]; then
    for sp in train val_seen val_unseen; do
        dispatch "${DATASET}" "${sp}"
    done
else
    dispatch "${DATASET}" "${SPLIT}"
fi

echo ""
echo "============================================================"
echo "Stage A done. To verify everything is loadable by habitat:"
echo "    python convert/verify_vlnce.py --sweep"
echo "Next: bash scripts/collect_trajectories.sh all   # train rgb (Stage B)"
echo "============================================================"
