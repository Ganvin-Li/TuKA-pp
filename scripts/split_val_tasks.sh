#!/usr/bin/env bash
# Build per-task val_seen / val_unseen .json.gz files at
#   data/task/Task_<k>/val_seen.json.gz
#   data/task/Task_<k>/val_unseen.json.gz
#
# This is CPU-only and finishes in seconds (no habitat sim, no GPU). Each
# task's file is a strict subset of data/datasets/<src>/<split>/<split>.json.gz
# filtered to the task's scan -- the eval launcher uses these instead of the
# full split so per-task metrics only count episodes inside that scan.
#
# Usage:
#   bash scripts/split_val_tasks.sh           # both val_seen and val_unseen
#   bash scripts/split_val_tasks.sh val_seen  # just val_seen

set -u

# Default ALL three splits so the user always has the per-task files needed
# for both online inference (val_seen / val_unseen) AND train-set rollout
# sanity checks (train). The previous default omitted train, which forced
# a separate manual run.
SPLITS=${1:-"train,val_seen,val_unseen"}

# Sanity: every source val.json.gz must exist (output of Stage A).
for ds in r2r reverie cvdn; do
    for sp in val_seen val_unseen; do
        f="data/datasets/${ds}/${sp}/${sp}.json.gz"
        if [[ ! -f "${f}" ]]; then
            echo "!! missing source: ${f}"
            echo "   Run Stage A first (bash scripts/convert_to_vlnce.sh)"
            exit 1
        fi
    done
done

python convert/split_val_into_tasks.py \
    --task_def     convert/task_definition_40.json \
    --datasets_root data/datasets \
    --output_root   data/task \
    --splits        "${SPLITS}"

echo ""
echo "============================================================"
echo "Per-task val data written under data/task/Task_<k>/."
echo "Inspect e.g.:"
echo "    python -c \"import json,gzip; d=json.load(gzip.open('data/task/Task_1/val_seen.json.gz','rt')); print('Task_1 val_seen eps:', len(d['episodes']))\""
echo "============================================================"
