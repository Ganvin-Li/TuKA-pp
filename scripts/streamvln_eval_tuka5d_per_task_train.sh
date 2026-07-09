#!/usr/bin/env bash
# Per-task TRAIN-split inference for TuKA-5D. Identical to
# streamvln_eval_tuka5d_per_task.sh except SPLIT defaults to "train" and
# the output goes under results/per_task_tuka5d_train/.
#
# WHY use this:
#   - Sanity check: a perfectly-trained model should achieve ~near 100% SR
#     on its own training data. If trained SR on train is also low, the
#     model is broken upstream (not just generalisation).
#   - Compare train vs val_seen gap: large gap = overfit; small gap +
#     low val = underfit / capacity issue.
#
# PRE-REQUISITE: per-task train.json.gz must exist under data/task/Task_<k>/.
# Build them once via:
#     bash scripts/split_val_tasks.sh train
#
# Usage:
#   bash scripts/streamvln_eval_tuka5d_per_task_train.sh           # all 23 on train
#   bash scripts/streamvln_eval_tuka5d_per_task_train.sh 1,3,15    # specific ids
#
# Under the hood we just delegate to the main launcher with SPLIT=train.

ONLY=${1:-""}
SPLIT=train bash "$(dirname "$0")/streamvln_eval_tuka5d_per_task.sh" "${ONLY}"
