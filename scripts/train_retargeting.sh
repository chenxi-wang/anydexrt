#!/usr/bin/env bash
# Run from new_codebase after activating your Python environment.
# Edit shared robot, data, glove and training settings in scripts/config.sh.
source scripts/config.sh || exit 1
# --epochs: number of passes through the training dataset.
# --repeats: how many times the motion dataset is repeated per epoch.

python train_retargeting.py \
  --config "$ROBOT_CONFIG" \
  --data-dir "$ROBOT_DATA_DIR" \
  --checkpoint "$RETARGETING_CHECKPOINT" \
  --epochs 5 \
  --batch-size "$BATCH_SIZE" \
  --workers "$NUM_WORKERS" \
  --seed "$RANDOM_SEED" \
  --repeats 100
