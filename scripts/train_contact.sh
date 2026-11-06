#!/usr/bin/env bash
# Run from new_codebase after activating your Python environment.
# Edit shared robot, data, glove and training settings in scripts/config.sh.
source scripts/config.sh || exit 1
# --epochs: number of passes through the training dataset.
# --samples: number of generated training samples per epoch.

python train_contact.py \
  --data-dir "$CONTACT_DATA_DIR" \
  --checkpoint "$CONTACT_CHECKPOINT" \
  --epochs 20 \
  --batch-size "$BATCH_SIZE" \
  --workers "$NUM_WORKERS" \
  --seed "$RANDOM_SEED" \
  --samples 1000000
