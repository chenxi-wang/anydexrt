#!/usr/bin/env bash
# Run from new_codebase after activating your Python environment.
# Edit shared robot, data, glove and training settings in scripts/config.sh.
source scripts/config.sh || exit 1
# Contact correction is enabled below; remove both --contact-checkpoint and
# --contact-data-dir to disable it. Configure their paths in scripts/config.sh.
# CONTACT_DATA_DIR must contain the training glove normalization statistics.

# Set --contact-threshold in [0, 1) to tune confidence fusion.
python live.py \
  --config "$ROBOT_CONFIG" \
  --data-dir "$ROBOT_DATA_DIR" \
  --checkpoint "$RETARGETING_CHECKPOINT" \
  --glove-serial "$GLOVE_SERIAL" \
  --address "$GLOVE_ADDRESS" \
  --hand-type "$GLOVE_HAND_TYPE" \
  --device "$INFERENCE_DEVICE" \
  --contact-checkpoint "$CONTACT_CHECKPOINT" \
  --contact-data-dir "$CONTACT_DATA_DIR" \
  --contact-threshold 0.5
