#!/usr/bin/env bash
# Run from new_codebase after activating your Python environment.
# Edit shared robot, data, glove and training settings in scripts/config.sh.
source scripts/config.sh || exit 1
# --output-dir: save predicted XYZ arrays; remove this option to only display.

python visualize_retargeting.py \
  --config "$ROBOT_CONFIG" \
  --data-dir "$ROBOT_DATA_DIR" \
  --checkpoint "$RETARGETING_CHECKPOINT" \
  --device "$INFERENCE_DEVICE" \
  --output-dir "$PREDICTION_DIR"
