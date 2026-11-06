#!/usr/bin/env bash
# Run from new_codebase after activating your Python environment.
# Edit shared robot, data, glove and training settings in scripts/config.sh.
source scripts/config.sh || exit 1

# Press r to start each finger recording; q cancels without saving partial data.
python collect.py motion \
  --config "$ROBOT_CONFIG" \
  --data-root "$DATA_ROOT" \
  --glove-serial "$GLOVE_SERIAL" \
  --address "$GLOVE_ADDRESS" \
  --hand-type "$GLOVE_HAND_TYPE"
