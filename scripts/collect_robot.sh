#!/usr/bin/env bash
# Run from new_codebase after activating your Python environment.
# Edit shared robot, data, glove and training settings in scripts/config.sh.
source scripts/config.sh || exit 1

python collect.py robot \
  --config "$ROBOT_CONFIG" \
  --data-root "$DATA_ROOT"
