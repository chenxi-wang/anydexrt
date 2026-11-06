# Shared settings for all workflow scripts. Run workflows from new_codebase.
# Edit this file once; do not pass these settings as Bash command-line arguments.

# Robot name must match both configs/<name>.json and its hand_name field.
# Changing ROBOT_NAME also updates the derived data, checkpoint and output paths.
ROBOT_NAME="wuji"
DATA_ROOT="data"
CHECKPOINT_ROOT="checkpoints"
OUTPUT_ROOT="outputs"

# MANUS serial, streaming endpoint, and the side of the glove being worn.
GLOVE_SERIAL="4EC25730"
GLOVE_ADDRESS="tcp://127.0.0.1:8000"
GLOVE_HAND_TYPE="right"

# Inference device: cuda or cpu. Training always uses CUDA.
INFERENCE_DEVICE="cuda"

# Shared training settings. Epochs and dataset sizes stay in each training script.
BATCH_SIZE=2048
NUM_WORKERS=4
RANDOM_SEED=0

# Derived paths. Edit these only when using a custom directory layout.
ROBOT_CONFIG="configs/${ROBOT_NAME}.json"
ROBOT_DATA_DIR="${DATA_ROOT}/${ROBOT_NAME}"
CONTACT_DATA_DIR="${DATA_ROOT}/global_data"
RETARGETING_CHECKPOINT="${CHECKPOINT_ROOT}/${ROBOT_NAME}.pth"
CONTACT_CHECKPOINT="${CHECKPOINT_ROOT}/contact.pth"
PREDICTION_DIR="${OUTPUT_ROOT}/${ROBOT_NAME}"
