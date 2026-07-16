SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
export PYTHONPATH="${SCRIPT_DIR}/ViPSAM_:${SCRIPT_DIR}:${PYTHONPATH}"

DATASET="${1:-smc}" # smc | apis | hecktor
python train.py --config ViPSAM_/configs/train_config_${DATASET}.json

# bash train.sh smc | apis | hecktor