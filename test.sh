SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
export PYTHONPATH="${SCRIPT_DIR}/ViPSAM_:${SCRIPT_DIR}:${PYTHONPATH}"

DATASET="${1:-smc}"  # smc | apis | hecktor
python test.py --config "ViPSAM_/configs/test_config_${DATASET}.json"
# bash test.sh smc | apis | hecktor