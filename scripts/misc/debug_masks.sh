set -euo pipefail

export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

python3 -m multi_binaryalign.debug_masks "$@"
