#!/usr/bin/env bash
# Set up SAM 3, used only to generate the element masks ArchAttend steers.
#
#   bash setup/sam3.sh
#   conda activate archworld-sam3
#   python intervention/make_masks.py --root data --out masks
#
# SAM 3 wants torch 2.7 and numpy<2, which conflict with the 2.3.1 / 2.2.6 the
# evaluation runs on, so it gets its own environment. It is only needed once,
# to generate the masks; evaluation itself never loads it.
set -euo pipefail
command -v conda >/dev/null 2>&1 || { echo "conda not found; install Miniconda first." >&2; exit 1; }
source "$(conda info --base)/etc/profile.d/conda.sh"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPS_DIR="${ARCHWORLD_DEPS:-$REPO_ROOT/third_party}"
mkdir -p "$DEPS_DIR"

clone_at() {
    local url="$1" dir="$2" commit="$3"
    [ -d "$dir/.git" ] || { echo "  cloning $(basename "$dir")"; git clone --quiet "$url" "$dir"; }
    git -C "$dir" fetch --quiet origin "$commit" 2>/dev/null || true
    git -C "$dir" checkout --quiet "$commit"
}

if conda env list | grep -qE "^archworld-sam3\s"; then
    echo "  conda env 'archworld-sam3' already exists"
else
    conda create -y -q -n archworld-sam3 python=3.12 >/dev/null
fi
conda activate archworld-sam3

# Must precede the editable install: sam3/model_builder.py imports pkg_resources
# at module level, and setuptools 82 removed it.
pip install -q "setuptools<82" wheel

echo "  installing torch"
pip install -q torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126

clone_at "https://github.com/facebookresearch/sam3.git" "$DEPS_DIR/sam3" \
         "11dec2936de97f2857c1f76b66d982d5a001155d"
pip install -q -e "$DEPS_DIR/sam3"
# Imported by model_builder but not declared by the package.
pip install -q einops psutil "numpy<2" pillow

cat <<'NOTE'

done, with one manual step left: the SAM 3 weights are gated.

  1. request access at https://huggingface.co/facebook/sam3
  2. once approved:  hf auth login     (or export HF_TOKEN=...)
  3. conda activate archworld-sam3
     python intervention/make_masks.py --root data --out masks

The 3.45 GB checkpoint downloads on the first call.
NOTE
