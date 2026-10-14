#!/usr/bin/env bash
# One environment for all three models.
#
#   bash setup/install.sh
#   conda activate archworld
#
# The models' own requirements pin incompatible torch and numpy versions, so
# each is installed with --no-deps and the shared versions below are what
# actually runs. This is the configuration the published numbers came from.
set -euo pipefail

# `conda activate` is a shell function; a non-interactive script has to load
# conda's profile hook before it can call it.
command -v conda >/dev/null 2>&1 || { echo "conda not found; install Miniconda first." >&2; exit 1; }
source "$(conda info --base)/etc/profile.d/conda.sh"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPS_DIR="${ARCHWORLD_DEPS:-$REPO_ROOT/third_party}"
ENV_NAME="${ARCHWORLD_ENV:-archworld}"
mkdir -p "$DEPS_DIR"

clone_at() {
    local url="$1" dir="$2" commit="$3"
    [ -d "$dir/.git" ] || { echo "  cloning $(basename "$dir")"; git clone --quiet "$url" "$dir"; }
    git -C "$dir" fetch --quiet origin "$commit" 2>/dev/null || true
    git -C "$dir" checkout --quiet "$commit"
}

if conda env list | grep -qE "^${ENV_NAME}\s"; then
    echo "  conda env '$ENV_NAME' already exists"
else
    echo "  creating conda env '$ENV_NAME'"
    conda create -y -q -n "$ENV_NAME" python=3.12 >/dev/null
fi
conda activate "$ENV_NAME"

echo "  installing torch"
pip install -q torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121

# Each model is pinned to the commit this code was developed against.
clone_at https://github.com/facebookresearch/vggt.git            "$DEPS_DIR/vggt"             97bbde571faddde3ace3cfa7724a20448026c4c8
clone_at https://github.com/ByteDance-Seed/Depth-Anything-3.git  "$DEPS_DIR/Depth-Anything-3" 41736238f5bced4debf3f2a12375d2466874866d
clone_at https://github.com/yyfz/Pi3.git                         "$DEPS_DIR/Pi3"              9fa3ddb3f8d53041f8b2738df404f62223bbaa7b

echo "  installing the models"
for d in vggt Depth-Anything-3 Pi3; do
    pip install -q --no-deps -e "$DEPS_DIR/$d"
done

echo "  installing the evaluation dependencies"
pip install -q -r "$REPO_ROOT/requirements.txt"

echo
echo "done. Checkpoints download on first run."
echo "  conda activate $ENV_NAME"
echo "  python intervention/run.py --model da3 --root data --masks-dir masks"
echo
echo "To generate the element masks ArchAttend needs:  bash setup/sam3.sh"
