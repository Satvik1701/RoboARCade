#!/usr/bin/env bash
# Install the RGBD runtime and the agent runtimes selected by secrets/*.txt.
set -euo pipefail
cd "$(dirname "$0")/.."

via=0
plan_runtime=""
while [ $# -gt 0 ]; do
    case "$1" in
        --via) via=1; shift ;;
        --plan)
            case "${2:-}" in
                claude|codex) plan_runtime=$2; shift 2 ;;
                *) echo '--plan requires claude or codex' >&2; exit 2 ;;
            esac ;;
        --help|-h) echo 'Usage: bash scripts/setup.sh [--via] [--plan claude|codex]'; exit 0 ;;
        *) echo 'Usage: bash scripts/setup.sh [--via] [--plan claude|codex]' >&2; exit 2 ;;
    esac
done
[ "$(uname -s)-$(uname -m)" = Linux-x86_64 ] || {
    echo 'The sandbox requires Linux x86_64.' >&2; exit 1;
}
export PATH="$PWD/tools/codex/node_modules/.bin:$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
    installer=$(mktemp)
    curl -fsSL https://astral.sh/uv/install.sh -o "$installer"
    sh "$installer"
    rm "$installer"
fi
uv sync --frozen
source set_env.sh
providers=$(python - "$plan_runtime" <<'PY'
import sys
from core.accounts import api_key, require_api_key
require_api_key('hf')
providers = [p for p in ('claude', 'codex', 'antigravity') if api_key(p) or p == sys.argv[1]]
if not providers:
    raise SystemExit('Add a key to secrets/anthropic_key.txt, secrets/openai_key.txt, or secrets/gemini_key.txt.')
print(' '.join(providers))
PY
)
packages=()
command -v bwrap >/dev/null || packages+=(bubblewrap)
command -v xz >/dev/null || packages+=(xz-utils)
if [[ " $providers " == *" codex "* ]] && ! command -v codex >/dev/null; then
    command -v npm >/dev/null || packages+=(nodejs npm)
fi
if [ "${#packages[@]}" -gt 0 ]; then
    echo "Missing host prerequisites: ${packages[*]}." >&2
    echo 'Setup does not install system packages or use sudo. Ask your administrator to provide these, then rerun setup.' >&2
    exit 1
fi
# Ubuntu's AppArmor policy may allow Bubblewrap while denying raw unshare.
# Exercise the sandbox binary instead of rejecting a working installation.
bwrap --ro-bind / / --unshare-user --unshare-net --unshare-pid --die-with-parent /bin/true || {
    echo 'Bubblewrap cannot create the sandbox. Ask your administrator to allow its user namespaces.' >&2
    exit 1
}
for provider in $providers; do
    case "$provider" in
        claude)
            if ! command -v claude >/dev/null; then
                installer=$(mktemp)
                curl -fsSL https://claude.ai/install.sh -o "$installer"
                bash "$installer"
                rm "$installer"
            fi
            claude --version
            ;;
        codex)
            if ! command -v codex >/dev/null; then
                npm install --prefix tools/codex @openai/codex
            fi
            codex --version
            python -c 'from core.accounts import api_key, codex_api_home; api_key("codex") and codex_api_home()'
            ;;
        antigravity)
            if [ ! -x data/antigravity-sdk/bin/python ]; then
                uv venv --python "$(readlink -f .venv/bin/python)" data/antigravity-sdk
            fi
            uv pip install --python data/antigravity-sdk/bin/python google-antigravity==0.1.22
            ;;
    esac
done
mkdir -p tools
blender_dir=tools/blender-4.5.12-linux-x64
if [ ! -x "$blender_dir/blender" ]; then
    stage=$(mktemp -d tools/blender-install.XXXXXX)
    curl -fL https://download.blender.org/release/Blender4.5/blender-4.5.12-linux-x64.tar.xz \
        | tar -xJ -C "$stage"
    mv "$stage/blender-4.5.12-linux-x64" "$blender_dir"
    rmdir "$stage"
fi
if ! PYTHONPATH="$PWD/tools/blender-py" "$blender_dir/4.5/python/bin/python3.11" -c \
    'import mujoco, typing_extensions, etils, absl, glfw; from PIL import Image; assert mujoco.__version__ == "3.10.0"' \
    >/dev/null 2>&1; then
    "$blender_dir/4.5/python/bin/python3.11" -m pip install --target tools/blender-py --upgrade --no-deps \
        mujoco==3.10.0 pillow typing_extensions etils absl-py glfw
fi
python -c 'from core.render import available; ok, why = available(); assert ok, why'
python scripts/sync_assets.py pull
if [ "$via" = 1 ]; then
    python -m playwright install chromium
    python - <<'PY'
from playwright.sync_api import sync_playwright, Error
try:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        browser.close()
except Error as exc:
    raise SystemExit('Chromium could not start. VIA requires the host Chromium libraries; '
                     'ask your administrator to provide any missing dependencies.\n' + str(exc))
PY
fi
echo 'Ready. Run source set_env.sh, then:'
if [ -n "$plan_runtime" ]; then
    echo "python -m core.accounts login $plan_runtime myplan"
    echo 'python -m core.run_eval --model MODEL_ID --plan myplan --seeds 1'
else
    echo 'python -m core.run_eval --model MODEL_ID --seeds 1'
fi
