#!/usr/bin/env bash
# Install the RGBD runtime and the agent runtimes selected by secrets/*.txt.
set -euo pipefail
cd "$(dirname "$0")/.."

case "${1:-}" in
    ""|--via) ;;
    --help|-h) echo 'Usage: bash scripts/setup.sh [--via]'; exit 0 ;;
    *) echo 'Usage: bash scripts/setup.sh [--via]' >&2; exit 2 ;;
esac
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
providers=$(python - <<'PY'
from core.accounts import api_key, require_api_key
require_api_key('hf')
providers = [p for p in ('claude', 'codex', 'antigravity') if api_key(p)]
if not providers:
    raise SystemExit('Add a key to secrets/anthropic_key.txt, secrets/openai_key.txt, or secrets/gemini_key.txt.')
print(' '.join(providers))
PY
)
packages=()
command -v bwrap >/dev/null || packages+=(bubblewrap)
if [[ " $providers " == *" codex "* ]] && ! command -v codex >/dev/null; then
    command -v npm >/dev/null || packages+=(nodejs npm)
fi
if [ "${#packages[@]}" -gt 0 ]; then
    command -v apt-get >/dev/null || {
        echo "Install these system packages, then rerun setup: ${packages[*]}" >&2; exit 1;
    }
    elevate=()
    [ "$(id -u)" = 0 ] || elevate=(sudo)
    "${elevate[@]}" apt-get update
    "${elevate[@]}" apt-get install -y "${packages[@]}"
fi
unshare -Urn true || { echo 'Enable unprivileged user namespaces for the sandbox.' >&2; exit 1; }
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
            python -c 'from core.accounts import codex_api_home; codex_api_home()'
            ;;
        antigravity)
            if [ ! -x data/antigravity-sdk/bin/python ]; then
                uv venv --python "$(readlink -f .venv/bin/python)" data/antigravity-sdk
            fi
            uv pip install --python data/antigravity-sdk/bin/python google-antigravity==0.1.22
            ;;
    esac
done
python scripts/sync_assets.py pull
if [ "${1:-}" = --via ]; then
    python -m playwright install --with-deps chromium
fi
echo 'Ready. Run source set_env.sh, then:'
echo 'python -m core.run_eval --task tidy_counter --model MODEL_ID --seeds 1 --render mujoco --no-video'
