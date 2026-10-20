#!/usr/bin/env bash
# Builds the agent's sandbox and runs the agent inside it.
#
# Usage: agent_sandbox.sh BASE_PORT [--pkg DIR] -- claude -p ...
#        agent_sandbox.sh --check [--pkg DIR]   # probe.sh, same mounts
# The box is an allowlist: an empty tmpfs root plus the paths in MOUNTS, nothing else;
# core/ shows only the MCP server's import surface (SANDBOX_IMPORT_FILES in
# core/agent_sandbox/__init__.py), read-only. The env is pruned to an allowlist below:
# secrets reach the box by inheritance, never argv.
set -euo pipefail

MODE_CHECK=0
if [ "${1:-}" = "--check" ]; then MODE_CHECK=1; shift; else BASE_PORT=$1; shift; fi
PKG=""
while [ $# -gt 0 ] && [ "$1" != "--" ]; do
    case "$1" in
        --pkg)    PKG=$2; shift 2 ;;
        *) echo "agent_sandbox.sh: unknown arg $1" >&2; exit 2 ;;
    esac
done
[ "${1:-}" = "--" ] && shift
REPO=$(cd "$(dirname "$0")/../.." && pwd)
IS_AGY=0
IS_AGY_SDK=0
[ "${1:-}" = agy-sdk ] && IS_AGY_SDK=1
[[ "${1:-}" = agy || "$IS_AGY_SDK" = 1 ]] && IS_AGY=1
IS_CODEX=0
[ "${1:-}" = codex ] && IS_CODEX=1
BWRAP_BIN=${ROBOARCADE_BWRAP:-bwrap}
CONFIG_MOUNT=(); PROJECTS_MOUNT=()
source "$REPO/core/agent_sandbox/config_mounts.sh"
if [ "$IS_AGY" = 1 ]; then
    AGY_ROOT=${ROBOARCADE_AGY_STATE:?Missing per-seed Antigravity state}
    CONFIG_MOUNT=(--bind "$AGY_ROOT/state" /tmp/home/.gemini/antigravity-cli
                  --ro-bind "$AGY_ROOT/config/settings.json" /tmp/home/.gemini/antigravity-cli/settings.json
                  --ro-bind "$AGY_ROOT/config/mcp_config.json" /tmp/home/.gemini/config/mcp_config.json
                  --ro-bind "$AGY_ROOT/config/robot.md" /tmp/home/.gemini/config/agents/robot.md
                  --ro-bind "$AGY_ROOT/config/sdk.json" /tmp/home/.gemini/config/roboarcade-sdk.json
                  --dir /tmp/home/workspace)
elif [ "$IS_CODEX" = 1 ]; then
    roboarcade_codex_config_mounts
else
    roboarcade_claude_config_mounts "${ROBOARCADE_PROJECTS_DIR:-}"
fi
PY="$REPO/.venv/bin/python"
BR="$REPO/core/agent_sandbox/connect_sandbox_to_sim_and_network.py"
mapfile -t SANDBOX_IMPORT_FILES < <(cd "$REPO" && "$PY" -c "from core.agent_sandbox import SANDBOX_IMPORT_FILES as f; print(*f, sep=chr(10))")
IMPORT_BIND=()
for f in "${SANDBOX_IMPORT_FILES[@]}"; do
    IMPORT_BIND+=(--ro-bind "$REPO/core/$f" "$REPO/core/$f")
done
PYEXE=$(readlink -f "$PY"); PYHOME=$(dirname "$PYEXE"); PYLIB=$(dirname "$PYHOME")/lib
if [ "$IS_AGY_SDK" = 1 ]; then
    SDK_PY=${ROBOARCADE_AGY_PYTHON:?Set ROBOARCADE_AGY_PYTHON to the isolated Antigravity SDK Python}
    SDK_ENV=$(cd "$(dirname "$SDK_PY")/.." && pwd)
    [ "$(readlink -f "$SDK_PY")" = "$PYEXE" ] || { echo "SDK and evaluator must use the same base Python interpreter" >&2; exit 2; }
    "$SDK_PY" -c 'import importlib.metadata; assert importlib.metadata.version("google-antigravity") == "0.1.22"'
    AGENT_BIND=(--ro-bind "$SDK_ENV" "$SDK_ENV"
                --ro-bind "$REPO/core/antigravity_sdk.py" "$REPO/core/antigravity_sdk.py")
    set -- "$SDK_PY" "$REPO/core/antigravity_sdk.py" "${@:2}"
    export ROBOARCADE_AGY_SDK=1
elif [ "$IS_AGY" = 1 ]; then
    AGY_BIN=$(command -v agy)
    AGENT_BIND=(--ro-bind "$(readlink -f "$AGY_BIN")" /opt/roboarcade/bin/agy)
elif [ "$IS_CODEX" = 1 ]; then
    if [ "$MODE_CHECK" = 1 ]; then
        CODEX_BIN="$REPO/data/sc/bin/codex"
    else
        CODEX_BIN=$(command -v codex || true)
        [ -z "$CODEX_BIN" ] || CODEX_BIN=$(readlink -f "$CODEX_BIN")
    fi
    [ -x "$CODEX_BIN" ] \
        || { echo "agent_sandbox.sh: codex binary missing" >&2; exit 2; }
    if [ "$(basename "$CODEX_BIN")" = codex.js ]; then
        CODEX_DIR=$(cd "$(dirname "$CODEX_BIN")/../node_modules/@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl/bin" && pwd)
    else
        CODEX_DIR=$(dirname "$CODEX_BIN")
    fi
    [ -x "$CODEX_DIR/codex-code-mode-host" ] \
        || { echo "agent_sandbox.sh: codex-code-mode-host missing beside codex" >&2; exit 2; }
    [ "$MODE_CHECK" = 0 ] || CODEX_VERSION=$("$CODEX_DIR/codex" --version)
    AGENT_BIND=(--ro-bind "$CODEX_DIR/codex" /opt/roboarcade/bin/codex
                --ro-bind "$CODEX_DIR/codex-code-mode-host" /opt/roboarcade/bin/codex-code-mode-host)
elif CLAUDE_BIN=$(command -v claude); then
    AGENT_BIND=(--ro-bind "$(readlink -f "$CLAUDE_BIN")" /opt/roboarcade/bin/claude)
elif [ "$MODE_CHECK" = 0 ]; then
    echo "agent_sandbox.sh: agent binary not on PATH" >&2; exit 2
else
    AGENT_BIND=()
fi
COMPACT_PROMPT=$REPO/core/compaction/prompt.txt
RUNDIR=${XDG_RUNTIME_DIR:-}
[ -n "$RUNDIR" ] && [ -d "$RUNDIR" ] && [ -w "$RUNDIR" ] || RUNDIR=${TMPDIR:-/tmp}
RUN=$(mktemp -d "$RUNDIR/roboarcade-iso.XXXXXX")
PROXY_PORT=18888

PKG_BIND=(); PKG_MODE=none
if [ -n "$PKG" ]; then
    PKG=$(cd "$PKG" && pwd)
    PKG_BIND=(--ro-bind "$PKG" "$PKG"); PKG_MODE=ro
fi

EXTRA_BIND=()
[ -n "${ROBOARCADE_SHOTS_DIR:-}" ] && [ -d "$ROBOARCADE_SHOTS_DIR" ] \
    && EXTRA_BIND+=(--bind "$ROBOARCADE_SHOTS_DIR" "$ROBOARCADE_SHOTS_DIR")
IFS=: read -ra _instr <<< "${ROBOARCADE_INSTRUCT_FILES:-}"
for f in "${_instr[@]}"; do
    [ -f "$f" ] && EXTRA_BIND+=(--ro-bind "$f" "$f")
done
if [ "$IS_CODEX" = 1 ]; then
    [ -d "${ROBOARCADE_AGENT_CWD:-}" ] \
        || { echo "agent_sandbox.sh: ROBOARCADE_AGENT_CWD is missing" >&2; exit 2; }
    EXTRA_BIND+=(--ro-bind "$ROBOARCADE_AGENT_CWD" "$ROBOARCADE_AGENT_CWD")
fi
CHECK_BIND=()
[ "$MODE_CHECK" = 1 ] && CHECK_BIND=(--ro-bind "$REPO/core/agent_sandbox/probe.sh" "$REPO/core/agent_sandbox/probe.sh")

BWRAP_BIN=$(command -v "$BWRAP_BIN")
for v in $(compgen -e); do
    case " MCP_TIMEOUT ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN CLAUDE_CODE_OAUTH_TOKEN CODEX_HOME GEMINI_API_KEY " in
        *" $v "*) ;;
        *) [[ "$v" == ROBOARCADE_* ]] || unset "$v" 2>/dev/null || true ;;
    esac
done
export PATH="/opt/roboarcade/bin:/usr/bin:/bin" HOME=/tmp/home TMPDIR=/tmp LANG=C.UTF-8
if [ "$IS_AGY" = 1 ]; then
    export ROBOARCADE_EGRESS_HOSTS=generativelanguage.googleapis.com
    API_HOST=generativelanguage.googleapis.com
elif [ "$IS_CODEX" = 1 ]; then
    export ROBOARCADE_EGRESS_HOSTS=chatgpt.com,auth.openai.com,api.openai.com
    [ "$MODE_CHECK" = 0 ] || export ROBOARCADE_CODEX_VERSION="$CODEX_VERSION"
    API_HOST=chatgpt.com
else
    export CLAUDE_CONFIG_DIR="$CLAUDE_CFG" ROBOARCADE_EGRESS_HOSTS=api.anthropic.com,platform.claude.com
    if [ -n "$CLAUDE_AUTH_STORE" ]; then
        export CLAUDE_SECURESTORAGE_CONFIG_DIR="$CLAUDE_AUTH_STORE"
    fi
    API_HOST=api.anthropic.com
fi

pids=()
cleanup() { kill "${pids[@]}" 2>/dev/null || true; rm -rf "$RUN"; }
trap cleanup EXIT INT TERM

MOUNTS=(
    --tmpfs /
    --ro-bind /usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu
    --ro-bind /usr/lib64 /usr/lib64
    --ro-bind /usr/bin/bash /usr/bin/bash --ro-bind /usr/bin/sh /usr/bin/sh
    --ro-bind /usr/bin/cat /usr/bin/cat --ro-bind /usr/bin/curl /usr/bin/curl
    --ro-bind /usr/bin/grep /usr/bin/grep --ro-bind /usr/bin/ls /usr/bin/ls
    --ro-bind /usr/bin/rm /usr/bin/rm --ro-bind /usr/bin/touch /usr/bin/touch
    --symlink usr/lib /lib --symlink usr/lib64 /lib64 --symlink usr/bin /bin
    --ro-bind /etc/hosts /etc/hosts
    --ro-bind /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt
    --ro-bind-try /etc/localtime /etc/localtime
    --dev /dev --proc /proc
    --tmpfs /tmp --dir /tmp/home
    --ro-bind "$PYEXE" "$PYEXE" --symlink "$(basename "$PYEXE")" "$PYHOME/python3"
    --ro-bind "$PYLIB/$(basename "$PYEXE")" "$PYLIB/$(basename "$PYEXE")"
    --tmpfs "$PYLIB/$(basename "$PYEXE")/site-packages"
    --remount-ro "$PYLIB/$(basename "$PYEXE")/site-packages"
    --ro-bind "$PYLIB/libz.so.1" "$PYLIB/libz.so.1" --ro-bind "$PYLIB/libbz2.so.1.0" "$PYLIB/libbz2.so.1.0"
    --ro-bind "$PYLIB/libcrypto.so.3" "$PYLIB/libcrypto.so.3" --ro-bind "$PYLIB/libssl.so.3" "$PYLIB/libssl.so.3"
    --ro-bind "$PYLIB/libffi.so.8" "$PYLIB/libffi.so.8" --ro-bind "$PYLIB/liblzma.so.5" "$PYLIB/liblzma.so.5"
    --ro-bind "$PYLIB/libuuid.so.1" "$PYLIB/libuuid.so.1" --ro-bind "$PYLIB/libgcc_s.so.1" "$PYLIB/libgcc_s.so.1"
    --ro-bind "$PYLIB/libstdc++.so.6" "$PYLIB/libstdc++.so.6"
    --ro-bind-try "$PYLIB/libsqlite3.so.0" "$PYLIB/libsqlite3.so.0"
    --ro-bind-try "$PYLIB/libreadline.so.8" "$PYLIB/libreadline.so.8"
    --ro-bind-try "$PYLIB/libncursesw.so.6" "$PYLIB/libncursesw.so.6"
    --ro-bind-try "$PYLIB/libpanelw.so.6" "$PYLIB/libpanelw.so.6"
    --ro-bind-try "$PYLIB/libtinfow.so.6" "$PYLIB/libtinfow.so.6"
    --ro-bind-try "$PYLIB/libexpat.so.1" "$PYLIB/libexpat.so.1"
    --ro-bind-try "$PYLIB/libcrypt.so.2" "$PYLIB/libcrypt.so.2"
    --ro-bind-try "$PYLIB/libnsl.so.3" "$PYLIB/libnsl.so.3"
    "${AGENT_BIND[@]}"
    --ro-bind "$REPO/.venv" "$REPO/.venv"
    "${IMPORT_BIND[@]}"
    --ro-bind "$BR" "$BR"
    --ro-bind "$COMPACT_PROMPT" "$COMPACT_PROMPT"
    --ro-bind "$REPO/.mcp.json" "$REPO/.mcp.json"
    "${CHECK_BIND[@]}"
    "${PKG_BIND[@]}"
    "${EXTRA_BIND[@]}"
    --bind "$RUN" "$RUN"
    "${CONFIG_MOUNT[@]}"
    "${PROJECTS_MOUNT[@]}"
    --remount-ro /
    --unshare-net
    --unshare-pid
    --die-with-parent
)
[ "$IS_AGY" = 0 ] || MOUNTS+=(--chdir /tmp/home/workspace)

if [ "$MODE_CHECK" = 1 ]; then
    "$BWRAP_BIN" "${MOUNTS[@]}" \
        -- bash "$REPO/core/agent_sandbox/probe.sh" "$REPO" "$PKG" "$PKG_MODE" "${CODEX_AUTH:-}"
    exit
fi

PAIRS=("$PROXY_PORT=$RUN/proxy.sock")
for i in 0 1 2 3; do PAIRS+=("$((BASE_PORT + i))=$RUN/p$i.sock"); done
"$PY" "$BR" host "$PROXY_PORT" "${PAIRS[@]}" & pids+=($!)

"$BWRAP_BIN" "${MOUNTS[@]}" \
    --setenv HTTP_PROXY  "http://127.0.0.1:$PROXY_PORT" \
    --setenv HTTPS_PROXY "http://127.0.0.1:$PROXY_PORT" \
    --setenv http_proxy  "http://127.0.0.1:$PROXY_PORT" \
    --setenv https_proxy "http://127.0.0.1:$PROXY_PORT" \
    --setenv NO_PROXY    "127.0.0.1,localhost" \
    --setenv no_proxy    "127.0.0.1,localhost" \
    -- "$PY" "$BR" box "${PAIRS[@]}" -- bash -c '
        REPO=$1; API_HOST=$2; shift 2
        [ -e "$REPO/assets/tasks" ] && { echo "FATAL: assets visible" >&2; exit 1; }
        [ -e "$REPO/tasks" ] && { echo "FATAL: task packs visible" >&2; exit 1; }
        [ -e "$REPO/interfaces" ] && { echo "FATAL: shipped interfaces visible" >&2; exit 1; }
        [ -e "$REPO/core/sim_env_server.py" ] && { echo "FATAL: sim internals visible" >&2; exit 1; }
        [ -s "$REPO/CLAUDE.md" ] && { echo "FATAL: repo CLAUDE.md visible" >&2; exit 1; }
        curl -s --noproxy "*" --max-time 3 https://example.com >/dev/null 2>&1 \
            && { echo "FATAL: direct egress works" >&2; exit 1; }
        curl -sS -o /dev/null --max-time 10 --retry 2 --retry-all-errors --retry-delay 1 "https://$API_HOST" \
            || { echo "FATAL: proxy path to $API_HOST broken" >&2; exit 1; }
        exec "$@"
    ' -- "$REPO" "$API_HOST" "$@" &
BWRAP_PID=$!
pids+=("$BWRAP_PID")
wait "$BWRAP_PID"
