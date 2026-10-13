# The login and config mounts of the agent's sandbox.
#
# Sourced by agent_sandbox.sh; no CLI, credential reads or network activity, so tests can run it on synthetic files.
roboarcade_auth_directory() {
    local runtime=$1 root=$2 entry relative
    [ -d "$root/store" ] || { echo 'Dedicated auth root must contain store/' >&2; return 2; }
    for entry in "$root"/* "$root"/.[!.]* "$root"/..?* "$root/store"/* "$root/store"/.[!.]* "$root/store"/..?*; do
        [ -e "$entry" ] || [ -L "$entry" ] || continue
        [ ! -L "$entry" ] || { echo 'Auth-only store must not contain symlinks' >&2; return 2; }
        relative=${entry#"$root"/}
        case "$runtime:$relative" in
            *:store|claude:store.lock|claude:store/.credentials.json|claude:store/.oauth_refresh.lock|claude:store/.credentials.json.tmp.*|codex:store/auth.json) ;;
            *) echo 'Auth-only store contains an unexpected entry; use a dedicated login directory' >&2; return 2 ;;
        esac
    done
}

roboarcade_codex_config_mounts() {
    local auth_root
    CODEX_CFG=${CODEX_HOME:?CODEX_HOME must name the private seed state directory}
    [ -d "$CODEX_CFG" ] && [ -L "$CODEX_CFG/auth.json" ] || {
        echo 'Codex needs private CODEX_HOME with an auth.json link to a dedicated auth store' >&2; return 2;
    }
    CODEX_AUTH=$(readlink -f "$CODEX_CFG/auth.json") || return 2
    [ -f "$CODEX_AUTH" ] && [ "${CODEX_AUTH##*/}" = auth.json ] && [ "$(basename "${CODEX_AUTH%/*}")" = store ] || {
        echo 'Codex login must be provisioned at a dedicated auth-root/store/auth.json' >&2; return 2;
    }
    auth_root=$(dirname "${CODEX_AUTH%/*}")
    roboarcade_auth_directory codex "$auth_root" || return 2
    CONFIG_MOUNT=(--bind "$CODEX_CFG" "$CODEX_CFG" --bind "$auth_root" "$auth_root")
    PROJECTS_MOUNT=()
}

roboarcade_claude_config_mounts() {
    local projects_dir=${1:-} state_dir="" auth_root
    CLAUDE_CFG=/tmp/home/.claude
    CLAUDE_AUTH_STORE=""
    CONFIG_MOUNT=(--perms 0700 --tmpfs "$CLAUDE_CFG")
    PROJECTS_MOUNT=(--dir "$CLAUDE_CFG/projects")
    if [ -n "$projects_dir" ]; then
        (umask 077; mkdir -p "$projects_dir") || return 2
        projects_dir=$(cd "$projects_dir" && pwd -P) || return 2
        (umask 077; mkdir -p "${projects_dir}.runtime") || return 2
        state_dir=$(cd "${projects_dir}.runtime" && pwd -P) || return 2
        CONFIG_MOUNT=(--bind "$state_dir" "$CLAUDE_CFG")
        PROJECTS_MOUNT=(--bind "$projects_dir" "$CLAUDE_CFG/projects")
    fi
    if [ -z "${ANTHROPIC_API_KEY:-}${ANTHROPIC_AUTH_TOKEN:-}${CLAUDE_CODE_OAUTH_TOKEN:-}" ]; then
        auth_root=${ROBOARCADE_CLAUDE_AUTH_DIR:-}
        [ -n "$auth_root" ] && [ -d "$auth_root/store" ] || {
            echo 'Claude OAuth requires a separately provisioned ROBOARCADE_CLAUDE_AUTH_DIR/store; personal config is never mounted' >&2; return 2;
        }
        auth_root=$(cd "$auth_root" && pwd -P) || return 2
        if [ -n "${CLAUDE_CONFIG_DIR:-}" ] && [ "$(readlink -m "$CLAUDE_CONFIG_DIR")" != "$auth_root/store" ]; then
            echo 'OAuth config/plan selector must name ROBOARCADE_CLAUDE_AUTH_DIR/store; refusing a mismatched login' >&2; return 2
        fi
        roboarcade_auth_directory claude "$auth_root" || return 2
        [ -f "$auth_root/store/.credentials.json" ] || { echo 'Auth-only store has no credential file; provision a separate login first' >&2; return 2; }
        CLAUDE_AUTH_STORE=/tmp/roboarcade-auth/store
        CONFIG_MOUNT+=(--bind "$auth_root" /tmp/roboarcade-auth)
    fi
}
