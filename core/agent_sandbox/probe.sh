#!/usr/bin/env bash
# Checks the sandbox's invariants from inside it.
#
# Runs inside the bwrap box (`agent_sandbox.sh --check`); one line per invariant:
#   name|PASS|FAIL|detail   and exit 0 iff all pass (read by the pytest)
# Args: REPO PKG_ABS PKG_MODE(ro|none) [CODEX_AUTH]. The caller may set
# ROBOARCADE_CANARY=<token> and plant files containing it under assets/, tasks/, data/
# outside the box.
shopt -s nullglob globstar
REPO=${1:?}; PKG=${2:-}; PKG_MODE=${3:-none}; CODEX_AUTH=${4:-}
rc=0
report() {
    local name=$1 want=$2 got=$3 detail=${4:-}
    if [ "$want" = "$got" ]; then printf '%s|PASS|%s\n' "$name" "$detail"
    else printf '%s|FAIL|%s\n' "$name" "$detail"; rc=1; fi
}
exists()   { [ -e "$1" ] && echo present || echo absent; }
canary_leaks() { [ -f "$1" ] && grep -q "${ROBOARCADE_CANARY:-__no_canary__}" "$1" 2>/dev/null && echo leaked || echo safe; }
can_write() { if touch "$1" 2>/dev/null; then rm -f "$1"; echo writable; else echo readonly; fi; }
grade_sheet_leaks() {
    local allowed=${ROBOARCADE_SHOTS_DIR:-} path
    for path in "$REPO/data/run_eval"/**/*; do
        if [ -n "$allowed" ] && [[ "$path" == "$allowed" || "$path" == "$allowed"/* ]]; then
            continue
        fi
        if [ -f "$path" ] || [ -L "$path" ]; then echo leaked; return; fi
    done
    echo safe
}

report answer_key.predicates   absent "$(exists "$REPO/core/grade_episode.py")"            "success predicate source"
report answer_key.tasks        absent "$(exists "$REPO/tasks")"                       "per-task success trees and init poses"
report answer_key.canary_assets safe  "$(canary_leaks "$REPO/assets/__sbx_canary__")" "planted secret in assets/"
report answer_key.canary_envs   safe  "$(canary_leaks "$REPO/tasks/__sbx_canary__")" "planted secret in tasks/"

report grade_sheet.data        absent "$(exists "$REPO/data/__sbx_canary__")" "verdict.json area masked"
report grade_sheet.run_eval    safe   "$(grade_sheet_leaks)"                   "prior eval verdicts"

report repo.readonly           readonly "$(can_write "$REPO/core/.sbxprobe")"           "core/ package code"
report repo.sim_internals      absent   "$(exists "$REPO/core/sim_env_server.py")"          "sim/controller source hidden"
report repo.interfaces_hidden  absent   "$(exists "$REPO/interfaces")"             "shipped interfaces hidden"
report repo.worktrees_hidden   absent   "$(exists "$REPO/.claude/worktrees")"              "git worktrees (full checkouts) hidden"
report root.readonly           readonly "$(can_write /.sbxprobe)"                          "tmpfs root remounted ro"
report repo.only_allowlist     absent   "$(exists "$REPO/pyproject.toml")"                 "unnamed repo files absent"
report host.home_absent        absent   "$(exists "$REPO/../.cache")"                       "sibling dirs outside the box absent"
report repo.claude_md_absent   absent   "$(exists "$REPO/CLAUDE.md")"                      "repo CLAUDE.md absent"
report repo.kernel_visible     present  "$(exists "$REPO/core/kernel.py")"              "sanctioned API source"

if [ -n "$PKG" ]; then
    report package.readable    present  "$(exists "$PKG/tools.py")" "interface loads"
    report package.readonly readonly "$(can_write "$PKG/.sbxprobe")" "eval can't tamper the package"
fi

if curl -s --noproxy '*' --max-time 4 https://1.1.1.1 >/dev/null 2>&1; then eg=egress; else eg=noegress; fi
report net.egress_blocked      noegress "$eg" "direct internet"

if [ -z "$CODEX_AUTH" ] && [ -z "${ROBOARCADE_AGY_STATE:-}" ]; then
    CFG=${CLAUDE_CONFIG_DIR:-$HOME/.claude}
    if [ -z "$(ls -A "$CFG/projects" 2>/dev/null)" ]; then mem=empty; else mem=populated; fi
    report memory.projects_masked  empty "$mem" "MEMORY.md / task hints"
    for name in CLAUDE.md settings.json rules skills commands agents history.jsonl file-history session-env; do
        report "memory.config_$name" absent "$(exists "$CFG/$name")" "personal config not mounted"
    done
    report memory.config_writable writable "$(can_write "$CFG/.sbxprobe")" "private runtime state"
fi

if [ -n "${ROBOARCADE_AGY_STATE:-}" ]; then
    CFG=$HOME/.gemini
    report antigravity.binary present "$(exists /opt/roboarcade/bin/agy)" "native CLI"
    report antigravity.state_writable writable "$(can_write "$CFG/antigravity-cli/.sbxprobe")" "seed-local history"
    report antigravity.settings_readonly readonly "$(can_write "$CFG/antigravity-cli/settings.json")" "fixed model and permissions"
    report antigravity.mcp_readonly readonly "$(can_write "$CFG/config/mcp_config.json")" "only robot MCP"
    report antigravity.agent_readonly readonly "$(can_write "$CFG/config/agents/robot.md")" "only interface guide"
    report antigravity.secrets_hidden absent "$(exists "$REPO/secrets")" "credential files hidden"
fi

if [ -n "$CODEX_AUTH" ]; then
    report codex.binary_version    "$ROBOARCADE_CODEX_VERSION" "$(codex --version 2>/dev/null)" "$ROBOARCADE_CODEX_VERSION"
    report codex.companion         present   "$(exists /opt/roboarcade/bin/codex-code-mode-host)" "matching companion"
    report codex.home_writable     writable  "$(can_write "$CODEX_HOME/.sbxprobe")"      "seed-local CODEX_HOME"
    report codex.auth_resolves     present   "$(exists "$CODEX_HOME/auth.json")"         "auth.json symlink reaches the shared login"
    mapfile -t _shared < <(ls -A "${CODEX_AUTH%/*}" 2>/dev/null)
    report codex.shared_home_masked auth.json "${_shared[*]}" "auth-only store has no sqlite/sessions"
    _homes=("${CODEX_HOME%/*}"/codex_home_*)
    report codex.other_homes_hidden 1        "${#_homes[@]}"                                   "no other seed's home under the same parent"
fi

exit $rc
