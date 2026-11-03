# Accounts

Every evaluation bills one account, decided once per run so every seed bills the same way: a subscription plan
named with `--plan`, or else `ANTHROPIC_API_KEY`, falling back to `secrets/anthropic_key.txt`
(another file path through `ROBOARCADE_KEY_FILE`). With neither, a sandboxed Claude run cannot start: the sandbox never mounts a personal
login. The code is `core/accounts.py`; the sandbox side is `core/agent_sandbox/config_mounts.sh`.

## API keys

Claude needs the native [Claude Code CLI](https://code.claude.com/docs/en/setup) and
`ANTHROPIC_API_KEY`. No subscription login is required. An explicit `--plan` takes
precedence over the environment key; the key file is used only when the environment key is empty.

For GPT models, install Codex and register an API-key login once:

```bash
npm install -g @openai/codex
export OPENAI_API_KEY="your-openai-api-key"
mkdir -p secrets/codex_auth/openai/store
printenv OPENAI_API_KEY | CODEX_HOME=secrets/codex_auth/openai/store codex login --with-api-key
python -m core.accounts login codex openai
python -m core.run_eval --task bowl_on_plate --model gpt-6-astra --plan openai --seeds 1 --render mujoco --no-video
```

This uses API-key billing even though the credential store is selected with `--plan`.
The `secrets/` directory is gitignored. Gemini setup is documented in
[Antigravity](antigravity.md).

## Plans

A plan is a subscription login kept in its own auth-only root: `secrets/claude_auth/<name>/store/.credentials.json`
for Claude, `secrets/codex_auth/<name>/store/auth.json` for Codex. The registries `secrets/claude_plans.txt` and
`secrets/codex_plans.txt` map a plan name to its store, one `name path` per line.

The sandbox mounts the store and nothing else of the login: Claude gets the root at `/tmp/roboarcade-auth` with the
store as its credential directory and a private per-seed config directory; a Codex seed home links the store's
`auth.json`. Any other entry in the store (the CLI's own `.claude.json`, `backups/`, a Codex `sessions/`) refuses the
launch, because a login made the ordinary way carries settings, instructions and history that must not reach the
model.

## Setting up a plan

```bash
python -m core.accounts login claude max_a     # opens the CLI: /login, pick the subscription, exit
python -m core.accounts login codex pro_f      # codex login --device-auth
```

The CLI runs with its configuration in a throwaway directory, so the store receives the credential file and nothing
else. The command then checks the store with the sandbox's own rule and adds the plan to the registry. Run it
again on an existing store to tidy it: anything beyond the credential file moves to `<root>.setup/` beside it, so
a login that was made with the CLI's config pointed at the store becomes usable without logging in again. Never
copy or move a live credential store; log in afresh instead.

## Choosing a plan

`--plan NAME` bills that plan for every seed. `--plan auto` picks the registered plan with the most room left in its
rate windows: a Claude plan's meters come from the login's usage endpoint, a Codex plan's from one cheap probe turn
(`gpt-5.6-sol`, no tools), which costs a few tokens on that plan.

During a run the launcher keeps the agent billing something that has room:

- A Codex seed that hits its plan's limit re-picks the plan with the most room (under `--plan auto`) and continues
  on it; when every registered plan is at 97 percent or more it holds and retries, with the episode clock paused.
- A Claude seed whose plan window is full falls back to the API key for the rest of the episode, if the key file
  exists; without a key it stays on the plan and waits.

The run's `meta.json` records which account each seed billed (`auth`), and the dashboard's cost page prices plan runs
at API-equivalent rates (see Cost accounting).
