# Accounts

API-key billing is the default for all supported runtimes. Add your provider key
and HF token under `secrets/` and run `bash scripts/setup.sh`; see the
[quick start](../README.md#quickstart). Only runtimes with a supplied
key are installed. Gemini uses the isolated Antigravity SDK environment.

| Runtime | Key file | Optional environment override |
|---|---|---|
| Claude | `secrets/anthropic_key.txt` | `ANTHROPIC_API_KEY` |
| Codex | `secrets/openai_key.txt` | `OPENAI_API_KEY` |
| Antigravity | `secrets/gemini_key.txt` | `GEMINI_API_KEY` |
| HF asset downloads | `secrets/hf_key.txt` | `HF_TOKEN` |

Nonempty environment keys take precedence over files. Custom file paths use
`ROBOARCADE_KEY_FILE`, `ROBOARCADE_OPENAI_KEY_FILE`, `ROBOARCADE_GEMINI_KEY_FILE`,
and `ROBOARCADE_HF_KEY_FILE`, respectively.

Codex API credentials are provisioned noninteractively using
[`codex login --with-api-key`](https://developers.openai.com/codex/auth/), in a
dedicated `secrets/codex_api/store/` directory. Each seed gets private runtime state;
personal CLI settings and logins are never used. Key rotation is picked up on the
next evaluation. Gemini's backend details are in [Antigravity](antigravity.md).

An explicit `--plan NAME` selects a registered Claude or Codex subscription instead
of API-key billing. Without `--plan`, a missing provider key stops the run before
starting the simulator. API-key runs do not use subscription rate meters.

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

For a plan-only installation, put the HF token in `secrets/hf_key.txt` and run
`bash scripts/setup.sh --plan claude` or `bash scripts/setup.sh --plan codex`,
then `source set_env.sh`. No model API key is required. Add `--via` for Chromium.

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
