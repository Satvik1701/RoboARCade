# Agent sandbox

The model under evaluation runs in a bubblewrap sandbox built by `core/agent_sandbox/agent_sandbox.sh`: an empty
tmpfs root plus an allowlist of mounts (its `MOUNTS` array; what is bound and why is on the [Implementation notes](code/IMPLEMENTATION_NOTES.md) page). Inside the box the agent sees
no network, no task files, no episode folder and only the four `core/` files its MCP server imports. The only ways
out are the Unix sockets of `core/agent_sandbox/connect_sandbox_to_sim_and_network.py`: the four ports of the agent's own
simulator, and an HTTP CONNECT proxy that allows the model API's name on port 443 and nothing else.
`core/agent_sandbox/probe.sh` runs inside the real box under `agent_sandbox.sh --check` and prints one line per
invariant. Requires `bwrap` and unprivileged user namespaces
(`unshare -Urn true`); on a machine without bubblewrap point `ROBOARCADE_BWRAP` at a vendored binary.

## Logins and configuration

`core/agent_sandbox/config_mounts.sh` mounts the agent's login and config. Both runtimes get private, per-seed
state: Claude's config at `/tmp/home/.claude` (a tmpfs, or the seed's own `.runtime` directory when the run keeps
state for a same-seed resume), Codex's `CODEX_HOME` as a seed-local temporary home. No personal config, instructions,
skills, history or other seed's state is ever mounted; `build_agent_cmd` passes `--setting-sources ""` so Claude reads
no settings beyond the explicit `--settings` argument.

Authentication is a separate, shared, read-mostly mount:

- API key or environment token: no login directory is mounted at all.
- Claude OAuth: `ROBOARCADE_CLAUDE_AUTH_DIR` (set by the launcher from the plan's registered `store/`) names an auth-only root, mounted whole at `/tmp/roboarcade-auth` with
  `CLAUDE_SECURESTORAGE_CONFIG_DIR` pointing at its `store/`. The store may hold only `.credentials.json`, the CLI's
  refresh lock and its atomic-write temporaries; any other entry or a symlink refuses the launch. A supplied
  `CLAUDE_CONFIG_DIR` must resolve to that same `store/`, so a plan selector cannot pick a different account. Without
  a provisioned store an OAuth launch fails closed; it never falls back to personal configuration.
- Codex: the seed home's `auth.json` is a symlink into the selected plan's auth-only root, whose store may hold only
  `auth.json`. Plan switching relinks it; the next launch mounts only the newly selected root.

`python -m core.accounts login claude|codex NAME` provisions a login that satisfies these rules (the Accounts page
has the details); never copy or move a live credential store. These checks run at launch; they do not watch the
filesystem afterwards, and a refresh that replaced the seed's symlink instead of the shared target would not be
caught until the next launch.

## The camera request boundary

The `camera_raw` tool (via_cam and, through it, via_bimanual_cam) treats a camera name as data: path separators,
URL delimiters and encodings are rejected before any request, only names the simulator's `env.json` lists are
fetched, and the JPEG fetch follows no redirects, caps the body at 8 MiB and 4096x4096, and decodes the image before
returning its bytes. An invalid answer is a text error, never server bytes mislabelled as an image.

