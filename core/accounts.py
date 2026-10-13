"""Which account an agent run bills: API key or subscription plan.

Covers the API key file, the registered subscription plans, their rate meters, and switching a Codex seed to a
plan with room when its window fills (agent_log.plan_limited reads the fill off the log)."""
import json
import logging
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import urllib.request

from core.agent_log import codex_session_file, codex_session_id
from core.agent_sandbox import ISOLATED_SCRIPT, REPO_ROOT, codex_seed_home, codex_text_cmd, require_codex_home

logger = logging.getLogger(__name__)

KEY_FILE = "secrets/anthropic_key.txt"

PLANS_FILE = {"claude": "secrets/claude_plans.txt", "codex": "secrets/codex_plans.txt"}

AUTH_ROOT = {"claude": "secrets/claude_auth", "codex": "secrets/codex_auth"}

LOGIN_FILE = {"claude": ".credentials.json", "codex": "auth.json"}

METER_MODEL = "gpt-5.6-sol"

PLAN_5H_TOKENS = 6_500_000

PLAN_7D_TOKENS = 45_000_000


def choose_auth(plan: str | None = None) -> dict:
    if plan:
        return {"auth": "plan", "reason": f"plan {plan}"}
    path = os.environ.get("ROBOARCADE_KEY_FILE") or KEY_FILE
    try:
        key = pathlib.Path(path).read_text().strip()
    except OSError:
        key = ""
    if not key:
        return {"auth": "plan", "reason": f"no key at {path}"}
    return {"auth": "api_key", "reason": f"key at {path}", "key": key}


def apply_auth(env: dict, decision: dict) -> dict:
    for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"):
        env.pop(k, None)
    if decision.get("auth") == "api_key":
        env["ANTHROPIC_API_KEY"] = decision["key"]
    return env


def plan_usage(config_dir: str | None = None) -> dict:
    try:
        creds = json.load(open(os.path.join(config_dir or os.path.expanduser("~/.claude"),
                                            ".credentials.json")))
        access = creds["claudeAiOauth"]["accessToken"]
        req = urllib.request.Request(
            "https://api.anthropic.com/api/oauth/usage",
            headers={"Authorization": f"Bearer {access}",
                     "anthropic-beta": "oauth-2025-04-20"})
        with urllib.request.urlopen(req, timeout=10) as r:
            d = json.load(r)
        out = {"five_hour_pct": (d.get("five_hour") or {}).get("utilization"),
               "seven_day_pct": (d.get("seven_day") or {}).get("utilization")}
        extra = d.get("extra_usage") or {}
        if extra.get("is_enabled"):
            out["extra_credits_used"] = extra.get("used_credits")
            out["extra_credits_limit"] = extra.get("monthly_limit")
        return out
    except Exception:
        return {}


def read_plans(runtime: str) -> dict:
    try:
        lines = pathlib.Path(PLANS_FILE[runtime]).read_text().splitlines()
    except OSError:
        return {}
    return dict(line.split(None, 1) for line in lines
                if line.strip() and not line.lstrip().startswith("#"))


def codex_rate_limits(shared_home: str) -> list:
    home = codex_seed_home("meter", shared_home)
    try:
        env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
        env["CODEX_HOME"] = home
        log = os.path.join(home, "probe.log")
        try:
            with open(log, "w") as f:
                subprocess.run(codex_text_cmd(METER_MODEL, "low", os.path.join(home, "probe.md"),
                                              cwd=home),
                               input="Reply with the single word ok.", text=True, stdout=f,
                               stderr=subprocess.STDOUT, cwd=home, env=env, timeout=180)
        except subprocess.TimeoutExpired:
            logger.warning("plan meter probe timed out; treating the meter as unreadable")
            return []
        session = codex_session_file(home, codex_session_id(log))
        limits = {}
        if session:
            with open(session) as f:
                for line in f:
                    if '"rate_limits"' in line:
                        limits = json.loads(line)["payload"].get("rate_limits") or limits
        with open(log) as f:
            if "hit your usage limit" in f.read():
                return [100.0]
        return [(limits.get(k) or {}).get("used_percent") for k in ("primary", "secondary")]
    finally:
        shutil.rmtree(home, ignore_errors=True)


def plan_meter(runtime: str, home: str) -> float | None:
    if runtime == "claude":
        usage = plan_usage(home)
        used = [usage.get("five_hour_pct"), usage.get("seven_day_pct")]
    else:
        used = codex_rate_limits(home)
    used = [x for x in used if x is not None]
    return max(used) if used else None


def pick_plan(runtime: str, name: str) -> tuple[str, str]:
    plans = read_plans(runtime)
    if not plans:
        raise SystemExit(f"no plans in {PLANS_FILE[runtime]}")
    if name != "auto" and name not in plans:
        raise SystemExit(f"no plan {name!r} in {PLANS_FILE[runtime]}: {', '.join(plans)}")
    names = list(plans) if name == "auto" else [name]
    for n in names:
        if not os.path.isfile(os.path.join(plans[n], LOGIN_FILE[runtime])):
            raise SystemExit(f"plan {n!r} has no login in {plans[n]}; run: python -m core.accounts login {runtime} {n}")
    if name != "auto":
        return name, plans[name]
    meters = {n: plan_meter(runtime, plans[n]) for n in names}
    best = min(names, key=lambda n: (meters[n] is None, meters[n] or 0.0))
    logger.info("plan meters (% of the fullest window used): "
                + ", ".join(f"{n} {'?' if m is None else f'{m:.0f}'}" for n, m in meters.items())
                + f"; picked {best}")
    return best, plans[best]


PLAN_FULL = 97


def codex_plan_with_room(exp) -> tuple[str, str] | None:
    plans = read_plans("codex") if exp.plan_auto else {exp.plan: exp.plan_dir}
    meters = {n: plan_meter("codex", d) for n, d in plans.items()}
    best = min(meters, key=lambda n: (meters[n] is None, meters[n] or 0.0))
    logger.info("Codex plan meters: " + ", ".join(
        f"{n} {'?' if m is None else f'{m:.0f}'}" for n, m in meters.items()))
    return (best, plans[best]) if (meters[best] or 0.0) < PLAN_FULL else None


def repoint_codex_login(home: str, plan_dir: str) -> None:
    link = os.path.join(home, "auth.json")
    os.remove(link)
    os.symlink(os.path.join(require_codex_home(plan_dir), "auth.json"), link)


def seed_auth(exp) -> dict:
    auth = choose_auth(exp.plan)
    if exp.plan and (plan_meter("claude", exp.plan_dir) or 0) >= 100:
        key = choose_auth()
        if key["auth"] == "api_key":
            key["reason"] = f"plan {exp.plan} window full; " + key["reason"]
            return key
    return auth


def store_refusal(runtime: str, root: str) -> str | None:
    check = subprocess.run(["bash", "-c", 'source "$1" && roboarcade_auth_directory "$2" "$3"', "_",
                            os.path.join(os.path.dirname(ISOLATED_SCRIPT), "config_mounts.sh"), runtime, root],
                           capture_output=True, text=True)
    if check.returncode:
        return check.stderr.strip() or "unexpected entry"
    if not os.path.isfile(os.path.join(root, "store", LOGIN_FILE[runtime])):
        return f"no {LOGIN_FILE[runtime]} in the store"
    return None


def login(runtime: str, name: str) -> str:
    root = os.path.join(REPO_ROOT, AUTH_ROOT[runtime], name)
    store = os.path.join(root, "store")
    os.makedirs(store, mode=0o700, exist_ok=True)
    if not os.path.isfile(os.path.join(store, LOGIN_FILE[runtime])):
        setup = tempfile.mkdtemp(prefix="roboarcade_login_")
        try:
            if runtime == "claude":
                subprocess.run(["claude"], env={**os.environ, "CLAUDE_CONFIG_DIR": setup,
                                                "CLAUDE_SECURESTORAGE_CONFIG_DIR": store})
            else:
                subprocess.run(["codex", "login", "--device-auth"], env={**os.environ, "CODEX_HOME": setup})
            written = os.path.join(setup, LOGIN_FILE[runtime])
            if os.path.isfile(written) and not os.path.isfile(os.path.join(store, LOGIN_FILE[runtime])):
                shutil.move(written, store)
        finally:
            shutil.rmtree(setup, ignore_errors=True)
    keep = {LOGIN_FILE[runtime], ".oauth_refresh.lock"}
    for entry in os.listdir(store):
        if entry not in keep and not entry.startswith(LOGIN_FILE[runtime] + ".tmp."):
            os.makedirs(root + ".setup", mode=0o700, exist_ok=True)
            shutil.move(os.path.join(store, entry), os.path.join(root + ".setup", entry))
    refusal = store_refusal(runtime, root)
    if refusal:
        raise SystemExit(f"{store}: {refusal}")
    registry = pathlib.Path(PLANS_FILE[runtime])
    if name not in read_plans(runtime):
        with registry.open("a") as f:
            f.write(f"{name} {os.path.relpath(store, REPO_ROOT)}\n")
    return store


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "login" or sys.argv[2] not in PLANS_FILE:
        raise SystemExit("usage: python -m core.accounts login claude|codex NAME")
    print(login(sys.argv[2], sys.argv[3]))

