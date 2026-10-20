"""Reads an agent's log: events, screenshots, counts, cost and failure signs.

Covers Claude stream-json logs and Codex session files: the events and screenshots the dashboard shows, tool and
turn counts, the condensed conversation.txt, token and cost metrics, and the failures the seed runner checks for
(auth, a toolless or detached MCP server, a stalled API, a full plan window).
"""
import base64
import hashlib
import json
import pathlib
import re
from datetime import datetime
from pathlib import Path

from core.kernel import UI_UNATTACHED


REPO = Path(__file__).resolve().parents[1]

SHOT_CACHE = REPO / "data" / ".dashboard_cache"



def transcript_file(folder):
    for pattern in ("antigravity.jsonl", "codex_session*.jsonl", "*claude*.log"):
        for c in sorted(Path(folder).glob(pattern)):
            if c.is_file():
                return c
    return None


def transcript(jsonl_path):
    p = Path(jsonl_path)
    try:
        st = p.stat()
    except OSError:
        return {"events": [], "shots": [], "n_wp": 0}
    side = SHOT_CACHE / f"{hashlib.sha256(str(p.resolve()).encode()).hexdigest()[:20]}.json"
    try:
        c = json.loads(side.read_text())
        if c.get("format_version") == 6 and c.get("size") == st.st_size and c.get("mtime") == st.st_mtime and "events" in c:
            return c
    except Exception:
        pass
    events, shots, names, wp, off = [], [], {}, 0, 0
    agy_messages, agy_calls = {}, set()

    def add_result(tool_id, blocks, is_error):
        if isinstance(blocks, list):
            for j, b in enumerate(blocks):
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "image":
                    shots.append([names.get(tool_id, "?"), start, len(raw), j])
                    events.append({"kind": "shot", "img": len(shots) - 1, "wp": wp,
                                   "tid": tool_id})
                elif b.get("type") == "text":
                    add_result(tool_id, b.get("text", ""), is_error)
            return
        if isinstance(blocks, str) and blocks.strip():
            events.append({"kind": "result", "text": blocks.strip(), "wp": wp,
                           "err": is_error, "tid": tool_id})

    for raw in open(p, "rb"):
        start, off = off, off + len(raw)
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        ts = obj.get("timestamp")
        try:
            ts_epoch = datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp() if ts else None
        except ValueError:
            ts_epoch = None
        n0 = len(events)
        if obj.get("event") == "step_update":
            step = obj["step_update"]
            tid = f"{step.get('conversation_id')}:{step.get('step_index')}"
            if step.get("step_type") == "agent_response" and step.get("text_delta"):
                if tid not in agy_messages:
                    agy_messages[tid] = {"kind": "says", "text": "", "wp": wp}
                    events.append(agy_messages[tid])
                agy_messages[tid]["text"] += step["text_delta"]
            info = step.get("tool_info") or {}
            if info:
                name = info.get("name") or step.get("tool_name", "")
                args = info.get("parameters") or {}
                if name == "call_mcp_tool":
                    name, args = args.get("ToolName", name), args.get("Arguments") or {}
                if tid not in agy_calls:
                    agy_calls.add(tid)
                    names[tid] = name
                    events.append({"kind": "does", "tool": name, "args": args, "wp": wp, "id": tid})
                    if "execute" in name:
                        wp += 1
                if step.get("state") in ("DONE", "ERROR"):
                    output = info.get("output")
                    if isinstance(output, dict):
                        output = output.get("content", json.dumps(output))
                    add_result(tid, output, bool(info.get("error")))
                    if isinstance(output, str):
                        for media in re.findall(
                                r"\[Resource offloaded to file:///tmp/home/\.gemini/antigravity-cli/"
                                r"(brain/[^\]\s]+\.(?:jpg|png))\]", output):
                            shots.append([name, start, len(raw), media])
                            events.append({"kind": "shot", "img": len(shots) - 1, "wp": wp, "tid": tid})
                    if info.get("error"):
                        add_result(tid, info["error"].get("message", str(info["error"])), True)
        elif obj.get("event") == "compaction":
            events.append({"kind": "says", "text": "[context compacted]", "wp": wp})
        elif obj.get("type") == "assistant":
            for c in obj.get("message", {}).get("content", []):
                t = c.get("type")
                if t == "thinking" and c.get("thinking", "").strip():
                    events.append({"kind": "thinks", "text": c["thinking"].strip(), "wp": wp})
                elif t == "text" and c.get("text", "").strip():
                    events.append({"kind": "says", "text": c["text"].strip(), "wp": wp})
                elif t == "tool_use":
                    name = re.sub(r"^mcp__(roboarcade|robocua)__", "", c.get("name") or "")
                    names[c.get("id")] = name
                    events.append({"kind": "does", "tool": name, "args": c.get("input") or {},
                                   "wp": wp, "id": c.get("id")})
                    if "execute" in name:
                        wp += 1
        elif obj.get("type") == "user":
            for c in obj.get("message", {}).get("content", []):
                if c.get("type") == "tool_result":
                    add_result(c.get("tool_use_id"), c.get("content"), bool(c.get("is_error")))
        elif obj.get("subtype") == "compact_boundary":
            m = obj.get("compactMetadata") or {}
            events.append({"kind": "compact", "wp": wp,
                           "trigger": m.get("trigger"),
                           "pre": m.get("preTokens"),
                           "post": m.get("postTokens")})
        elif obj.get("type") == "event_msg" and (obj.get("payload") or {}).get("type") == "item_completed":
            it = obj["payload"].get("item") or {}
            kind = it.get("type")
            if kind == "AgentMessage":
                text = " ".join(b.get("text", "") for b in it.get("content") or [] if isinstance(b, dict))
                if text.strip():
                    events.append({"kind": "says", "text": text.strip(), "wp": wp})
            elif kind == "Reasoning":
                text = "\n".join(s for s in it.get("summary_text") or [] if isinstance(s, str))
                if text.strip():
                    events.append({"kind": "thinks", "text": text.strip(), "wp": wp})
            elif kind == "McpToolCall":
                name = it.get("tool") or ""
                names[it.get("id")] = name
                events.append({"kind": "does", "tool": name, "args": it.get("arguments") or {},
                               "wp": wp, "id": it.get("id")})
                if "execute" in name:
                    wp += 1
                res = it.get("result") or {}
                add_result(it.get("id"), res.get("content"), bool(res.get("isError")))
        elif obj.get("type") == "compacted":
            events.append({"kind": "compact", "wp": wp, "trigger": "auto", "pre": None, "post": None})
        for e in events[n0:]:
            e["t"] = ts_epoch
    out = {"size": st.st_size, "mtime": st.st_mtime, "format_version": 6, "events": events, "shots": shots, "n_wp": wp}
    SHOT_CACHE.mkdir(parents=True, exist_ok=True)
    side.write_text(json.dumps(out))
    return out


_SHOTS: dict = {}


def shot_bytes(jsonl_path, k):
    try:
        st = Path(jsonl_path).stat()
    except OSError:
        return None
    key, stamp = str(jsonl_path), (st.st_size, st.st_mtime)
    if _SHOTS.get(key, (None,))[0] != stamp:
        if len(_SHOTS) >= 64:
            del _SHOTS[next(iter(_SHOTS))]
        _SHOTS[key] = (stamp, transcript(jsonl_path)["shots"])
    shots = _SHOTS[key][1]
    if not 0 <= k < len(shots):
        return None
    _tool, off, length, j = shots[k]
    if isinstance(j, str):
        root = Path(jsonl_path).parent / "antigravity" / "state"
        media = (root / j).resolve()
        if not media.is_relative_to((root / "brain").resolve()) or not media.is_file():
            return None
        return ("image/png" if media.suffix == ".png" else "image/jpeg", media.read_bytes())
    try:
        with open(jsonl_path, "rb") as fh:
            fh.seek(off)
            obj = json.loads(fh.read(length))
    except Exception:
        return None
    blocks = [c.get("content") for c in obj.get("message", {}).get("content", []) or []
              if isinstance(c, dict) and isinstance(c.get("content"), list)]
    codex = ((obj.get("payload") or {}).get("item") or {}).get("result") or {}
    if isinstance(codex.get("content"), list):
        blocks.append(codex["content"])
    agy = (obj.get("step_update") or {}).get("tool_info") or {}
    output = agy.get("output")
    if isinstance(output, dict) and isinstance(output.get("content"), list):
        blocks.append(output["content"])
    elif isinstance(output, list):
        blocks.append(output)
    for content in blocks:
        b = content[j] if j < len(content) else None
        if isinstance(b, dict) and b.get("type") == "image":
            src = b.get("source") or b
            return (src.get("media_type") or src.get("mimeType") or "image/jpeg",
                    base64.b64decode(src.get("data", "")))
    return None


def transcript_tools(jsonl_path):
    p = Path(jsonl_path)
    try:
        st = p.stat()
    except OSError:
        return []
    side = SHOT_CACHE / f"{hashlib.sha256(str(p.resolve()).encode()).hexdigest()[:20]}.tools.json"
    try:
        c = json.loads(side.read_text())
        if c.get("size") == st.st_size and c.get("mtime") == st.st_mtime:
            return c["tools"]
    except Exception:
        pass
    tools = [e.get("tool", "") for e in transcript(jsonl_path)["events"] if e.get("kind") == "does"]
    SHOT_CACHE.mkdir(parents=True, exist_ok=True)
    side.write_text(json.dumps({"size": st.st_size, "mtime": st.st_mtime, "tools": tools}))
    return tools


def transcript_replies(jsonl_path):
    p = Path(jsonl_path)
    try:
        st = p.stat()
    except OSError:
        return None
    side = SHOT_CACHE / f"{hashlib.sha256(str(p.resolve()).encode()).hexdigest()[:20]}.replies.json"
    try:
        c = json.loads(side.read_text())
        if c.get("size") == st.st_size and c.get("mtime") == st.st_mtime:
            return c["replies"]
    except Exception:
        pass
    ids = set()
    with open(p, errors="replace") as fh:
        for line in fh:
            if '"assistant"' not in line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("type") == "assistant" and (d.get("message") or {}).get("id"):
                ids.add(d["message"]["id"])
    replies = len(ids) or None
    SHOT_CACHE.mkdir(parents=True, exist_ok=True)
    side.write_text(json.dumps({"size": st.st_size, "mtime": st.st_mtime, "replies": replies}))
    return replies


def condense(log_path: str) -> str:
    turns: list[list[str]] = []
    last_message_id = None
    with open(log_path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            m = d.get("message")
            m = m if isinstance(m, dict) else {}
            content = m.get("content") if isinstance(m.get("content"), list) else []
            if d.get("type") == "assistant":
                if m.get("id") != last_message_id:
                    last_message_id = m.get("id")
                    turns.append([])
                for b in content:
                    if b.get("type") == "text" and b["text"].strip():
                        turns[-1].append(b["text"].strip())
                    elif b.get("type") == "tool_use":
                        turns[-1].append(f"-> {b['name']}({json.dumps(b.get('input') or {})[:120]})")
            elif d.get("type") == "user" and turns:
                for b in content:
                    if b.get("type") == "tool_result":
                        c = b.get("content")
                        text = c if isinstance(c, str) else " ".join(
                            x.get("text", "") for x in c or () if x.get("type") == "text")
                        err = "ERROR " if b.get("is_error") else ""
                        turns[-1].append(f"   = {err}{' '.join(text.split())[:200]}")
                    elif b.get("type") == "text":
                        turns[-1].append(f"[summary] {' '.join(b['text'].split())[:600]}")
            elif d.get("subtype") == "compact_boundary":
                turns.append(["[context compacted]"])
                last_message_id = None
    return "\n\n".join("\n".join(t) for t in turns)


def condense_codex(path: str) -> str:
    turns: list[list[str]] = [[]]
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("type") == "compacted":
                turns += [["[context compacted]"], []]
                continue
            p = d.get("payload") or {}
            it = (p.get("item") or {}) if d.get("type") == "event_msg" and p.get("type") == "item_completed" else {}
            if it.get("type") == "AgentMessage":
                text = " ".join(b.get("text", "") for b in it.get("content") or [] if isinstance(b, dict))
                if text.strip():
                    turns.append([text.strip()])
            elif it.get("type") == "McpToolCall":
                turns[-1].append(f"-> {it.get('tool')}({json.dumps(it.get('arguments') or {})[:120]})")
                res = it.get("result") or {}
                c = res.get("content")
                text = c if isinstance(c, str) else " ".join(
                    x.get("text", "") for x in c or () if isinstance(x, dict) and x.get("type") == "text")
                err = "ERROR " if res.get("isError") else ""
                turns[-1].append(f"   = {err}{' '.join(text.split())[:200]}")
    return "\n\n".join("\n".join(t) for t in turns if t)


AUTH_FAILURE_MARKERS = ("not logged in", "please run /login", "run claude auth login",
                        "oauth token has been revoked", "oauth access token has expired",
                        "oauth session expired", "failed to authenticate",
                        "refresh token was revoked", "refresh token was invalidated",
                        "session has ended",
                        "token expired", "401 ", "invalid api key", "invalid_api_key",
                        "authentication_error")


API_FAULT_MARKERS = ("api error: 5", "overloaded")


def _last_result(log_path: str, session_id: str | None = None) -> dict:
    last = {}
    try:
        with open(log_path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if d.get("type") == "result" and (session_id is None or d.get("session_id") == session_id):
                    last = d
    except OSError:
        pass
    return last


def _last_codex_error(log_path: str) -> str:
    last = ""
    try:
        for line in open(log_path):
            event = json.loads(line)
            item = event.get("item") or {}
            if event.get("type") == "turn.failed":
                last = str((event.get("error") or {}).get("message") or event.get("error") or "")
            elif item.get("type") == "error":
                last = str(item.get("message") or "")
    except (OSError, ValueError):
        pass
    return last


def auth_failed(log_path: str, session_id: str | None = None) -> str | None:
    result = _last_result(log_path, session_id)
    last = (str(result.get("result") or "") if result.get("is_error") else "") or _last_codex_error(log_path)
    if last and any(m in last.lower() for m in AUTH_FAILURE_MARKERS):
        return last[:200]
    return None


def api_stalled(log_path: str) -> str | None:
    r = _last_result(log_path)
    msg, turns = str(r.get("result") or ""), r.get("num_turns") or 0
    if not msg:
        try:
            with open(log_path) as f:
                for line in f:
                    event = json.loads(line)
                    if event.get("type") == "turn.started":
                        turns = 0
                    elif event.get("type") == "item.completed":
                        turns += 1
                    elif event.get("type") == "turn.failed":
                        msg = str((event.get("error") or {}).get("message") or event.get("error") or "")
        except (OSError, ValueError):
            pass
    if turns <= 2 and any(m in msg.lower() for m in API_FAULT_MARKERS):
        return msg[:200]
    return None


PLAN_LIMIT_RE = re.compile(r"usage limit reached|hit your (?:\w+ )?limit")


def plan_limited(runtime: str, log_path: str, session_id: str | None, plan_dir: str | None = None) -> bool:
    if runtime == "codex":
        return bool(PLAN_LIMIT_RE.search(_last_codex_error(log_path).lower()))
    if PLAN_LIMIT_RE.search(str(_last_result(log_path, session_id).get("result") or "").lower()):
        return True
    try:
        with open(log_path) as f:
            limited = any(d.get("error") == "rate_limit" and (session_id is None or d.get("session_id") == session_id)
                          for d in (json.loads(line) for line in f if line.strip()))
    except (OSError, ValueError):
        return False
    if not limited:
        return False
    from core.accounts import plan_meter
    return (plan_meter("claude", plan_dir) or 0) >= 100


def session_started(stream_jsonl: str, session_id: str | None) -> bool:
    try:
        with open(stream_jsonl) as f:
            return bool(session_id) and any(
                session_id in line and ('"init"' in line or '"thread.started"' in line)
                for line in f)
    except OSError:
        return False


def mcp_toolless(stream_jsonl: str) -> bool:
    status, used = None, False
    codex_started = codex_completed = False
    try:
        with open(stream_jsonl) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if d.get("subtype") == "init":
                    ss = d.get("mcp_servers") or []
                    status = next((s.get("status") for s in ss
                                   if s.get("name") == "roboarcade"), None)
                    used = False
                elif d.get("type") == "thread.started":
                    codex_started, codex_completed, used = True, False, False
                elif d.get("type") == "turn.completed":
                    codex_completed = True
                elif d.get("type") == "assistant":
                    for b in (d.get("message") or {}).get("content") or []:
                        if (b.get("type") == "tool_use"
                                and str(b.get("name", "")).startswith("mcp__roboarcade")):
                            used = True
                elif d.get("type") == "item.completed":
                    item = d.get("item") or {}
                    if (item.get("type") == "mcp_tool_call" and item.get("server") == "roboarcade"
                            and not item.get("error") and item.get("status") != "failed"):
                        used = True
    except OSError:
        return False
    return ((status is not None and status != "connected" and not used)
            or (codex_started and codex_completed and not used))


def mcp_ui_unattached(shots_dir: str) -> bool:
    return Path(shots_dir, UI_UNATTACHED).is_file()


MTOK_USD = {"claude-opus-5": (5.0, 25.0, 10.0, 0.50),
            "claude-fable-5-1": (10.0, 50.0, 20.0, 0.25)}


CODEX_MTOK_USD = {
    "gpt-5.6-sol": {"input": 4.0, "cached_input": 0.40,
                    "cache_write": 5.0, "output": 20.0},
    "gpt-6-astra": {"input": 10.0, "cached_input": 1.0,
                    "cache_write": 12.5, "output": 50.0},
    "gpt-6.1-sol": {"input": 2.0, "cached_input": 0.10,
                    "cache_write": 2.5, "output": 10.0},
}


CODEX_LONG_CONTEXT = 272_000


def _est_cost(model: str | None, toks: dict) -> float | None:
    rates = MTOK_USD.get(model or "")
    if not rates:
        return None
    r_in, r_out, r_cw, r_cr = rates
    return round((toks["input_tokens"] * r_in + toks["output_tokens"] * r_out
                  + toks["cache_creation_input_tokens"] * r_cw
                  + toks["cache_read_input_tokens"] * r_cr) / 1e6, 4)


def agent_metrics(stream_jsonl: str) -> dict:
    KEYS = ("input_tokens", "output_tokens",
            "cache_creation_input_tokens", "cache_read_input_tokens")
    result = None
    result_usage = dict.fromkeys(KEYS, 0)
    usage_after_last_result: dict = {}
    api_key_source = None
    model = None
    compactions: list[dict] = []
    cost_sum, wall_sum, api_sum, turns_sum = 0.0, 0.0, 0.0, 0
    try:
        with open(stream_jsonl) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if d.get("subtype") == "init":
                    api_key_source = d.get("apiKeySource")
                    model = d.get("model")
                if d.get("subtype") == "compact_boundary":
                    compactions.append(d.get("compact_metadata") or {})
                if d.get("type") == "result":
                    result = d
                    cost_sum += float(d.get("total_cost_usd") or 0)
                    wall_sum += float(d.get("duration_ms") or 0) / 1000.0
                    api_sum += float(d.get("duration_api_ms") or 0) / 1000.0
                    turns_sum += int(d.get("num_turns") or 0)
                    for k in KEYS:
                        result_usage[k] += int((d.get("usage") or {}).get(k) or 0)
                    usage_after_last_result.clear()
                elif d.get("type") == "assistant":
                    m = d.get("message") or {}
                    if m.get("usage"):
                        usage_after_last_result[m.get("id") or f"_{len(usage_after_last_result)}"] = m["usage"]
    except OSError:
        return {}
    wall = wall_sum if result else None
    api = api_sum if result else None
    tail_toks = {k: sum(int(u.get(k) or 0) for u in usage_after_last_result.values()) for k in KEYS}
    toks = {k: result_usage[k] + tail_toks[k] for k in KEYS}
    tail_cost = _est_cost(model, tail_toks) if any(tail_toks.values()) else 0.0
    cost = None if tail_cost is None else (round(cost_sum + tail_cost, 4) or None)
    return {
        "wall_seconds": round(wall, 1) if wall is not None else None,
        "api_seconds": round(api, 1) if api is not None else None,
        "tool_seconds": round(wall - api, 1) if wall is not None and api is not None else None,
        "tokens_in": toks["input_tokens"], "tokens_out": toks["output_tokens"],
        "cache_creation": toks["cache_creation_input_tokens"],
        "cache_read": toks["cache_read_input_tokens"],
        "tokens_budget": toks["input_tokens"] + toks["output_tokens"]
                         + toks["cache_creation_input_tokens"],
        "cost_usd": cost,
        "cost_estimated": cost is not None and bool(tail_cost),
        "compactions": compactions,
        "billed_to": None if api_key_source is None
                     else ("plan" if api_key_source == "none" else "api_key"),
        "num_turns": turns_sum,
        "clean_exit": result is not None,
    }


def codex_session_file(codex_home: str, session_id: str | None) -> pathlib.Path | None:
    sessions = pathlib.Path(codex_home, "sessions")
    if not session_id or not sessions.is_dir():
        return None
    return next(sessions.rglob(f"*-{session_id}.jsonl"), None)


def codex_session_id(stream_jsonl: str) -> str | None:
    session = None
    try:
        with open(stream_jsonl) as f:
            for line in f:
                try:
                    event = json.loads(line)
                except Exception:
                    continue
                if event.get("type") == "thread.started":
                    session = event.get("thread_id")
    except OSError:
        pass
    return session


def codex_metrics(stream_jsonl: str, codex_home: str, session_id: str | None,
                  model: str, wall_seconds: float) -> dict:
    requests, compactions, context_window = [], [], None
    previous_total = None
    if session_id:
        transcript = codex_session_file(codex_home, session_id)
        if transcript:
            with transcript.open() as f:
                for line in f:
                    try:
                        event = json.loads(line)
                    except Exception:
                        continue
                    if event.get("type") == "compacted":
                        payload = event.get("payload") or {}
                        compact = {"trigger": payload.get("trigger") or "auto"}
                        for key in ("window_number", "first_window_id", "previous_window_id",
                                    "window_id", "compaction_response_id"):
                            if payload.get(key) is not None:
                                compact[key] = payload[key]
                        compactions.append(compact)
                    payload = event.get("payload") or {}
                    if event.get("type") != "event_msg" or payload.get("type") != "token_count":
                        continue
                    info = payload.get("info") or {}
                    context_window = info.get("model_context_window") or context_window
                    usage = info.get("last_token_usage") or {}
                    total = info.get("total_token_usage")
                    if total is not None and total == previous_total:
                        continue
                    previous_total = total
                    if int(usage.get("total_tokens") or 0):
                        requests.append(usage)

    clean_exit, fallback = False, None
    try:
        with open(stream_jsonl) as f:
            for line in f:
                try:
                    event = json.loads(line)
                except Exception:
                    continue
                if event.get("type") == "turn.completed":
                    clean_exit = True
                    fallback = event.get("usage") or fallback
    except OSError:
        pass
    if not requests and fallback:
        requests = [fallback]

    totals = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
    cost = 0.0
    rates = CODEX_MTOK_USD.get(model)
    for usage in requests:
        input_tokens = int(usage.get("input_tokens") or 0)
        cached = int(usage.get("cached_input_tokens") or 0)
        cache_write = int(usage.get("cache_write_input_tokens") or 0)
        uncached = max(0, input_tokens - cached - cache_write)
        output = int(usage.get("output_tokens") or 0)
        totals["input"] += uncached
        totals["output"] += output
        totals["cache_read"] += cached
        totals["cache_write"] += cache_write
        if rates:
            in_mul, out_mul = (2.0, 1.5) if input_tokens > CODEX_LONG_CONTEXT else (1.0, 1.0)
            cost += (uncached * rates["input"] * in_mul
                     + cached * rates["cached_input"] * in_mul
                     + cache_write * rates["cache_write"] * in_mul
                     + output * rates["output"] * out_mul) / 1e6
    return {
        "wall_seconds": round(wall_seconds, 1),
        "api_seconds": None,
        "tool_seconds": None,
        "tokens_in": totals["input"],
        "tokens_out": totals["output"],
        "cache_creation": totals["cache_write"],
        "cache_read": totals["cache_read"],
        "tokens_budget": totals["input"] + totals["output"] + totals["cache_write"],
        "cost_usd": round(cost, 4) if rates and requests else None,
        "cost_estimated": bool(rates and requests),
        "compactions": compactions,
        "billed_to": "plan",
        "num_turns": len(requests) or None,
        "clean_exit": clean_exit,
        "context_window": context_window,
    }
