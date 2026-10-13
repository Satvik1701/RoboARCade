"""The optional browser page for the agent and its headless Chromium.

Served by sim_env_server: the static page (the interface package's ui/index.html, else core/UI) under a
Content-Security-Policy confining it to this stack's localhost ports, and a headless Chromium on BASE_PORT + 3 that the
MCP server attaches to over CDP. Interfaces whose manifest.json sets "ui": false get neither; the simulator's API on
BASE_PORT + 1 and + 2 is the same for every interface.
"""
import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from websockets.sync.client import connect as websocket_connect

from core.utils import get_json, interface_dir

logger = logging.getLogger(__name__)

INSTANCE_ID = f"{os.getpid()}-{int(time.time() * 1000)}"

UI_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "UI")


def package_ui_dir() -> str | None:
    spec = os.environ.get("ROBOARCADE_INTERFACE", "")
    if not spec:
        return None
    directory = interface_dir(spec) / "ui"
    if (directory / "index.html").is_file():
        return str(directory)
    return None


def csp(base_port: int) -> str:
    origins = " ".join(
        f"{scheme}://{name}:{port}"
        for name in ("localhost", "127.0.0.1")
        for port in range(base_port, base_port + 4)
        for scheme in ("http", "ws")
    )
    return f"default-src 'self' 'unsafe-inline' 'unsafe-eval' data: blob: {origins}"


class UiRequestHandler(SimpleHTTPRequestHandler):
    csp_value = ""

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Security-Policy", self.csp_value)
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        if self.path.split("?")[0] == "/instance.json":
            body = json.dumps({"id": INSTANCE_ID}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def translate_path(self, path):
        local = super().translate_path(path)
        if os.path.exists(local):
            return local
        rel = os.path.relpath(local, self.directory)
        return os.path.join(UI_DIR, rel)


def http_server(page_dir: str | None, port: int) -> ThreadingHTTPServer:
    root = os.path.abspath(page_dir) if page_dir else UI_DIR
    handler = partial(UiRequestHandler, directory=root)
    UiRequestHandler.csp_value = csp(port)
    httpd = ThreadingHTTPServer(("localhost", port), handler)
    logger.info(f"Starting UI HTTP server on port {port} (root: {root})")
    return httpd


CHROMIUM_START_ATTEMPTS = 3
CHROMIUM_READY_TIMEOUT_S = 60.0
CHROMIUM_LOG_MAX_BYTES = 1 << 20


_CHROMIUM_PATH: str | None = None


def _userns_restricted() -> bool:
    try:
        return open("/proc/sys/kernel/apparmor_restrict_unprivileged_userns").read().strip() == "1"
    except OSError:
        return False


def _chromium_executable() -> str:
    global _CHROMIUM_PATH
    if _CHROMIUM_PATH is None:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            _CHROMIUM_PATH = p.chromium.executable_path
    return _CHROMIUM_PATH


def cdp_ready(base_port: int, timeout: float = 0.5) -> bool:
    info = get_json(f"http://localhost:{base_port + 3}/json/version", timeout=timeout)
    endpoint = info.get("webSocketDebuggerUrl") if isinstance(info, dict) else None
    if not endpoint:
        return False
    try:
        with websocket_connect(endpoint, open_timeout=timeout, close_timeout=timeout,
                               proxy=None) as websocket:
            websocket.send(json.dumps({"id": 1, "method": "Browser.getVersion"}))
            response = json.loads(websocket.recv(timeout=timeout))
    except Exception:
        return False
    return (isinstance(response, dict) and response.get("id") == 1
            and isinstance(response.get("result"), dict))


def end_episode(base_port: int) -> None:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        browser = pw.chromium.connect_over_cdp(f"http://127.0.0.1:{base_port + 3}")
        try:
            browser.contexts[0].pages[0].evaluate(
                "() => window.roboarcade.endEpisode('harness')")
        finally:
            browser.close()


class AgentBrowser:
    def __init__(self, base_port, log_path):
        self.base_port, self.log_path = base_port, log_path
        self.proc = self.profile = self.log_thread = None
        self.log_bytes, self.log_truncated, self.log_started = 0, False, False

    def _drain(self, pipe):
        marker = (f"\n[roboarcade] Chromium stderr truncated at "
                  f"{CHROMIUM_LOG_MAX_BYTES} bytes\n").encode()
        data_limit = CHROMIUM_LOG_MAX_BYTES - len(marker)
        with pipe, open(self.log_path, "ab", buffering=0) as log:
            while chunk := pipe.read(65536):
                keep = max(0, min(len(chunk), data_limit - self.log_bytes))
                if keep:
                    log.write(chunk[:keep])
                    self.log_bytes += keep
                if keep < len(chunk) and not self.log_truncated:
                    log.write(marker)
                    self.log_bytes += len(marker)
                    self.log_truncated = True

    def _launch(self):
        if not self.log_started:
            parent = os.path.dirname(self.log_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(self.log_path, "wb"):
                pass
            self.log_started = True
        self.profile = tempfile.mkdtemp(prefix="roboarcade_chrome_")
        args = [
            _chromium_executable(),
            f"--remote-debugging-port={self.base_port + 3}",
            f"--user-data-dir={self.profile}",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-background-networking",
            "--disable-extensions",
            "--disable-sync",
            "--disable-default-apps",
            "--force-device-scale-factor=1",
            "--window-size=1400,900",
            "--test-type",
            "--disable-infobars",
            "--disable-background-timer-throttling",
            "--disable-renderer-backgrounding",
            "--disable-backgrounding-occluded-windows",
            "--headless=new",
            "--enable-unsafe-swiftshader",
            *(["--no-sandbox"] if _userns_restricted() else []),
            f"http://localhost:{self.base_port}",
        ]
        self.proc = subprocess.Popen(
            args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        assert self.proc.stderr is not None
        self.log_thread = threading.Thread(
            target=self._drain, args=(self.proc.stderr,), daemon=True)
        self.log_thread.start()
        logger.info(f"[session_browser] raw Chromium pid={self.proc.pid}, CDP :{self.base_port + 3}")

    def _ready(self) -> bool:
        deadline = time.monotonic() + CHROMIUM_READY_TIMEOUT_S
        while time.monotonic() < deadline:
            if self.proc is None or self.proc.poll() is not None:
                return False
            if cdp_ready(self.base_port):
                return True
            time.sleep(0.25)
        return False

    def start(self):
        for attempt in range(1, CHROMIUM_START_ATTEMPTS + 1):
            self._launch()
            if self._ready():
                logger.info(f"[session_browser] CDP WebSocket ready on attempt {attempt}")
                return
            logger.warning(f"[session_browser] CDP WebSocket failed on attempt {attempt}/"
                           f"{CHROMIUM_START_ATTEMPTS}; replacing Chromium")
            self.stop()
        raise RuntimeError(
            f"Chromium CDP WebSocket failed after {CHROMIUM_START_ATTEMPTS} launches")

    def stop(self):
        proc = self.proc
        self.proc = None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        log_thread = self.log_thread
        self.log_thread = None
        if log_thread is not None:
            log_thread.join(timeout=2)
        if self.profile:
            shutil.rmtree(self.profile, ignore_errors=True)
            self.profile = None
