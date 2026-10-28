"""Socket bridges that let the sandboxed agent reach its simulator and the model API.

Unix-socket bridges for the bwrap box (--unshare-net: sockets are the only way across).
  host PROXY_PORT PORT=SOCK...        outside the box: each SOCK -> 127.0.0.1:PORT; the PROXY_PORT
                                      one is the HTTP CONNECT proxy instead (names in ALLOW only)
  box  PORT=SOCK... -- CMD...         box:  each 127.0.0.1:PORT -> SOCK, bound before CMD runs
"""
import os
import signal
import socket
import subprocess
import sys
import threading
import time

ALLOW = set(filter(None, os.environ["ROBOARCADE_EGRESS_HOSTS"].split(",")))


def splice(a: socket.socket, b: socket.socket):
    a.settimeout(None)
    b.settimeout(None)

    def pump(src, dst):
        try:
            while data := src.recv(65536):
                dst.sendall(data)
        except OSError:
            pass
        finally:
            for s in (src, dst):
                try:
                    s.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
    t = threading.Thread(target=pump, args=(b, a), daemon=True)
    t.start()
    pump(a, b)
    t.join()
    a.close()
    b.close()


def serve(listener, handle):
    while True:
        conn, _ = listener.accept()
        threading.Thread(target=handle, args=(conn,), daemon=True).start()


def unix_listener(path: str) -> socket.socket:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    s = socket.socket(socket.AF_UNIX)
    s.bind(path)
    s.listen(64)
    return s


def proxy(conn):
    req = b""
    while b"\r\n\r\n" not in req:
        chunk = conn.recv(4096)
        if not chunk:
            return conn.close()
        req += chunk
    line = req.split(b"\r\n", 1)[0].decode(errors="replace")
    parts = line.split()
    if len(parts) != 3 or parts[0] != "CONNECT" or ":" not in parts[1]:
        conn.sendall(b"HTTP/1.1 400 Bad Request\r\n\r\n")
        return conn.close()
    host, port = parts[1].rsplit(":", 1)
    if host.lower() not in ALLOW or port != "443":
        conn.sendall(b"HTTP/1.1 403 Forbidden (egress allowlist)\r\n\r\n")
        return conn.close()
    try:
        up = socket.create_connection((host, int(port)), timeout=15)
    except OSError:
        conn.sendall(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
        return conn.close()
    conn.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
    splice(conn, up)


def forward(conn, dial):
    try:
        up = dial()
    except OSError:
        return conn.close()
    splice(conn, up)


def tcp_listener(port: int) -> socket.socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", port))
    s.listen(64)
    return s


def unix_dial(path):
    s = socket.socket(socket.AF_UNIX)
    s.connect(path)
    return s


def pairs(args):
    return [(int(p), sock) for p, sock in (a.split("=", 1) for a in args)]


def wait_for_host_sockets(bridges, timeout=10):
    """The host starts concurrently; do not run the agent before its bridges bind."""
    deadline = time.monotonic() + timeout
    while not all(os.path.exists(path) for _, path in bridges):
        if time.monotonic() >= deadline:
            raise TimeoutError("Host sandbox bridges did not become ready")
        time.sleep(0.01)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "host":
        proxy_port = int(sys.argv[2])
        for port, sock in pairs(sys.argv[3:]):
            ln = unix_listener(sock)
            handle = proxy if port == proxy_port else \
                lambda conn, port=port: forward(conn, lambda: socket.create_connection(
                    ("127.0.0.1", port), timeout=10))
            threading.Thread(target=serve, args=(ln, handle), daemon=True).start()
        threading.Event().wait()
    elif mode == "box":
        split = sys.argv.index("--")
        bridges = pairs(sys.argv[2:split])
        wait_for_host_sockets(bridges)
        for port, sock in bridges:
            ln = tcp_listener(port)
            threading.Thread(target=serve, args=(ln, lambda conn, sock=sock: forward(
                conn, lambda: unix_dial(sock))), daemon=True).start()
        child = subprocess.Popen(sys.argv[split + 1:])
        signal.signal(signal.SIGTERM, lambda *_: child.terminate())
        rc = child.wait()
        sys.exit(rc if rc >= 0 else 128 - rc)
    else:
        sys.exit(f"unknown mode {mode}")
