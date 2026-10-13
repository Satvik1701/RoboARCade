"""Linux parent-death protection for opt-in shared renderer subprocesses."""
import ctypes
import os
import signal

PR_SET_PDEATHSIG = 1


def follow_parent(expected_pid):
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "Cannot protect renderer child lifetime")
    if os.getppid() != expected_pid:
        raise RuntimeError("Renderer owner exited during child startup")
