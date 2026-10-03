"""
A/B watchdog: waits for the server on the port (default 127.0.0.1:8080) to be closed,
then runs the A/B benchmark (ab_runner.py) and reports.

Launch it as a long-lived background process. It polls the port every 5 s; once the
port stops accepting connections for 4 consecutive checks (~20 s), it invokes
ab_runner.py (which serves on the same port for the benchmark) and waits for it to
finish. It fires once per close window and never starts a server on the port itself;
it re-arms once that server is back. Same AB_HOST/AB_PORT environment overrides as
ab_runner.py.
"""
import os
import socket
import subprocess
import sys
import time
import datetime

AB_DIR = os.path.dirname(os.path.abspath(__file__))
WORKDIR = os.environ.get("AB_WORKDIR") or None
HOST = os.environ.get("AB_HOST", "127.0.0.1")
PORT = int(os.environ.get("AB_PORT", "8080"))
POLL_S = 5
CLOSED_CONFIRM = 4          # consecutive closed polls before triggering (~20 s; a restart
                            # can close the port for ~15 s, so keep the margin)
RUNNER_TIMEOUT_S = 3 * 3600 # 3 h safety cap on the A/B run
LOG = os.path.join(AB_DIR, "watchdog.log")
DONE = os.path.join(AB_DIR, "AB_DONE.txt")
RUNNER = os.path.join(AB_DIR, "ab_runner.py")


def log(msg):
    line = "[%s] %s" % (datetime.datetime.now().isoformat(timespec="seconds"), msg)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def port_open(host, port, timeout=2.0):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def wait_for_manual_restart():
    """The runner does not restart the server that was on the port. Wait for it to come back
    before re-arming so a failed run does not re-fire in a loop while the port stays free."""
    log("waiting for the server on :%d to be restarted before re-arming "
        "(the watchdog never starts it)" % PORT)
    deadline = time.time() + 6 * 3600
    while time.time() < deadline:
        if port_open(HOST, PORT):
            log("the server on :%d is back; re-arming" % PORT)
            return True
        time.sleep(10)
    log("the server did not return within 6h; stopping watchdog")
    return False


def main():
    if os.path.exists(DONE):
        log("AB_DONE.txt already present — A/B appears to have run; not re-triggering")
        return
    log("watchdog started; polling %s:%d every %ds" % (HOST, PORT, POLL_S))
    while True:
        closed_streak = 0
        while True:
            if port_open(HOST, PORT):
                closed_streak = 0
            else:
                closed_streak += 1
                if closed_streak >= CLOSED_CONFIRM:
                    log("port %d closed %dx in a row -> server stopped; launching A/B runner"
                        % (PORT, closed_streak))
                    break
                log("port %d closed (streak %d/%d)" % (PORT, closed_streak, CLOSED_CONFIRM))
            time.sleep(POLL_S)

        t0 = time.time()
        log("launching: %s %s" % (sys.executable, RUNNER))
        try:
            proc = subprocess.run([sys.executable, RUNNER], cwd=WORKDIR, timeout=RUNNER_TIMEOUT_S)
            log("runner exited code=%s after %.0fs" % (proc.returncode, time.time() - t0))
        except Exception as e:
            log("runner raised: %r after %.0fs" % (e, time.time() - t0))

        if os.path.exists(DONE):
            log("AB_DONE.txt present — A/B complete; stopping watchdog")
            return

        # Failure: the port stays free until whoever stopped the server restarts it.
        # Wait for that restart before re-arming to avoid re-firing into a down window.
        if not wait_for_manual_restart():
            return
        log("re-armed: waiting for the next :8080 close window")


if __name__ == "__main__":
    main()
