import os
import sys
import time

import psutil

PORT = int(os.environ.get("PORT", "8899"))


def _free_port(port):
    """If something is already listening on PORT (an old instance of this
    server), terminate it cleanly so a repeated start becomes a clean restart
    instead of a bind error. Returns True if a process was killed."""
    killed = False
    for conn in psutil.net_connections(kind="tcp"):
        if conn.laddr and conn.laddr.port == port and conn.status == "LISTEN":
            pid = conn.pid
            if not pid:
                continue
            try:
                p = psutil.Process(pid)
                # Never kill our own (parent) process.
                if p.pid == os.getpid():
                    continue
                p.terminate()
                try:
                    p.wait(timeout=5)
                except psutil.TimeoutExpired:
                    p.kill()
                killed = True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                pass
    return killed


if __name__ == "__main__":
    # Clean, idempotent takeover: a repeated start cleanly restarts the server.
    if _free_port(PORT):
        time.sleep(1)
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=PORT, reload=False)
