"""End-to-End functional test: opencode (headless) -> inference-server.

Proves the full loop works:
  1. opencode sends a real code-fix task to our local inference server.
  2. Server passes it to the model; the model uses tools (read/edit/bash).
  3. The model fixes src/calculator.py; opencode executes the edits.
  4. Reference tests confirm the fix is correct.
  5. We pull the server's own /api/log to prove requests+responses flowed.

Usage:
    python tests/e2e/run_e2e.py [--model opencode-local/qwen3.5-4b-super-coder] [--timeout 900]

Exit code 0 = PASS, 1 = FAIL.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
SAMPLE = os.path.join(ROOT, "sample_bug")
WORK = os.path.join(ROOT, "work")
SERVER = "http://127.0.0.1:8080"
DEFAULT_MODEL = "opencode-local/qwen3.5-4b-super-coder"
TASK = (
    "The unit tests in this project are failing. "
    "Read src/calculator.py, find the logic bugs in add, multiply, and is_even, "
    "fix them with the edit tool (do not change the test file), then run "
    "the tests and confirm they pass."
)


def http_get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read().decode())


def server_log(n=50):
    try:
        return http_get(f"{SERVER}/api/log?n={n}").get("log", [])
    except Exception:
        return []


def reset_work():
    if os.path.isdir(WORK):
        shutil.rmtree(WORK)
    shutil.copytree(SAMPLE, WORK)


def tests_pass(cwd):
    r = subprocess.run([sys.executable, os.path.join("tests", "test_calculator.py")],
                       cwd=cwd, capture_output=True, text=True, timeout=120)
    return r.returncode == 0


def health():
    try:
        return http_get(f"{SERVER}/health").get("status")
    except Exception:
        return None


def find_opencode():
    exe = shutil.which("opencode")
    if exe and not exe.lower().endswith(".ps1"):
        return exe
    # npm shims on Windows: prefer the .cmd for subprocess (no shells involved)
    for cand in ("opencode.cmd", "opencode.exe", "opencode"):
        c = shutil.which(cand)
        if c and not c.lower().endswith(".ps1"):
            return c
    return exe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    print(f"== E2E test: model={args.model} server={SERVER} ==")

    if health() != "ok":
        print("FAIL: inference server not healthy (run start.py / api/engines/load first)")
        return 1
    print("1) server healthy")

    reset_work()
    if tests_pass(WORK):
        print("FAIL: baseline tests already pass - sample bugs missing")
        return 1
    print("2) baseline tests FAIL as expected")

    print("3) running headless opencode against local server...")
    oc = find_opencode()
    if not oc:
        print("FAIL: opencode CLI not found on PATH")
        return 1
    print(f"   using opencode at: {oc}")
    cmd = [oc, "run", "--auto", "--model", args.model, TASK]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=WORK, capture_output=True, text=True,
                              timeout=args.timeout,
                              env=dict(os.environ, OPENCODE_TELEMETRY="off"))
    except subprocess.TimeoutExpired:
        print(f"   FAIL: opencode timed out after {args.timeout}s")
        return 1
    dur = time.time() - t0
    output = (proc.stdout or "") + (proc.stderr or "")
    print(f"   opencode exited {proc.returncode} in {dur:.0f}s ({len(output)} chars output)")

    fixed = tests_pass(WORK)
    print(f"3) tests {'PASS' if fixed else 'FAIL'} after opencode's edits in {WORK}")

    log = server_log()
    errors = [e for e in log if isinstance(e, str) and ("error" in e.lower() or "500" in e)]
    tools = [e for e in log if isinstance(e, str) and ("tool_calls" in e or "call_" in e.lower())]
    print(f"4) server log: {len(log)} recent entries, {len(errors)} error(s), {len(tools)} tool-call entry(ies)")

    if errors:
        print("   server errors: ")
        for e in errors[:5]:
            print("   -", str(e)[:160])

    print("== RESULT:", "PASS" if fixed else "FAIL", "==")
    return 0 if fixed else 1


if __name__ == "__main__":
    sys.exit(main())
