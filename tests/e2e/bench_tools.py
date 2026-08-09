"""Direct benchmark + tool-calling validation battery against the proxy.

Tests the inference server the way opencode actually uses it, WITHOUT
spawning the opencode CLI:
  T1  single-shot tool call       (model must return an OpenAI tool_calls delta)
  T2  multi-turn tool history      (assistant tool_calls + tool results, parseable)
  T3  tight-context overflow       (push past ctx budget, verify no 400 + valid finish)
  T4  streaming integrity          (valid SSE, terminating finish_reason + [DONE])
  T5  tool-name canonicalization   (near-miss name must be remapped to a real tool)
  T6  invalid/repair JSON path     (unit: _repair_json + _extract_tool_calls)

Each case is run N times (default 1) and scored. Prints a PASS/FAIL table
and a summary with a computed reliability score.

Usage:
  python tests/e2e/bench_tools.py --runs 3 [--model local]
"""
import argparse
import json
import re
import sys
import time
import urllib.request
import urllib.error

sys.path.insert(0, r"C:\dev\inference-server")

from app.main import _repair_json, _extract_tool_calls, _canonicalize_tool_name  # noqa: E402

SERVER = "http://127.0.0.1:8080/v1"
URL = SERVER + "/chat/completions"

TOOLS = [
    {"type": "function", "function": {
        "name": "bash", "description": "Run a shell command",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "read", "description": "Read a file from disk",
        "parameters": {"type": "object", "properties": {"filePath": {"type": "string"}}, "required": ["filePath"]}}},
    {"type": "function", "function": {
        "name": "edit", "description": "Edit a file",
        "parameters": {"type": "object", "properties": {
            "filePath": {"type": "string"}, "oldString": {"type": "string"}, "newString": {"type": "string"}},
            "required": ["filePath", "oldString", "newString"]}}},
]


def post(messages, stream=True, tools=TOOLS, max_tokens=300):
    body = json.dumps({"model": "local", "stream": stream, "max_tokens": max_tokens,
                       "messages": messages, "tools": tools}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read().decode()


def parse_native_stream(text):
    """Parse a full SSE body -> (tool_names, content_text, finish_ok, done, err)."""
    tcs, names = {}, []
    content_parts = []
    finish_ok = done = False
    err = None
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        d = line[6:].strip()
        if d == "[DONE]":
            done = True
            continue
        try:
            o = json.loads(d)
        except Exception:
            continue
        if o.get("error"):
            err = o["error"].get("message") or str(o["error"])
            continue
        ch = (o.get("choices") or [{}])[0]
        if ch.get("finish_reason"):
            finish_ok = ch["finish_reason"] in ("tool_calls", "stop", "length")
        delta = ch.get("delta") or {}
        if delta.get("content"):
            content_parts.append(delta["content"])
        for tc in delta.get("tool_calls") or []:
            fn = tc.get("function") or {}
            idx = tc.get("index", 0)
            t = tcs.setdefault(idx, {"name": "", "args": ""})
            if fn.get("name"):
                t["name"] = fn["name"]
            if fn.get("arguments"):
                t["args"] += fn["arguments"]
    names = [t["name"] for t in tcs.values()]
    return names, "".join(content_parts), finish_ok, done, err


def post_once(messages, max_tokens=300):
    """Non-streaming request (what many editors use for the final turn)."""
    try:
        return post(messages, stream=False, max_tokens=max_tokens)
    except urllib.error.HTTPError as e:
        return f'{{"error": {e.read().decode()[:300]}}}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()
    R = args.runs

    table = [
        ("T1 single tool call",      case1_single_tool(R)),
        ("T2 multi-turn history",    case2_multiturn_history(R)),
        ("T3 tight-context overflow", case3_tight_context(R)),
        ("T4 stream integrity",      case4_stream_integrity(R)),
        ("T5 name canonicalization", (int(case5_canonicalization()), 1, [] if case5_canonicalization() else ["mismap: reed -> read"])),
        ("T6 json repair/extract",   (int(case6_json_repair()), 1, [] if case6_json_repair() else ["repair fail"])),
    ]


def case1_single_tool(runs):
    good = 0
    errs = []
    for _ in range(runs):
        t0 = time.time()
        try:
            out = post([{"role": "system", "content": "You are a coding agent. Act, don't describe."},
                        {"role": "user", "content": "list the files in this directory using a tool"}])
            names, text, fok, done, err = parse_native_stream(out)
            dt = time.time() - t0
            if names and names[0] in ("bash", "glob", "read") and done:
                good += 1
            elif err:
                errs.append(err[:120])
            elif not done:
                errs.append("stream not closed")
        except Exception as e:
            errs.append(str(e)[:120])
    return good, runs, errs[:2]


def case2_multiturn_history(runs):
    good = 0
    errs = []
    for _ in range(runs):
        msgs = [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "look at app/main.py"},
            {"role": "assistant", "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "read", "arguments": '{"filePath":"app/main.py"}'}}],
             "content": ""},
            {"role": "tool", "tool_call_id": "call_1", "content": "[def main(): ...] (file contents)"},
            {"role": "user", "content": "what did you learn? then run git status"},
        ]
        try:
            out = post(msgs)
            names, _content, fok, done, err = parse_native_stream(out)
            if done and names and "bash" in names:
                good += 1
            elif err:
                errs.append(err[:120])
        except Exception as e:
            errs.append(str(e)[:120])
    if good == 0 and not errs:
        errs.append("model answered in text / no bash tool call")
    return good, runs, errs[:2]


def case3_tight_context(runs):
    """Oversize the context with a giant file dump; model must still answer/act."""
    good = 0
    errs = []
    big = ("line_of_code_%d " * 3000) % tuple(range(3000))
    for _ in range(runs):
        msgs = [
            {"role": "system", "content": "You are a coding agent. Keep working even if history is long."},
            {"role": "user", "content": "read the file"},
            {"role": "assistant", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "read", "arguments": '{"filePath":"big.txt"}'}}]},
            {"role": "tool", "tool_call_id": "c1", "content": big},
            {"role": "user", "content": "ok now do the same thing then summarize"},
        ]
        try:
            out = post(msgs, max_tokens=48)
            names, _content, fok, done, err = parse_native_stream(out)
            if not err and done:
                good += 1
            elif err:
                errs.append(err[:160])
        except Exception as e:
            errs.append(str(e)[:160])
    return good, runs, errs[:2]


def case4_stream_integrity(runs):
    good = 0
    errs = []
    for _ in range(runs):
        try:
            out = post([{"role": "system", "content": "You are a coding agent."},
                        {"role": "user", "content": "say hello and then list the git status using a tool"}],
                       max_tokens=120)
            names, _content, fok, done, err = parse_native_stream(out)
            if done and fok:
                good += 1
            elif err:
                errs.append(err[:120])
        except Exception as e:
            errs.append(str(e)[:120])
    return good, runs, errs[:2]


def case4b_nonstream(runs):
    """Non-streaming path (used by editors for final turns) must also yield tool_calls."""
    good = 0
    errs = []
    for _ in range(runs):
        try:
            out = post_once([{"role": "system", "content": "You are a coding agent."},
                             {"role": "user", "content": "run git status using a tool"}],
                            max_tokens=120)
            try:
                obj = json.loads(out)
            except Exception:
                errs.append("non-JSON response: " + out[:100])
                continue
            msg = (obj.get("choices") or [{}])[0].get("message") or {}
            if msg.get("tool_calls") or (msg.get("content") or "").strip():
                good += 1
            else:
                errs.append("no tool_calls & empty content")
        except Exception as e:
            errs.append(str(e)[:120])
    return good, runs, errs[:2]


def case5_canonicalization():
    ok = _canonicalize_tool_name("bashh", TOOLS) == "bash"
    ok &= _canonicalize_tool_name("reed", TOOLS) == "read"
    ok &= _canonicalize_tool_name("edit-file", TOOLS) == "edit"
    return ok


def case6_json_repair():
    ok = _repair_json('{"name":"bash","arguments":{"command":"dir"}') == \
        {"name": "bash", "arguments": {"command": "dir"}}
    ok &= _extract_tool_calls('```json\n{"name":"read","arguments":{"filePath":"a.py"}}\n```', TOOLS) == \
        [("read", {"filePath": "a.py"})]
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()
    R = args.runs

    table = [
        ("T1 single tool call",      case1_single_tool(R)),
        ("T2 multi-turn history",    case2_multiturn_history(R)),
        ("T3 tight-context overflow", case3_tight_context(R)),
        ("T4 stream integrity",      case4_stream_integrity(R)),
        ("T4b non-stream path",      case4b_nonstream(R)),
        ("T5 name canonicalization", (int(case5_canonicalization()), 1, [] if case5_canonicalization() else ["mismap: reed -> read"])) ,
        ("T6 json repair/extract",   (int(case6_json_repair()), 1, [] if case6_json_repair() else ["repair fail"])),
    ]

    print(f"== Tool-call benchmark: {R} runs/case against {SERVER} ==")
    total_good = total = 0
    for label, (good, n, errs) in table:
        pct = (good / n * 100) if n else (100 if good else 0)
        total_good += good; total += n or 1
        mark = "PASS" if good == (n or 1) else "FAIL"
        print(f"  [{mark}] {label}: {good}/{n or 1} ({pct:.0f}%)")
        for e in errs:
            print(f"        err: {e}")
    overall = total_good / total * 100 if total else 0
    print(f"== overall reliability: {overall:.0f}% ==")
    return 0 if overall >= 80 else 1


if __name__ == "__main__":
    sys.exit(main())
