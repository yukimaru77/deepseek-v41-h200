#!/usr/bin/env python3
"""H200 A/H serving benchmark: single-request decode lanes with full per-request telemetry.

Standard library only; runs on the H200 host against the loopback API. Extends tools/dsv41/kit_bench.py's
`Client.stream` (the historical chunk-timed metric is kept bit-for-bit: (completion_tokens - 1) /
(last_nonempty_content_chunk - first_nonempty_content_chunk)) and adds, per request:

  * finish reason, prompt / cached / completion token counts (usage), full generated text and its sha256;
  * every content-chunk arrival time (so true emitted-token timing can be recomputed);
  * /health deltas around the request: engine rounds, drafted, accepted, server decode / prefill seconds;
  * the server's own `done` line (token-id sha, server tok/s, rounds, accepted/drafted) read from the rank-0 log;
  * nvidia-smi snapshots (memory, util, clocks, power, temperature) before and after each lane.

Lanes:
  A  : the public kit_bench set A prompts (code / prose / structured), 512 tokens, 32-token warm-up, N reps.
  H  : a fixed set of realistic code tasks and Japanese / English prose, 1024 tokens, same protocol.

Never change prompts in place; add a new lane name instead so earlier result files stay comparable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.request

LANES = {
    "A": [
        ("code", "Write a complete Python implementation of a thread-safe LRU cache with per-entry TTL expiry. "
                 "Include type hints, docstrings, and a set of unittest test cases covering eviction and expiry."),
        ("prose", "Write a long, vivid essay about the history of lighthouses: how they were built, how their lamps "
                  "and lenses evolved, and the lives of the keepers who tended them through storms."),
        ("structured", "Return a JSON array of 15 fictional employees. Each object has the fields id (int), name, "
                       "department, title, salary (int), start_date (YYYY-MM-DD), manager_id (int or null) and "
                       "skills (array of strings). Output only the JSON, pretty-printed with two-space indents."),
    ],
    "H": [
        ("code_rust_ratelimit",
         "Implement in Rust a token-bucket rate limiter library for async code using tokio: a `RateLimiter` struct "
         "with per-key buckets (a HashMap behind a tokio Mutex), configurable capacity and refill rate, an async "
         "`acquire(key, n)` that waits until enough tokens are available, and a non-blocking `try_acquire`. Include "
         "doc comments, a custom error type, and a full set of `#[tokio::test]` unit tests."),
        ("code_ts_pagination",
         "Write a TypeScript React module that implements a `usePaginatedQuery<T>` hook (fetching pages from a REST "
         "endpoint with AbortController cancellation, retries with exponential backoff, and caching of previously "
         "fetched pages) and a `DataTable` component that uses it with column sorting, page-size selection and "
         "loading/error states. Include full type definitions and Jest + React Testing Library tests."),
        ("code_py_csv_sqlite",
         "Write a Python 3.11 command-line tool that imports one or more CSV files into a SQLite database: infer "
         "column types from a sample of rows, create tables if missing, upsert rows by a user-specified primary key, "
         "run inside a transaction with batching, and log progress. Use argparse, dataclasses, type hints and only "
         "the standard library. Then write pytest tests using tmp_path that cover type inference, upserts and "
         "malformed rows."),
        ("code_go_todo_api",
         "Write a Go HTTP JSON API for a todo service using only the standard library: handlers for create, list, "
         "get, update and delete, an in-memory store guarded by sync.RWMutex, request-logging and panic-recovery "
         "middleware, input validation with clear error responses, and graceful shutdown on SIGINT/SIGTERM. Then "
         "write table-driven tests using net/http/httptest."),
        ("code_cpp_spsc",
         "Implement a header-only C++20 lock-free single-producer single-consumer ring buffer class template with "
         "power-of-two capacity, cache-line padding to avoid false sharing, try_push / try_pop and bulk "
         "operations. Explain the memory ordering of every atomic operation in comments, then write a GoogleTest "
         "suite including a two-thread stress test."),
        ("prose_ja_railway",
         "明治時代から現代までの日本の鉄道の歩みについて、技術の進歩、経済への影響、人々の暮らしの変化を織り交ぜながら、"
         "読み応えのある長いエッセイを日本語で書いてください。見出しや箇条書きは使わず、段落で構成してください。"),
        ("prose_ja_report",
         "地方の中小製造業がデジタル化を進める際の課題と対策について、経営者向けの報告書を日本語で書いてください。"
         "現状の課題、具体的な施策、導入の手順、費用対効果の考え方、失敗しやすい点を、具体例を交えて詳しく説明してください。"),
        ("prose_en_story",
         "Write a long short story about a retired marine engineer who restores an old wooden sailboat with her "
         "estranged grandson over one summer on the coast of Maine. Use rich sensory detail and natural dialogue, "
         "and let their relationship change gradually."),
        ("prose_en_weather",
         "Write a detailed explanatory article for curious non-specialists about how modern weather forecasting "
         "works: observations, data assimilation, numerical models, ensembles, and the limits of predictability. "
         "Use concrete examples and analogies, in flowing paragraphs without bullet lists."),
    ],
}
LANE_TOKENS = {"A": 512, "H": 1024}
DONE_RE = re.compile(r"\[tensorfold\] done (req-\w+) prompt=(\d+) cached=(\d+) thinking=(\w+) tokens=(\d+) "
                     r"sha=(\w+) finish=(\w+) tok/s=([\d.]+) ttft=([-\d.]+)s prefill=([-\d.]+)s rounds=(\d+) "
                     r"accepted=(\d+)/(\d+)")
SMI_FIELDS = ("index,memory.used,memory.total,utilization.gpu,utilization.memory,clocks.sm,clocks.mem,power.draw,"
              "temperature.gpu,clocks_throttle_reasons.active")


def smi() -> list | str:
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={SMI_FIELDS}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
        keys = SMI_FIELDS.split(",")
        return [dict(zip(keys, [x.strip() for x in line.split(",")])) for line in out.strip().splitlines()]
    except Exception as e:  # noqa: BLE001
        return f"nvidia-smi failed: {e!r}"


class Client:
    def __init__(self, base: str, model: str):
        self.base, self.model = base.rstrip("/"), model

    def _req(self, path, body=None, timeout=3600):
        data = json.dumps(body).encode() if body is not None else None
        return urllib.request.urlopen(urllib.request.Request(self.base + path, data=data,
                                                             headers={"Content-Type": "application/json"}),
                                      timeout=timeout)

    def get(self, path, timeout=10):
        with self._req(path, None, timeout) as r:
            return json.loads(r.read())

    def stream(self, body: dict) -> dict:
        body = dict(body, model=self.model, stream=True, stream_options={"include_usage": True})
        start = time.perf_counter()
        first = last = None
        usage, pieces, chunk_times, finish, served_model = None, [], [], None, None
        with self._req("/v1/chat/completions", body) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[5:])
                served_model = chunk.get("model", served_model)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices", []):
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
                    piece = choice.get("text") or (choice.get("delta") or {}).get("content") or ""
                    if piece:
                        now = time.perf_counter()
                        first = first if first is not None else now
                        last = now
                        chunk_times.append(round(now - start, 6))
                        pieces.append(piece)
        end = time.perf_counter()
        n = int(usage["completion_tokens"]) if usage else None
        text = "".join(pieces)
        return {"ttft_s": (first - start) if first else None, "decode_s": (last - first) if first else None,
                "total_s": end - start, "tokens": n,
                "prompt_tokens": int(usage["prompt_tokens"]) if usage else None,
                "cached_tokens": ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens"),
                "decode_tps": (n - 1) / (last - first) if n and first and last > first else None,
                "chunks": len(chunk_times), "chunk_times": chunk_times, "finish_reason": finish,
                "served_model": served_model, "text": text,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest()}


def body_of(prompt: str, max_tokens: int) -> dict:
    return {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}


class LogTail:
    def __init__(self, path: str | None):
        self.path = path
        self.pos = os.path.getsize(path) if path and os.path.exists(path) else 0

    def new_done(self) -> list[dict]:
        if not self.path or not os.path.exists(self.path):
            return []
        with open(self.path, "rb") as f:
            f.seek(self.pos)
            data = f.read()
            self.pos = f.tell()
        rows = []
        for line in data.decode(errors="replace").splitlines():
            m = DONE_RE.search(line)
            if m:
                g = m.groups()
                rows.append({"req": g[0], "prompt": int(g[1]), "cached": int(g[2]), "thinking": g[3] == "True",
                             "tokens": int(g[4]), "token_sha": g[5], "finish": g[6], "server_tps": float(g[7]),
                             "server_ttft_s": float(g[8]), "server_prefill_s": float(g[9]), "rounds": int(g[10]),
                             "accepted": int(g[11]), "drafted": int(g[12])})
        return rows


def health_delta(h0: dict, h1: dict) -> dict:
    keys = ("requests_total", "completion_tokens_total", "decode_seconds_total", "prefill_seconds_total",
            "rounds_total", "drafted_total", "accepted_total", "cached_tokens_total")
    return {k: round(h1.get(k, 0) - h0.get(k, 0), 6) for k in keys}


def run_one(c: Client, tail: LogTail, prompt: str, max_tokens: int) -> dict:
    h0 = c.get("/health")
    if h0.get("requests_running"):
        raise SystemExit(f"refusing to measure: {h0['requests_running']} request(s) already running")
    r = c.stream(body_of(prompt, max_tokens))
    time.sleep(0.05)
    h1 = c.get("/health")
    d = health_delta(h0, h1)
    r["health_delta"] = d
    done = tail.new_done()
    r["server_done"] = done[-1] if done else None
    r["foreign_requests"] = d["requests_total"] - 1           # >0 means someone else used the server meanwhile
    toks = r["tokens"] or 0
    if d["rounds_total"]:
        r["tokens_per_round"] = round(toks / d["rounds_total"], 4)
    if d["drafted_total"]:
        r["accept_rate"] = round(d["accepted_total"] / d["drafted_total"], 4)
    if d["decode_seconds_total"] > 0 and toks > 1:
        r["server_decode_tps"] = round((toks - 1) / d["decode_seconds_total"], 3)
        r["server_ms_per_round"] = round(1000 * d["decode_seconds_total"] / max(1, d["rounds_total"]), 4)
    # True emitted-token rate over the chunk span: tokens after the first chunk / span, using the first chunk's
    # token count as unknown -> the kit metric already assumes 1; keep both views.
    return r


def summarize(rows: list[dict]) -> dict:
    def med(key):
        v = [x[key] for x in rows if x.get(key) is not None]
        return round(statistics.median(v), 4) if v else None

    def all_(key, nd=3):
        return [round(x[key], nd) if isinstance(x.get(key), float) else x.get(key) for x in rows]

    return {"decode_tps_median": med("decode_tps"), "decode_tps_all": all_("decode_tps"),
            "server_decode_tps_median": med("server_decode_tps"), "server_decode_tps_all": all_("server_decode_tps"),
            "tokens_per_round_all": all_("tokens_per_round", 4), "accept_rate_all": all_("accept_rate", 4),
            "server_ms_per_round_all": all_("server_ms_per_round", 4),
            "tokens_all": all_("tokens"), "finish_all": all_("finish_reason"), "cached_all": all_("cached_tokens"),
            "prompt_tokens_all": all_("prompt_tokens"), "chunks_all": all_("chunks"),
            "ttft_s_median": med("ttft_s"), "ttft_s_all": all_("ttft_s", 4),
            "identical_texts": len({x["text_sha256"] for x in rows}) == 1,
            "text_sha256": sorted({x["text_sha256"] for x in rows}),
            "token_sha": sorted({(x.get("server_done") or {}).get("token_sha") for x in rows} - {None}),
            "foreign_requests": sum(x["foreign_requests"] for x in rows)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:18891")
    p.add_argument("--model", default="DeepSeek-V4.1-Flash-1017-H200")
    p.add_argument("--lanes", default="A,H")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--only", default="", help="comma list of prompt names to run (default all)")
    p.add_argument("--tokens", type=int, default=0, help="override lane token budget")
    p.add_argument("--log", help="rank-0 server log (host path) for the done lines")
    p.add_argument("--label", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    c = Client(a.base, a.model)
    tail = LogTail(a.log)
    only = {x for x in a.only.split(",") if x}
    record = {"label": a.label, "base": a.base, "model_requested": a.model, "unix_start": time.time(),
              "argv": sys.argv, "health_start": c.get("/health"), "models": c.get("/v1/models"),
              "smi_start": smi(), "lanes": {}}
    for lane in a.lanes.split(","):
        tokens = a.tokens or LANE_TOKENS[lane]
        lane_rec = {"max_tokens": tokens, "reps": a.reps, "smi_before": smi(), "prompts": {}}
        for name, prompt in LANES[lane]:
            if only and name not in only:
                continue
            warm = run_one(c, tail, prompt, 32)
            runs = [run_one(c, tail, prompt, tokens) for _ in range(a.reps)]
            s = summarize(runs)
            lane_rec["prompts"][name] = {"summary": s, "warmup": {k: warm.get(k) for k in
                                                                  ("tokens", "cached_tokens", "ttft_s")},
                                         "runs": runs}
            print(json.dumps({"lane": lane, "prompt": name, **{k: s[k] for k in (
                "decode_tps_median", "decode_tps_all", "server_decode_tps_all", "tokens_per_round_all",
                "tokens_all", "finish_all", "cached_all", "identical_texts", "foreign_requests")}},
                ensure_ascii=False), flush=True)
            with open(a.out, "w") as f:      # checkpoint after each prompt
                json.dump(record | {"lanes": record["lanes"] | {lane: lane_rec}}, f, indent=1, ensure_ascii=False)
        lane_rec["smi_after"] = smi()
        record["lanes"][lane] = lane_rec
    record["health_end"] = c.get("/health")
    record["smi_end"] = smi()
    record["unix_end"] = time.time()
    with open(a.out, "w") as f:
        json.dump(record, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
