"""Capacity/quality validation, run explicitly on the H200 host; stdlib only.

Run explicitly with --base, --model, --out, and --fixture-dir; see docs/H200-BENCHMARKS.md.

Uses /metrics for total admitted requests (prefill + decode), /health.streams.decoding
for decode-only evidence, and nvidia-smi for memory. Does not infer 36 full contexts
from a configured pool: the long stage submits exactly TWO distinct near-1M prompts.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "tools/dsv41"))
from kit_bench import Client, load_key


class Recorder:
    def __init__(self, directory, args):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.phase = "startup"
        self.data = {"started_unix": time.time(), "configuration": {
            "base": args.base, "model": args.model, "streams": args.streams,
            "tokens": args.tokens, "stage": args.stage}, "rows": []}
        self.peaks = {}
        self.gpu_peaks = {}
        self.events = (directory / "events.jsonl").open("a", buffering=1)
        self.samples = (directory / "samples.jsonl").open("a", buffering=1)

    def _save(self):
        self.data["phase_peaks"] = self.peaks
        self.data["gpu_peak_used_mib"] = self.gpu_peaks
        temp = self.directory / "validation.json.tmp"
        temp.write_text(json.dumps(self.data, ensure_ascii=False, indent=2) + "\n")
        temp.replace(self.directory / "validation.json")

    def row(self, case, **fields):
        rec = {"case": case, "phase": self.phase, "unix": time.time(), **fields}
        with self.lock:
            self.data["rows"].append(rec)
            self.events.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._save()
        print(json.dumps({k: v for k, v in rec.items() if k not in ("token_ids", "text")},
                         ensure_ascii=False), flush=True)
        return rec

    def sample(self, kind, phase, **fields):
        with self.lock:
            self.samples.write(json.dumps({"unix": time.time(), "phase": phase, "kind": kind,
                                           **fields}, ensure_ascii=False) + "\n")
            if kind == "server":
                peaks = self.peaks.setdefault(phase, {})
                for name in ("total_live", "decoding", "prefilling"):
                    value = fields.get(name)
                    if isinstance(value, (int, float)):
                        peaks[name] = max(peaks.get(name, 0), value)
            if kind == "gpu":
                for gpu in fields.get("gpus", []):
                    key = str(gpu["index"])
                    self.gpu_peaks[key] = max(self.gpu_peaks.get(key, 0), gpu["used_mib"])


def monitor_server(client, rec, stop):
    # metrics.py:_requests reads decoder.live(), which includes filling. Only
    # cuda/health.py:Health.snapshot's streams.decoding is decode-only evidence.
    while not stop.is_set():
        start, phase = time.monotonic(), rec.phase
        fields = {}
        try:
            with client._req("/metrics", timeout=2) as response:
                text = response.read().decode()
            match = re.search(r"^tensorfold:requests_running\s+([0-9.]+)$", text, re.M)
            fields["total_live"] = float(match[1]) if match else None
            fields["metrics"] = text
        except Exception as exc:
            fields["metrics_error"] = repr(exc)
        try:
            health = client.get("/health", timeout=2)
            streams = health.get("streams") or {}
            fields.update(decoding=streams.get("decoding"), prefilling=streams.get("prefilling"),
                          max_streams=streams.get("max"), health=health)
        except Exception as exc:
            fields["health_error"] = repr(exc)
        rec.sample("server", phase, **fields)
        stop.wait(max(0, 0.2 - (time.monotonic() - start)))


def monitor_gpu(rec, stop):
    while not stop.is_set():
        start, phase = time.monotonic(), rec.phase
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
                                     "--format=csv,noheader,nounits"], capture_output=True,
                                    text=True, check=True, timeout=5)
            gpus = []
            for line in result.stdout.splitlines():
                index, used, total = (int(v.strip()) for v in line.split(","))
                gpus.append({"index": index, "used_mib": used, "total_mib": total})
            rec.sample("gpu", phase, gpus=gpus)
        except Exception as exc:
            rec.sample("gpu", phase, error=repr(exc))
        stop.wait(max(0, 2 - (time.monotonic() - start)))


def complete(client, rec, case, ids, tokens, **extra):
    started = time.time()
    body = {"model": client.model, "prompt": ids, "max_tokens": tokens, "temperature": 0,
            "return_token_ids": True, **extra}
    response = client.post("/v1/completions", body, timeout=7200)
    choice = response["choices"][0]
    usage = response.get("usage") or {}
    return rec.row(case, started_unix=started, elapsed_s=time.time() - started,
                   text=choice.get("text") or "", token_ids=(response.get("tensorfold") or {}).get("token_ids"),
                   finish_reason=choice.get("finish_reason"), usage=usage,
                   prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"),
                   cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
                   input_tokens=len(ids))


def burst(jobs):
    barrier = threading.Barrier(len(jobs))

    def run(job):
        barrier.wait(timeout=60)
        return job()

    with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        futures = [executor.submit(run, job) for job in jobs]
        # All workers finish before any failure is propagated.
        return [future.result() for future in futures]


def short_stage(client, rec, args):
    rec.phase = "unicode"
    expected = "春の京都で「珈琲」を飲む。ひらがな・カタカナ・漢字・髙橋・𠮷野・①②③・αβγ・≤≥≠・🍵🌸。"
    response = client.stream({"messages": [{"role": "user", "content":
        "次の一文を一字も変えずにそのまま出力してください。説明や引用符は付けないでください。\n" + expected}],
        "max_tokens": 128, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}},
        "/v1/chat/completions")
    unicode_ok = response["text"] == expected
    rec.row("unicode_exact", **response, expected=expected, passed=unicode_ok)

    # Distinct prefixes and answers expose slot/extent cross-talk that 63 copies
    # of the same prompt cannot detect. Compare every stream to its own baseline.
    secrets = [f"amber-{i:03d}-{7319 + i * 37}" for i in range(args.streams)]
    prompts = [f"独立セッションID: session-{i:03d}\nこのセッションだけの暗号語: {secret}\n"
               "最初の行に上の暗号語だけをそのまま出力してください。続けて、"
               "31 × 17 を計算し、筆算の手順を日本語で説明してください。"
               "さらに、分配法則を使った別の計算法と検算の方法を詳しく説明してください。"
               for i, secret in enumerate(secrets)]
    ids = [client.chat_ids(prompt) for prompt in prompts]
    rec.row("independent_session_inputs", prompts=prompts, expected_secrets=secrets,
            input_tokens=list(map(len, ids)))
    rec.phase = "solo_no_draft"
    solo = [complete(client, rec, f"solo_{i}", tokens, args.tokens, draft=False, ignore_eos=True)
            for i, tokens in enumerate(ids)]
    rec.phase = "concurrent_short"
    results = burst([lambda i=i: complete(client, rec, f"concurrent_{i}", ids[i], args.tokens,
                                         ignore_eos=True) for i in range(args.streams)])
    equal = [bool(reference["token_ids"]) and row["token_ids"] == reference["token_ids"]
             for reference, row in zip(solo, results)]
    secrets_ok = [reference["text"].lstrip().startswith(secret) and row["text"].lstrip().startswith(secret)
                  for reference, row, secret in zip(solo, results, secrets)]
    full = all(row["completion_tokens"] == args.tokens and len(row["token_ids"] or []) == args.tokens
               for row in solo + results)
    peaks = dict(rec.peaks.get("concurrent_short", {}))
    decode_seen = peaks.get("decoding", 0) >= args.streams
    rec.row("concurrent_exact_summary", exact_matches=sum(equal), requests=len(results),
            mismatches=[i for i, ok in enumerate(equal) if not ok], full_token_budget=full,
            independent_secret_matches=sum(secrets_ok),
            secret_mismatches=[i for i, ok in enumerate(secrets_ok) if not ok],
            total_live_peak=peaks.get("total_live"), decoding_peak=peaks.get("decoding"),
            observed_all_decoding=decode_seen,
            passed=unicode_ok and all(equal) and all(secrets_ok) and full and decode_seen)
    return unicode_ok and all(equal) and all(secrets_ok) and full and decode_seen


def long_stage(client, rec, args):
    rec.phase = "long_prepare"
    reference = json.loads(args.reference.read_text())
    prepared = next(row for row in reference["rows"]
                    if row.get("case") == "prompt_prepared" and row.get("target_length") == 1047000)
    text = gzip.decompress(args.prompt.read_bytes()).decode("utf-8")
    phrases, facts = prepared["phrases"], prepared["facts"]
    digest = hashlib.sha256(text.encode()).hexdigest()
    if digest != prepared["sha256"]:
        raise ValueError("Long prompt checksum differs from the 07 reference")
    prefixes = [f"独立資料{label}\n" + text for label in ("A", "B")]
    query = "\n\nWhat are the secret passphrases for vault A, vault B, and vault C mentioned in the notes? List all three passphrases only."
    ids = [client.chat_ids(prefix + query) for prefix in prefixes]
    total = sum(map(len, ids))
    rec.row("long_inputs", input_tokens=list(map(len, ids)), total_input_tokens=total,
            distinct_prefixes=["独立資料A", "独立資料B"], sha256=digest, phrases=phrases,
            submitted_long_requests=2, exceeds_two_million_tokens=total > 2_000_000,
            note="Tests two near-1M inputs; does not test 36 simultaneous full contexts.")
    if any(len(tokens) + 256 > 1048576 for tokens in ids):
        raise ValueError("Long prompt plus reply exceeds the per-request context limit")
    rec.phase = "concurrent_long"
    replies = burst([lambda i=i: complete(client, rec, f"long_{i}", ids[i], 128) for i in range(2)])
    found = [[phrase in reply["text"] for phrase in phrases] for reply in replies]
    accounted = all(reply["prompt_tokens"] == len(tokens) for reply, tokens in zip(replies, ids))
    rec.row("long_recall_summary", found=found, prompt_usage_matches_input=accounted,
            actual_prompt_tokens=sum(reply["prompt_tokens"] or 0 for reply in replies),
            passed=all(all(values) for values in found) and accounted and total > 2_000_000)

    rec.phase = "long_cached_followup"
    cached = [complete(client, rec, f"long_cached_{i}", ids[i], 128) for i in range(2)]
    cached_ok = all(row["cached_tokens"] > 0 and all(p in row["text"] for p in phrases) for row in cached)
    rec.row("cache_reuse_summary", cached_tokens=[row["cached_tokens"] for row in cached], passed=cached_ok)

    rec.phase = "japanese_summary"
    question = ("\n\nメモ中の3件の計画について、案件名・担当者・予算・納期を日本語で整理してください。"
                "その後、予算の合計と納期の早い順を述べてください。記載のない情報は補わないでください。")
    jp_ids = client.chat_ids(prefixes[0] + question)
    if len(jp_ids) + 384 > 1048576:
        raise ValueError("Japanese summary exceeds the per-request context limit")
    jp = complete(client, rec, "japanese_summary", jp_ids, 384)
    values = [v for fact in facts for v in re.findall(r"(?:案件名|担当者|予算|納期)は([^。]+)", fact)]
    normalized = jp["text"].replace(",", "").replace("，", "")
    checks = {value: value in normalized for value in values}
    checks["budget_total_1100"] = "1100" in normalized or "1,100" in jp["text"]
    checks["no_replacement_character"] = "\ufffd" not in normalized
    checks["cache_reused"] = jp["cached_tokens"] > 0
    jp_ok = all(checks.values())
    rec.row("japanese_summary_checks", checks=checks, passed=jp_ok,
            note="Checks factual fields and total; full prose and deadline ordering remain reviewable in text.")
    return all(all(values) for values in found) and accounted and total > 2_000_000 and cached_ok and jp_ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", default="DeepSeek-V4.1-Flash-1017-H200")
    parser.add_argument("--key-file")
    parser.add_argument("--streams", type=int, default=63)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--stage", choices=("all", "short", "long"), default="all")
    parser.add_argument("--out", type=Path, required=True, help="Local result directory; contains synthetic prompts and replies")
    parser.add_argument("--fixture-dir", type=Path, help="Output directory from prepare_validation.py; required for all/long")
    args = parser.parse_args()
    if args.stage != "short" and args.fixture_dir is None:
        parser.error("--fixture-dir is required for the all and long stages")
    if args.fixture_dir is not None:
        args.prompt = args.fixture_dir / "prompt-1047000.txt.gz"
        args.reference = args.fixture_dir / "validation-fixture.json"
    if args.tokens < 256 or args.streams < 1:
        parser.error("--tokens must be >=256 and --streams must be positive")
    rec = Recorder(args.out, args)
    client = Client(args.base, args.model, load_key(args.key_file))
    stop = threading.Event()
    monitors = [threading.Thread(target=monitor_server, args=(client, rec, stop), daemon=True),
                threading.Thread(target=monitor_gpu, args=(rec, stop), daemon=True)]
    for thread in monitors:
        thread.start()
    passed = False
    try:
        rec.row("models", response=client.get("/v1/models"))
        checks = []
        if args.stage in ("all", "short"):
            checks.append(short_stage(client, rec, args))
        if args.stage in ("all", "long"):
            checks.append(long_stage(client, rec, args))
        passed = all(checks)
    except Exception as exc:
        rec.row("error", error=repr(exc))
        raise
    finally:
        stop.set()
        for thread in monitors:
            thread.join(timeout=10)
        rec.phase = "completed"
        rec.row("completed", passed=passed)
        rec.events.close()
        rec.samples.close()
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
