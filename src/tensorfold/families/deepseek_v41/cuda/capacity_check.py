"""Optional, pre-service qualification of a large shared KV pool on every TP rank."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch


@torch.no_grad()
def check_capacity(e) -> None:
    mu, m = e.multi, e.model
    total, limit, slots = mu.extents.total, e.limit, len(mu.slots)
    report = {"rank": e.rank, "context": limit, "pool_tokens": total, "parallel": slots}

    def save(stage):
        report["stage"] = stage
        directory = os.environ.get("TF_DS_CAPACITY_REPORT")
        if directory:
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            (path / f"startup-rank{e.rank}.json").write_text(json.dumps(report, indent=2) + "\n")

    def memory():
        torch.cuda.synchronize()
        free, capacity = torch.cuda.mem_get_info()
        return {"allocated": torch.cuda.memory_allocated(), "reserved": torch.cuda.memory_reserved(),
                "device_free": free, "device_total": capacity,
                "peak_allocated": torch.cuda.max_memory_allocated()}

    # Allocate the largest decode workspace and access the far end of the pool. Each row has its own slot.
    long_size = mu.extents.size(limit + mu.max_rows + 12)
    far = total - long_size
    short = [1000] * 1024
    long = [1000] * limit
    windows = [(0, far, long_size, limit - 1, [1000], long)]
    windows += [(i, (i - 1) * 2048, 2048, 1023, [1000], short) for i in range(1, slots)]
    assert (slots - 1) * 2048 <= far
    mu.runner.forward(windows, replay=False)
    report["maximum_decode_shape"] = [slots, mu.runner.bucket(limit)]
    report["after_maximum_graph"] = memory()
    save("maximum_graph")
    del windows, long

    if os.environ.get("TF_DS_CAPACITY_GRAPH_STRESS", "0") == "1":
        # Fill the bounded cache with the largest batches / context buckets and force eviction.
        # This accounts for executable-graph driver memory, which torch.memory_reserved excludes.
        assert mu.runner.graph_limit == 64 and slots == 63
        for bucket in (limit // 2, limit):
            host = [1000] * bucket
            size = mu.extents.size(bucket + mu.max_rows + 12)
            for rows in range(32, 64):
                windows = [(0, total - size, size, bucket - 1, [1000], host)]
                windows += [(i, (i - 1) * 2048, 2048, 1023, [1000], short) for i in range(1, rows)]
                mu.runner.forward(windows, replay=False)
        report["after_full_graph_cache"] = memory()
        report["resident_round_graphs"] = len(mu.runner.graphs)
        report["graph_evictions"] = mu.runner.evictions
        assert len(mu.runner.graphs) == 64 and mu.runner.evictions > 0
        save("full_graph_cache")
        del windows, host

    # A real prompt, copied to distinct extents: one row alone must match every row in the full batch,
    # including a row whose packed KV address is beyond 8 GiB from the plane's start.
    size, n = 4096, 1024
    ids = [1000 + (i * 7919) % 60000 for i in range(n)]
    sc = m.pool_view(mu.pool, 0, 0, size)
    previous_quiet, e.quiet = e.quiet, True
    try:
        e.prefill(sc, mu.slots[0].dc, ids)
    finally:
        e.quiet = previous_quiet
    snapshot = mu._snapshot(0)
    report["snapshot_device"] = snapshot[0].device.type
    report["snapshot_pinned"] = snapshot[0].is_pinned()
    saved = tuple(x.clone() if x is not None else None for x in snapshot)
    rs = mu.pool.ring_size
    # Replay snapshots intentionally omit decoder rings; a direct decode comparison needs those too.
    decoder_rings = {i: x[:rs].clone() for i, x in enumerate(mu.pool.ring)
                     if i >= snapshot[0].shape[0]}
    for x in mu.pool.ring:
        x[:rs].zero_()
    for pair in mu.pool.comp_raw.values():
        for x in pair:
            x[:x.shape[0] // slots].zero_()
    mu._restore(0, snapshot)
    restored = mu._snapshot(0)
    assert all(a is None or torch.equal(a, b) for a, b in zip(saved, restored)), "snapshot roundtrip changed bits"
    del saved, restored
    for i, x in decoder_rings.items():
        mu.pool.ring[i][:rs].copy_(x)
    token, host = 1234, ids + [1234]
    solo, _ = mu.runner.forward([(0, 0, size, n, [token], host)])
    solo = solo[0].clone()
    windows = []
    for slot in range(slots):
        base = total - size if slot == slots - 1 else slot * size
        mu._copy_rows(0, base, n)
        mu._restore(slot, snapshot)
        for i, x in decoder_rings.items():
            mu.pool.ring[i][slot * rs:(slot + 1) * rs].copy_(x)
        windows.append((slot, base, size, n, [token], host))
    batch, _ = mu.runner.forward(windows)
    report["batch_finite"] = bool(torch.isfinite(batch).all())
    report["batch_exact_rows"] = [bool(torch.equal(solo, row)) for row in batch]
    report["far_extent_base"] = total - size
    report["snapshot_roundtrip_exact"] = True
    report["rope_rows"] = m._cs(2, mu._table_view)[0].shape[0]
    assert report["rope_rows"] == m.rope.rows(m.rope_cap), "RoPE grew with shared pool capacity"
    assert report["batch_finite"] and all(report["batch_exact_rows"]), "concurrent or far-offset logits differ"
    save("exactness")
    del solo, snapshot, decoder_rings, windows
    # Future graph input buffers (~261 MiB over all 693 shapes), a maximal overlapping KV move
    # (256 MiB), and temporary work need room in addition to resident weights/pool/graphs.
    reserve = torch.empty((1 << 30,), dtype=torch.uint8, device="cuda")
    reserve.zero_()
    report["with_1gib_extra_workspace"] = memory()
    del reserve
    torch.cuda.empty_cache()
    report["after_checks"] = memory()
    report["passed"] = True
    save("completed")
    print(f"[tensorfold] capacity check rank {e.rank}: pool {total}, slots {slots}, "
          f"full batch / far address / CPU snapshot exact; free "
          f"{report['after_checks']['device_free'] / 2**30:.2f} GiB", flush=True)
