"""One-shot all-gather of small fp32 rank partials between the GPUs of one host over CUDA peer writes (p2p_gather.cu).

Experiment 12 (H200 TP2 decode): NCCL's ring all-gather of a decode window's [R, 5120] fp32 partial took 12-19 us a call
in the H200 service's rounds (82 a round); here one kernel writes the partial straight into the peer's device memory
over PCIe peer-to-peer and waits for the peer's flag. The same bytes as ``NCCL.all_gather``; the kernel is captured in
graphs like any other. ``Hybrid`` (tensorfold.cuda.rdma) routes the tensors it fits (fp32, up to ``max_bytes`` a rank)
here and the rest to NCCL; the lane's watchdog reaches ``abort`` through the Hybrid's ``rdma`` attribute.

TF_P2P_SLOTS (default 4) ring slots, TF_P2P_BLOCKS (default 16) blocks a gather (the chunking; both ranks the same).
"""

from __future__ import annotations

import os
import pickle
from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    return load(name="tensorfold_p2p_gather_v2", sources=[str(here / "p2p_gather.cu")], extra_cuda_cflags=["-O3"],
                verbose=False)


class P2pGather:
    """``all_gather(send, recv)`` of fp32 tensors up to ``max_bytes`` a rank; every rank calls in the same order."""

    def __init__(self, store, rank: int, world: int, *, max_bytes: int, slots: int | None = None,
                 blocks: int | None = None, prefix: str = "tf_p2p") -> None:
        ext = _ext()
        slots = int(os.environ.get("TF_P2P_SLOTS") or 4) if slots is None else int(slots)
        blocks = int(os.environ.get("TF_P2P_BLOCKS") or 16) if blocks is None else int(blocks)
        self.ext, self.rank, self.world, self.max_bytes = ext, rank, world, int(max_bytes)
        error, info = "", b""
        try:
            self.h = ext.create(rank, world, self.max_bytes, slots, blocks)
            info = pickle.dumps((ext.local_info(self.h), slots, blocks, self.max_bytes))
        except Exception as exc:           # noqa: BLE001  (published, so every rank refuses together)
            error = f"{type(exc).__name__}: {exc}"
        store.set(f"{prefix}/info/{rank}", b"E" + error.encode() if error else b"I" + info)
        peers = [store.get(f"{prefix}/info/{r}") for r in range(world)]
        bad = [f"rank {r}: {p[1:].decode(errors='replace')}" for r, p in enumerate(peers) if p[:1] != b"I"]
        if bad:
            raise RuntimeError("P2P gather unavailable (" + "; ".join(bad) + ")")
        infos = [pickle.loads(peers[r][1:]) for r in range(world)]
        try:
            shapes = {(i[1], i[2], i[3]) for i in infos}
            if len(shapes) != 1:
                raise ValueError(f"ranks disagree on the rings (slots, blocks, bytes): {sorted(shapes)}")
            ext.connect(self.h, [i[0] for i in infos])
        except Exception as exc:           # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
        store.set(f"{prefix}/up/{rank}", b"E" + error.encode() if error else b"U")
        ups = [store.get(f"{prefix}/up/{r}") for r in range(world)]
        bad = [f"rank {r}: {u[1:].decode(errors='replace')}" for r, u in enumerate(ups) if u[:1] != b"U"]
        if bad:
            raise RuntimeError("P2P gather unavailable (" + "; ".join(bad) + ")")

    def fits(self, send: torch.Tensor) -> bool:
        return send.dtype == torch.float32 and send.numel() % 4 == 0 and send.numel() * 4 <= self.max_bytes

    def all_gather(self, send: torch.Tensor, recv: torch.Tensor) -> None:
        self.ext.gather(self.h, send, recv)

    def failure(self) -> str:
        return self.ext.failure(self.h)

    def abort(self, why: str) -> None:
        """A peer is gone: every gather on this rank (captured ones too) stops waiting and returns at once."""

        self.ext.abort(self.h, why)
