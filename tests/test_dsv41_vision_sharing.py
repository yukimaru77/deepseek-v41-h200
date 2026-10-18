"""Image row exchange preserves exact values without retaining world-sized GPU buffers."""

from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")

from tensorfold.families.deepseek_v41.cuda import engine


@pytest.fixture
def allocations(monkeypatch):
    created = []

    def factory(name):
        def allocate(shape, *, dtype, device):
            assert device == "cuda"
            result = getattr(torch, name)(shape, dtype=dtype, device="cpu")
            created.append((name, tuple(result.shape), result.numel()))
            return result
        return allocate

    monkeypatch.setattr(engine, "torch", NS(empty=factory("empty"), zeros=factory("zeros"),
                                            bfloat16=torch.bfloat16))
    return created


@pytest.mark.parametrize("world,rank", [(2, 0), (2, 1), (3, 2)])
@pytest.mark.parametrize("n", [1, 2048, 4099])
def test_chunked_exchange_returns_rank_zero_rows_and_bounded_storage(allocations, world, rank, n):
    dim = 8
    expected = torch.arange(n * dim, dtype=torch.float32).reshape(n, dim).to(torch.bfloat16)
    cursor, calls = 0, []

    def gather(send, recv):
        nonlocal cursor
        size = send.numel()
        assert recv.numel() == world * size
        if rank == 0:
            assert torch.equal(send, expected.flatten()[cursor:cursor + size])
        else:
            assert not send.any()
        recv.fill_(-123)  # remote ranks' values must never leak into the result
        recv[:size].copy_(expected.flatten()[cursor:cursor + size])
        calls.append(size)
        cursor += size

    state = NS(w=NS(cfg=NS(dim=dim)), rank=rank, world=world, nccl=NS(all_gather=gather))
    result = engine.DsEngine._share_rows(state, expected if rank == 0 else None, n)
    assert torch.equal(result, expected)
    assert cursor == n * dim and max(calls) <= 2048 * dim
    assert result.untyped_storage().nbytes() == n * dim * 2
    if rank == 0:
        assert result is expected  # no second full-sized output on the leader
        assert sum(size for _, _, size in allocations) == world * min(n, 2048) * dim
    else:
        assert sum(size for _, _, size in allocations) == n * dim + (world + 1) * min(n, 2048) * dim


def test_single_rank_sharing_reuses_rows_without_allocations(allocations):
    rows = torch.arange(12).reshape(3, 4).to(torch.bfloat16)
    assert engine.DsEngine._share_rows(NS(w=NS(cfg=NS(dim=4)), world=1), rows, 3) is rows
    assert allocations == []


@pytest.mark.parametrize("cut", [0, 11, 13, 22, 30])
def test_vision_row_assembly_keeps_order_and_skips_cached_spans(allocations, cut):
    dim = 8
    spans = [(10, NS(tokens=3, value=1)), (20, NS(tokens=4, value=2)), (30, NS(tokens=2, value=3))]
    called = []

    def encode(pic):
        called.append(pic.value)
        return (torch.arange(pic.tokens * dim).reshape(pic.tokens, dim) + pic.value * 100).to(torch.bfloat16)

    expected = torch.cat([encode(pic)[max(0, cut - start):] for start, pic in spans if start + pic.tokens > cut])
    called.clear()
    state = NS(w=NS(cfg=NS(dim=dim)), tower=NS(span_rows=encode))
    result = engine.DsEngine._vision_rows(state, spans, cut)
    assert torch.equal(result, expected)
    assert called == [pic.value for start, pic in spans if start + pic.tokens > cut]
    assert allocations == [("empty", tuple(expected.shape), expected.numel())]
