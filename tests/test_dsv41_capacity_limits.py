"""CPU-only request/pool limit regressions, executing the production methods without CUDA imports."""

import ast
from pathlib import Path
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import Mock


CUDA = Path(__file__).resolve().parents[1] / "src/tensorfold/families/deepseek_v41/cuda"
CONTEXT = 1 << 20
POOL = 32 * CONTEXT


class NoRoom(Exception):
    pass


def methods(filename, class_name, names, **dependencies):
    """Import actual method bodies, removing GPU decorators and postponing annotation evaluation."""
    source = CUDA / filename
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    selected = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in selected} != set(names):
        raise AssertionError(f"Missing production methods: {names}")
    for node in selected:
        node.decorator_list = []
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[]))
    namespace = dict(dependencies)
    exec(compile(module, str(source), "exec"), namespace)
    return {name: namespace[name] for name in names}


ADMISSION = methods("multi.py", "MultiDecoder", {"admit", "_need"},
                    KEEP=False, NoRoom=NoRoom, _pack=lambda sampling: sampling)
ROPE = methods("model.py", "Model", {"_cs"})["_cs"]


def decoder(pool=POOL, room=True):
    # The GPU-facing boundary is mocked; production admission and extent sizing run unchanged.
    state = SimpleNamespace(e=SimpleNamespace(limit=CONTEXT), extents=SimpleNamespace(total=pool),
                            max_rows=16, free=[0], kept={}, _room=Mock(return_value=room),
                            _admit=Mock(), sent=[])
    state._send = lambda op: state.sent.append((op[0], len(op[1]), op[2]))
    for name, method in ADMISSION.items():
        setattr(state, name, MethodType(method, state))
    return state


def request(prompt_tokens, output_tokens):
    return SimpleNamespace(prompt=range(prompt_tokens), count=output_tokens, constraint=None,
                           probabilities=None, vision=None, sampling=None, draft=True, stop_eos=True)


class CapacityLimitsTest(unittest.TestCase):
    def test_large_pool_does_not_admit_an_oversized_request(self):
        for prompt_tokens in (CONTEXT, CONTEXT + 1, 4 * CONTEXT):
            with self.subTest(prompt_tokens=prompt_tokens):
                state = decoder()
                with self.assertRaisesRegex(ValueError, "request context"):
                    state.admit(request(prompt_tokens, 1))
                state._room.assert_not_called()
                state._admit.assert_not_called()
                self.assertEqual(state.sent, [])

    def test_output_is_limited_by_remaining_request_context(self):
        for prompt_tokens, requested, expected in (
            (CONTEXT - 17, 65536, 17),
            (CONTEXT - 1, 4096, 1),
            (100, CONTEXT, CONTEXT - 100),
            (CONTEXT - 100, 7, 7),
        ):
            with self.subTest(prompt_tokens=prompt_tokens, requested=requested):
                state = decoder()
                stream = request(prompt_tokens, requested)
                state.admit(stream)
                self.assertEqual(stream.count, expected)
                self.assertLessEqual(len(stream.prompt) + stream.count, CONTEXT)
                state._room.assert_called_once_with(prompt_tokens + expected + 16 + 8 + 4)
                state._admit.assert_called_once_with(stream, 0, [], None)
                self.assertEqual(state.sent, [("admit", prompt_tokens, expected)])

    def test_pool_also_reserves_internal_rows_when_it_is_the_tighter_limit(self):
        state = decoder(pool=CONTEXT)
        stream = request(CONTEXT - 64, 128)
        state.admit(stream)
        self.assertEqual(stream.count, 36)
        self.assertEqual(state._need(stream), CONTEXT)
        state._admit.assert_called_once()

    def test_available_context_does_not_bypass_pool_exhaustion(self):
        state = decoder(room=False)
        with self.assertRaises(NoRoom):
            state.admit(request(8192, 256))
        state._admit.assert_not_called()
        self.assertEqual(state.sent, [])

    def test_rope_uses_request_capacity_for_full_pool_views(self):
        for pool in (32 * CONTEXT, 40 * CONTEXT):
            for layer, compressed in ((0, False), (1, True)):
                with self.subTest(pool=pool, layer=layer):
                    marker = object()
                    rope = SimpleNamespace(cs=Mock(return_value=marker))
                    model = SimpleNamespace(rope=rope, rope_cap=CONTEXT + 14,
                                            cfg=SimpleNamespace(compress_ratios=[0, 1]))
                    self.assertIs(ROPE(model, layer, SimpleNamespace(cap=pool)), marker)
                    rope.cs.assert_called_once_with(compressed, CONTEXT + 14)

    def test_standalone_model_without_engine_limit_keeps_cache_capacity(self):
        rope = SimpleNamespace(cs=Mock())
        model = SimpleNamespace(rope=rope, cfg=SimpleNamespace(compress_ratios=[1]))
        ROPE(model, 0, SimpleNamespace(cap=8192))
        rope.cs.assert_called_once_with(True, 8192)


if __name__ == "__main__":
    unittest.main()
