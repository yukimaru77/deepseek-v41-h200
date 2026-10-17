"""CPU-only snapshot retention regression tests; extract the two callbacks without importing CUDA modules."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "src/tensorfold/families/deepseek_v41/cuda/multi.py"
CHUNK = 2048


def callbacks(budget, shrink=True):
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MultiDecoder")
    select = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_kept_marks")
    select.decorator_list = []
    # Its only runtime dependency is PREFILL_CHUNK; the allocator and CUDA imports are irrelevant here.
    select.body = [n for n in select.body if not isinstance(n, ast.ImportFrom)]
    admit = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_admit")
    snap = next(n for n in ast.walk(admit) if isinstance(n, ast.FunctionDef) and n.name == "snap")
    ns = {"C": CHUNK, "KEEP_MARKS": budget, "KEEP_SHRINK": shrink}
    exec(compile(ast.Module(body=[select, snap], type_ignores=[]), str(SOURCE), "exec"), ns)
    return ns


class SnapshotBudgetTest(unittest.TestCase):
    def test_dense_and_doubling_boundaries_obey_budget(self):
        for shrink in (False, True):
            for budget in (1, 2, 3, 10):
                with self.subTest(shrink=shrink, budget=budget):
                    choose = callbacks(budget, shrink)["_kept_marks"]
                    for marks in (list(range(1, 265)), [2 ** i for i in range(20)]):
                        bounds = [i * CHUNK for i in marks]
                        kept = choose(dict.fromkeys(bounds))
                        self.assertEqual(len(kept), budget)
                        self.assertEqual(kept[-1], bounds[-1])
                        if budget >= 2:
                            self.assertEqual(kept[-2:], bounds[-2:])
                        self.assertTrue(set(kept) <= set(bounds))

    def test_small_inputs_keep_all_boundaries(self):
        choose = callbacks(10)["_kept_marks"]
        self.assertEqual(choose({}), [])
        self.assertEqual(choose({8192: object(), 2048: object()}), [2048, 8192])

    def test_long_continuations_bound_active_snapshots(self):
        ns = callbacks(10)
        state = SimpleNamespace(snaps={})
        captures = []

        def snapshot(index):
            self.assertEqual(index, 7)
            self.assertLessEqual(len(state.snaps), 10)
            captures.append(index)
            return object()

        ns.update(self=SimpleNamespace(_kept_marks=ns["_kept_marks"], _snapshot=snapshot),
                  s=state, index=7, marks=set())
        for turn in range(16):
            # Each new continuation inherits the preceding entry's retained snapshots.
            state.snaps = {b: state.snaps[b] for b in ns["_kept_marks"](state.snaps)}
            ns["marks"] = {(turn * 264 + i) * CHUNK for i in range(1, 265)}
            for end in sorted(ns["marks"]):
                ns["snap"](end)
                self.assertLessEqual(len(state.snaps), 10)
                self.assertEqual(max(state.snaps), end)
                if end > CHUNK:
                    self.assertIn(end - CHUNK, state.snaps)
            before = dict(state.snaps)
            ns["snap"](end + 1)  # Non-boundaries do not allocate or prune.
            self.assertEqual(state.snaps, before)
        self.assertEqual(len(captures), 16 * 264)


if __name__ == "__main__":
    unittest.main()
