"""Tests for single-instance colony toolkit (Thrivemind)."""

from __future__ import annotations

import random
import time

from library.harness.store import open_store, NamespacedStore
from library.tools.identity import AXIS_NAMES, Identity, format_persona
from library.tools.voting import borda_tally
from library.tools.thrivemind import (
    ThrivemindConfig,
    build_colony_snapshot,
    select_suggesters,
    run_spawn_cycle,
    update_approvals,
    load_colony,
    save_colony,
    spawn_initial_colony,
    load_config,
)


# ---------------------------------------------------------------------------
# Minimal mock context for colony store tests
# ---------------------------------------------------------------------------


class DummyCtx:
    def __init__(self, instance_id: str, store_db, thrivemind_cfg=None, files=None):
        self.instance_id = instance_id
        self._store_db = store_db
        self._thrivemind_cfg = thrivemind_cfg or {}
        self._files: dict[str, str] = files or {}

    def config(self, key: str):
        if key == "thrivemind":
            return self._thrivemind_cfg
        return None

    def store(self, namespace: str):
        return NamespacedStore(self._store_db, f"instance:{self.instance_id}:{namespace}")

    def read(self, path: str) -> str:
        return self._files.get(path, "")

    def write(self, path: str, content: str) -> None:
        self._files[path] = content

    def exists(self, path: str) -> bool:
        return path in self._files


# ---------------------------------------------------------------------------
# format_persona (via Identity with dims)
# ---------------------------------------------------------------------------


def _ind(dims: dict[str, float]) -> Identity:
    return Identity(name="test", dims=dims)


class TestFormatPersona:
    def test_extreme_trait(self):
        dims = {name: 0.0 for name in AXIS_NAMES}
        dims["conservative_liberal"] = 0.9
        result = format_persona(_ind(dims))
        assert "extremely" in result
        assert "conservative" in result

    def test_negative_pole(self):
        dims = {name: 0.0 for name in AXIS_NAMES}
        dims["optimistic_pessimistic"] = -0.7
        result = format_persona(_ind(dims))
        assert "very" in result
        assert "pessimistic" in result

    def test_midrange_label(self):
        dims = {name: 0.0 for name in AXIS_NAMES}
        dims["cautious_bold"] = 0.5
        result = format_persona(_ind(dims))
        assert "fairly" in result
        assert "cautious" in result

    def test_all_near_zero_fallback(self):
        dims = {name: 0.0 for name in AXIS_NAMES}
        dims["analytical_emotional"] = 0.1
        result = format_persona(_ind(dims))
        assert "barely" in result
        assert "analytical" in result

    def test_traits_ordered_by_magnitude(self):
        dims = {name: 0.0 for name in AXIS_NAMES}
        dims["conservative_liberal"] = 0.3
        dims["analytical_emotional"] = 0.85
        result = format_persona(_ind(dims))
        assert result.index("analytical") < result.index("conservative")


class TestColonySnapshot:
    def test_build_colony_snapshot_contains_table_and_sorted_by_approval(self):
        colony = [
            Identity(name="a", dims={n: 0.0 for n in AXIS_NAMES}, approval=1, created_at=1),
            Identity(name="b", dims={n: 0.0 for n in AXIS_NAMES}, approval=3, created_at=1),
        ]

        snapshot = build_colony_snapshot(colony)

        assert "# Colony" in snapshot
        assert "| Individual | Personality | Approval | Age |" in snapshot
        assert "| `b` |" in snapshot
        assert "| `a` |" in snapshot
        assert snapshot.index("| `b` |") < snapshot.index("| `a` |")


# ---------------------------------------------------------------------------
# select_suggesters
# ---------------------------------------------------------------------------


class TestSelectSuggesters:
    def _colony(self, approvals: list[int]) -> list[Identity]:
        return [
            Identity(name=str(i), dims={n: 0.0 for n in AXIS_NAMES}, approval=a)
            for i, a in enumerate(approvals)
        ]

    def test_approval_weighting_favors_higher(self):
        colony = self._colony([0, 5, 5, 5])
        rng = random.Random(42)
        counts = {ind.name: 0 for ind in colony}
        for _ in range(200):
            selected = select_suggesters(colony, 1, rng=rng)
            counts[selected[0].name] += 1
        assert counts["0"] < counts["1"]

    def test_negative_approval_excluded(self):
        colony = self._colony([-5, -5, -5])
        rng = random.Random(1)
        result = select_suggesters(colony, 2, rng=rng)
        assert len(result) == 2

    def test_n_greater_than_colony_size_caps(self):
        colony = self._colony([1, 2, 3])
        result = select_suggesters(colony, 10)
        assert len(result) == 3

    def test_no_duplicates(self):
        colony = self._colony([1, 2, 3, 4])
        rng = random.Random(7)
        for _ in range(50):
            result = select_suggesters(colony, 3, rng=rng)
            names = [ind.name for ind in result]
            assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# tally_borda (from voting module)
# ---------------------------------------------------------------------------


class TestTallyBorda:
    def test_correct_winner(self):
        candidates = {"a": "option a", "b": "option b", "c": "option c"}
        votes = {
            "v1": ["b", "a", "c"],
            "v2": ["b", "c", "a"],
            "v3": ["a", "b", "c"],
        }
        tally = borda_tally(candidates, votes)
        assert tally["winner_member"] == "b"
        assert tally["candidate_count"] == 3
        assert tally["vote_count"] == 3

    def test_returns_correct_fields(self):
        candidates = {"x": "text x", "y": "text y"}
        votes = {"v1": ["x", "y"]}
        tally = borda_tally(candidates, votes)
        assert "winner_member" in tally
        assert "winner_message" in tally
        assert "scores" in tally
        assert tally["winner_message"] == "text x"

    def test_no_votes_all_score_one(self):
        candidates = {"a": "aa", "b": "bb"}
        tally = borda_tally(candidates, {})
        assert tally["scores"]["a"] == 1
        assert tally["scores"]["b"] == 1

    def test_single_candidate_wins(self):
        candidates = {"only": "text"}
        tally = borda_tally(candidates, {})
        assert tally["winner_member"] == "only"


# ---------------------------------------------------------------------------
# run_spawn_cycle
# ---------------------------------------------------------------------------


def _make_colony(approvals: list[int], cfg: ThrivemindConfig) -> list[Identity]:
    now = int(time.time())
    return [
        Identity(
            name=str(i),
            dims={n: 0.1 for n in AXIS_NAMES},
            approval=a,
            created_at=now,
        )
        for i, a in enumerate(approvals)
    ]


class TestRunSpawnCycle:
    def test_colony_stays_within_bounds(self):
        cfg = ThrivemindConfig(min_colony_size=6, max_colony_size=6, approval_threshold=2)
        colony = _make_colony([0, 0, 0, 3, 4, 5], cfg)
        rng = random.Random(42)
        new_colony = run_spawn_cycle(colony, cfg, rng=rng)
        assert len(new_colony) == 6

    def test_colony_grows_when_under_max(self):
        cfg = ThrivemindConfig(min_colony_size=4, max_colony_size=8, approval_threshold=2)
        colony = _make_colony([0, 0, 3, 4], cfg)
        rng = random.Random(42)
        new_colony = run_spawn_cycle(colony, cfg, rng=rng)
        # Parents survive, offspring added
        assert len(new_colony) > 4
        assert len(new_colony) <= 8
        # Original members still present
        original_names = {ind.name for ind in colony}
        assert original_names.issubset({ind.name for ind in new_colony})

    def test_no_eligible_returns_unchanged(self):
        cfg = ThrivemindConfig(min_colony_size=4, max_colony_size=4, approval_threshold=5)
        colony = _make_colony([0, 1, 2, 3], cfg)
        new_colony = run_spawn_cycle(colony, cfg)
        assert len(new_colony) == len(colony)
        assert {ind.name for ind in new_colony} == {ind.name for ind in colony}

    def test_offspring_have_zero_approval(self):
        cfg = ThrivemindConfig(min_colony_size=4, max_colony_size=6, approval_threshold=2)
        colony = _make_colony([0, 0, 3, 4], cfg)
        rng = random.Random(99)
        new_colony = run_spawn_cycle(colony, cfg, rng=rng)
        original_names = {ind.name for ind in colony}
        for ind in new_colony:
            if ind.name not in original_names:
                assert ind.approval == 0

    def test_eligible_individuals_replaced_at_max(self):
        cfg = ThrivemindConfig(min_colony_size=4, max_colony_size=4, approval_threshold=3)
        colony = _make_colony([0, 0, 4, 5], cfg)
        rng = random.Random(1)
        new_colony = run_spawn_cycle(colony, cfg, rng=rng)
        new_names = {ind.name for ind in new_colony}
        assert "2" not in new_names
        assert "3" not in new_names

    def test_dim_inheritance(self):
        cfg = ThrivemindConfig(min_colony_size=2, max_colony_size=2, approval_threshold=1)
        primary = Identity(
            name="parent",
            dims={n: 0.5 for n in AXIS_NAMES},
            approval=3,
            created_at=0,
        )
        colony = [primary]
        rng = random.Random(0)
        new_colony = run_spawn_cycle(colony, cfg, rng=rng)
        assert len(new_colony) == 2
        for ind in new_colony:
            for val in ind.dims.values():
                assert -1.0 <= val <= 1.0

    def test_names_are_evocative(self):
        cfg = ThrivemindConfig(min_colony_size=4, max_colony_size=6, approval_threshold=2)
        colony = _make_colony([3, 4, 0, 0], cfg)
        rng = random.Random(42)
        new_colony = run_spawn_cycle(colony, cfg, rng=rng)
        original_names = {ind.name for ind in colony}
        for ind in new_colony:
            if ind.name not in original_names:
                parts = ind.name.split("-")
                assert len(parts) == 3, f"Expected adjective-adjective-noun, got: {ind.name}"


# ---------------------------------------------------------------------------
# update_approvals
# ---------------------------------------------------------------------------


class TestUpdateApprovals:
    def _colony(self, ids_approvals: list[tuple[str, int]]) -> list[Identity]:
        return [
            Identity(name=i, dims={n: 0.0 for n in AXIS_NAMES}, approval=a)
            for i, a in ids_approvals
        ]

    def test_winner_gets_plus_one_per_top_two_voter(self):
        colony = self._colony([("a", 0), ("b", 0), ("c", 0)])
        votes = {
            "a": ["b", "a", "c"],  # b in top 2 → +1
            "c": ["c", "b", "a"],  # b in top 2 → +1
        }
        cfg = ThrivemindConfig()
        updated = update_approvals(colony, votes, "b", cfg)
        id_map = {ind.name: ind for ind in updated}
        assert id_map["b"].approval == 2

    def test_voter_with_winner_in_second_pick_still_rewards(self):
        """Top-2 voting: second pick for winner also gives +1."""
        colony = self._colony([("a", 0), ("b", 0), ("c", 0)])
        votes = {
            "a": ["c", "b", "a"],  # b is 2nd pick → still +1 for b
        }
        cfg = ThrivemindConfig()
        updated = update_approvals(colony, votes, "b", cfg)
        id_map = {ind.name: ind for ind in updated}
        assert id_map["b"].approval == 1
        assert id_map["a"].approval == 0  # voted for winner in top 2, no penalty

    def test_non_winner_voter_loses_one(self):
        colony = self._colony([("a", 0), ("b", 0), ("c", 0)])
        votes = {
            "a": ["a", "c", "b"],  # b not in top 2 → -1 for voter a
            "b": ["b", "a"],       # b in top 2 → +1 for b
        }
        cfg = ThrivemindConfig()
        updated = update_approvals(colony, votes, "b", cfg)
        id_map = {ind.name: ind for ind in updated}
        assert id_map["a"].approval == -1
        assert id_map["b"].approval == 1

    def test_no_votes_no_change(self):
        colony = self._colony([("x", 5)])
        cfg = ThrivemindConfig()
        updated = update_approvals(colony, {}, "x", cfg)
        assert updated[0].approval == 5
