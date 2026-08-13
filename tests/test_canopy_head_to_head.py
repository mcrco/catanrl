from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np

from catanrl.eval.canopy_head_to_head import BatchingCanopyBackend, play_head_to_head_game


class _FirstLegalCanopyBackend:
    def register(self, _game) -> None:
        pass

    def unregister(self, _game) -> None:
        pass

    def decide(self, _game, actions):
        return actions[0]


def test_canopy_batcher_retains_idle_live_search_trees() -> None:
    calls = []

    class _Bridge:
        def decide_many(self, games_and_actions, *, active_game_ids):
            calls.append((games_and_actions, set(active_game_ids)))
            return [actions[0] for _game, actions in games_and_actions]

    first = SimpleNamespace(id="first")
    second = SimpleNamespace(id="second")
    backend = BatchingCanopyBackend(
        _Bridge(),  # type: ignore[arg-type]
        threading.Lock(),
        max_batch_size=2,
        max_wait_ms=50.0,
    )
    backend.register(first)  # type: ignore[arg-type]
    backend.register(second)  # type: ignore[arg-type]
    backend.start()
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(backend.decide, first, (11,)),
                executor.submit(backend.decide, second, (13,)),
            ]
            assert [future.result() for future in futures] == [11, 13]
    finally:
        backend.close()

    assert len(calls) == 1
    assert calls[0][1] == {"first", "second"}


def test_head_to_head_keeps_catanatron_authoritative(monkeypatch) -> None:
    def fake_search(*, game, **_kwargs):
        return SimpleNamespace(action=int(np.flatnonzero(game.valid_action_mask())[0]))

    monkeypatch.setattr("catanrl.eval.canopy_head_to_head._search", fake_search)

    record = play_head_to_head_game(
        episode_seed=601,
        candidate_seat=1,
        budget=1,
        args_dict={
            "map_type": "BASE",
            "discard_limit": 9,
            "vps_to_win": 15,
            "turns_limit": 1000,
            "max_actions": 20,
            "tree_reuse": False,
            "c_puct": 2.5,
            "canonical_pruning": True,
            "c_visit": 50.0,
            "c_scale": 1.0,
        },
        actor_indices=np.array([0]),
        critic_indices=np.array([0]),
        inference_backend=object(),  # type: ignore[arg-type]
        canopy_backend=_FirstLegalCanopyBackend(),  # type: ignore[arg-type]
        catanatron_lock=threading.Lock(),
    )

    assert record["authoritative_engine"] == "Catanatron"
    assert record["opponent"] == "cullback/canopy nexus-v3"
    assert record["seat"] == "second"
    assert record["actions"] >= 20
