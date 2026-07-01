from catanrl.eval.parallel_mcts_eval import (
    ParallelMCTSEvalResult,
    _assign_episode_seeds,
)
from catanrl.utils.seeding import derive_seed


def test_assign_episode_seeds_preserves_serial_seed_set_across_workers():
    assignments = _assign_episode_seeds(num_games=5, num_workers=2, seed=42)

    assert assignments == [
        [derive_seed(42, "episode", index) for index in (0, 2, 4)],
        [derive_seed(42, "episode", index) for index in (1, 3)],
    ]
    assert sorted(seed for assignment in assignments for seed in assignment) == sorted(
        derive_seed(42, "episode", index) for index in range(5)
    )


def test_parallel_mcts_eval_result_merges_worker_payloads():
    result = ParallelMCTSEvalResult()

    result.merge(
        {
            "wins": 1,
            "vps": [15, 11],
            "total_vps": [26, 25],
            "turns": [120, 130],
        }
    )
    result.merge(
        {
            "wins": 1,
            "vps": [15],
            "total_vps": [24],
            "turns": [110],
        }
    )

    assert result.games == 3
    assert result.wins == 2
    assert result.vps == [15, 11, 15]
    assert result.total_vps == [26, 25, 24]
    assert result.turns == [120, 130, 110]
