from __future__ import annotations

from unittest.mock import Mock, patch

from catanrl.eval.training_eval import eval_policy_value_against_baselines


def test_eval_policy_value_against_baselines_reports_first_second_and_combined_winrates():
    seat_rates = {"first": 0.8, "second": 0.2}
    observed_calls: list[tuple[str, int]] = []

    def fake_run_policy_value_eval_vectorized(*args, **kwargs):
        seat = kwargs["nn_seat"]
        num_games = kwargs["num_games"]
        observed_calls.append((seat, num_games))
        wins = int(seat_rates[seat] * num_games)
        turns = [10 if seat == "first" else 20] * num_games
        value_preds = [1.0 if seat == "first" else 2.0] * num_games
        returns = [0.5 if seat == "first" else 1.5] * num_games
        expert_labels = [0] * num_games
        expert_masked_preds = [0] * num_games
        return wins, turns, value_preds, returns, expert_labels, expert_masked_preds

    policy_model = Mock()
    critic_model = Mock()

    with patch(
        "catanrl.eval.training_eval.run_policy_value_eval_vectorized",
        side_effect=fake_run_policy_value_eval_vectorized,
    ):
        metrics = eval_policy_value_against_baselines(
            policy_model=policy_model,
            critic_model=critic_model,
            model_type="flat",
            map_type="BASE",
            num_envs=4,
            num_games=10,
            gamma=0.99,
            eval_opponent_configs=["F"],
            log_to_wandb=False,
        )

    assert observed_calls == [
        ("first", 5),
        ("second", 5),
        ("first", 5),
        ("second", 5),
    ]
    assert metrics["eval/games_vs_random_first"] == 5.0
    assert metrics["eval/games_vs_random_second"] == 5.0
    assert metrics["eval/win_rate_vs_random_first"] == 0.8
    assert metrics["eval/win_rate_vs_random_second"] == 0.2
    assert metrics["eval/win_rate_vs_random"] == 0.5
    assert metrics["eval/win_rate_vs_value_first"] == 0.8
    assert metrics["eval/win_rate_vs_value_second"] == 0.2
    assert metrics["eval/win_rate_vs_value"] == 0.5
    assert metrics["eval/avg_turns_vs_random"] == 15.0
    assert metrics["eval/avg_turns_vs_value"] == 15.0


def test_eval_policy_value_against_baselines_requires_even_games_for_seat_split():
    with patch("catanrl.eval.training_eval.run_policy_value_eval_vectorized"):
        try:
            eval_policy_value_against_baselines(
                policy_model=Mock(),
                critic_model=Mock(),
                model_type="flat",
                map_type="BASE",
                num_envs=2,
                num_games=3,
                gamma=0.99,
                eval_opponent_configs=["F"],
                log_to_wandb=False,
            )
        except ValueError as exc:
            assert "even num_games" in str(exc)
        else:
            raise AssertionError("Expected odd num_games seat-split eval to fail.")
