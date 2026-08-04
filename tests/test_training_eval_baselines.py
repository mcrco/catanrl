from unittest.mock import patch

import pytest

from catanrl.eval.training_eval import eval_policy_value_against_baselines


class _Model:
    def eval(self):
        return self


def _rollout_result(**kwargs):
    return 1, [10], [0.0], [0.0], [], []


def test_value_only_baseline_skips_random_eval():
    with patch(
        "catanrl.eval.training_eval.run_policy_value_eval_vectorized",
        side_effect=_rollout_result,
    ) as rollout:
        metrics = eval_policy_value_against_baselines(
            policy_model=_Model(),
            critic_model=_Model(),
            model_type="flat",
            map_type="BASE",
            num_envs=1,
            num_games=2,
            eval_baselines=["value"],
            number_placement="official_spiral",
            device="cpu",
            log_to_wandb=False,
        )

    assert rollout.call_count == 2
    assert all(call.kwargs["opponent_configs"] == ["F"] for call in rollout.call_args_list)
    assert all(
        call.kwargs["number_placement"] == "official_spiral" for call in rollout.call_args_list
    )
    assert metrics["eval/win_rate_vs_value"] == 1.0
    assert "eval/win_rate_vs_random" not in metrics


@pytest.mark.parametrize("baselines", [[], ["value", "value"], ["unknown"]])
def test_invalid_eval_baselines_are_rejected(baselines):
    with pytest.raises(ValueError):
        eval_policy_value_against_baselines(
            policy_model=_Model(),
            critic_model=_Model(),
            model_type="flat",
            map_type="BASE",
            num_envs=1,
            num_games=2,
            eval_baselines=baselines,
            device="cpu",
            log_to_wandb=False,
        )
