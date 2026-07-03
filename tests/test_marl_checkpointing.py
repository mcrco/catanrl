import torch

from catanrl.algorithms.ppo.marl_ppo_central_critic import save_best_checkpoint_pair


def test_best_checkpoint_uses_policy_complement_critic(tmp_path):
    policy = torch.nn.Linear(2, 1, bias=False)
    critic = torch.nn.Linear(2, 1, bias=False)

    with torch.no_grad():
        policy.weight.fill_(8.0)
        critic.weight.fill_(2.0)
    save_best_checkpoint_pair(policy, critic, str(tmp_path))

    with torch.no_grad():
        policy.weight.fill_(9.0)
        critic.weight.fill_(9.0)
    save_best_checkpoint_pair(policy, critic, str(tmp_path))

    saved_policy = torch.load(tmp_path / "policy_best.pt", weights_only=True)
    saved_critic = torch.load(tmp_path / "critic_best.pt", weights_only=True)
    torch.testing.assert_close(saved_policy["weight"], torch.full((1, 2), 9.0))
    torch.testing.assert_close(saved_critic["weight"], torch.full((1, 2), 9.0))
