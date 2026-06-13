from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import threading

import numpy as np
import torch

from scripts.eval_mcts_self_play import (
    _assign_episode_indices,
    _empty_serialized_stats,
    _merge_serialized_result,
)
from catanrl.players.nn_mcts_player import (
    _BatchedNNMCTSInferenceBackend,
    _CentralNNMCTSInferenceServer,
    _LocalNNMCTSInferenceBackend,
    _RemoteLeafEvaluationRequest,
    _RemoteNNMCTSInferenceBackend,
)


class _PolicyModel(torch.nn.Module):
    def forward(self, x):
        return torch.stack((x.sum(dim=1), x[:, 0] - x[:, 1], x[:, -1]), dim=1)


class _CriticModel(torch.nn.Module):
    def forward(self, x):
        return x.sum(dim=1, keepdim=True) / 10.0


def test_local_inference_backend_runs_separate_policy_and_critic_models():
    backend = _LocalNNMCTSInferenceBackend(
        policy_model=_PolicyModel(),
        critic_model=_CriticModel(),
        model_type="flat",
        device="cpu",
    )

    result = backend.evaluate_leaf(
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
        np.array([2.0, 3.0], dtype=np.float32),
    )

    np.testing.assert_allclose(result.policy_logits, np.array([6.0, -1.0, 3.0], dtype=np.float32))
    assert result.value == 0.5


def test_batched_inference_backend_correlates_parallel_leaf_requests():
    backend = _BatchedNNMCTSInferenceBackend(
        policy_model=_PolicyModel(),
        critic_model=_CriticModel(),
        model_type="flat",
        device="cpu",
        max_batch_size=8,
        max_wait_ms=10.0,
    )

    actor_inputs = [
        np.array([float(i), float(i + 1), float(i + 2)], dtype=np.float32)
        for i in range(4)
    ]
    critic_inputs = [
        np.array([float(i), float(i * 2)], dtype=np.float32)
        for i in range(4)
    ]

    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(
                executor.map(
                    lambda pair: backend.evaluate_leaf(*pair),
                    zip(actor_inputs, critic_inputs),
                )
            )
    finally:
        backend.close()

    for i, result in enumerate(results):
        np.testing.assert_allclose(
            result.policy_logits,
            np.array([3 * i + 3, -1.0, i + 2], dtype=np.float32),
        )
        np.testing.assert_allclose(result.value, (3 * i) / 10.0)


def test_remote_inference_backend_correlates_parallel_leaf_requests():
    ctx = mp.get_context("spawn")
    request_queue = ctx.Queue()
    response_queue = ctx.Queue()
    backend = _RemoteNNMCTSInferenceBackend(
        worker_id=0,
        request_queue=request_queue,
        response_queue=response_queue,
    )

    def responder():
        requests = [request_queue.get() for _ in range(4)]
        for request in reversed(requests):
            response_queue.put(
                {
                    "request_id": request.request_id,
                    "policy_logits": request.actor_features + 10.0,
                    "value": float(request.critic_features.sum()),
                }
            )

    thread = threading.Thread(target=responder)
    thread.start()

    try:
        actor_inputs = [
            np.array([float(i), float(i + 1), float(i + 2)], dtype=np.float32)
            for i in range(4)
        ]
        critic_inputs = [
            np.array([float(i), float(i * 2)], dtype=np.float32)
            for i in range(4)
        ]

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(
                executor.map(
                    lambda pair: backend.evaluate_leaf(*pair),
                    zip(actor_inputs, critic_inputs),
                )
            )
    finally:
        backend.close()
        response_queue.put(None)
        thread.join(timeout=5.0)

    for i, result in enumerate(results):
        np.testing.assert_allclose(result.policy_logits, actor_inputs[i] + 10.0)
        np.testing.assert_allclose(result.value, critic_inputs[i].sum())


def test_central_inference_server_batches_mixed_worker_requests():
    ctx = mp.get_context("spawn")
    request_queue = ctx.Queue()
    response_queues = [ctx.Queue(), ctx.Queue()]
    server = _CentralNNMCTSInferenceServer(
        policy_model=_PolicyModel(),
        critic_model=_CriticModel(),
        model_type="flat",
        device="cpu",
        request_queue=request_queue,
        response_queues=response_queues,
        max_batch_size=8,
        max_wait_ms=20.0,
    )

    server.start()
    try:
        requests = [
            _RemoteLeafEvaluationRequest(
                request_id=10,
                worker_id=0,
                actor_features=np.array([1.0, 2.0, 3.0], dtype=np.float32),
                critic_features=np.array([2.0, 3.0], dtype=np.float32),
            ),
            _RemoteLeafEvaluationRequest(
                request_id=20,
                worker_id=1,
                actor_features=np.array([4.0, 5.0, 6.0], dtype=np.float32),
                critic_features=np.array([1.0, 2.0], dtype=np.float32),
            ),
        ]
        for request in requests:
            request_queue.put(request)

        response_0 = response_queues[0].get(timeout=5.0)
        response_1 = response_queues[1].get(timeout=5.0)
    finally:
        server.stop()

    assert response_0["request_id"] == 10
    np.testing.assert_allclose(response_0["policy_logits"], np.array([6.0, -1.0, 3.0]))
    np.testing.assert_allclose(response_0["value"], 0.5)
    assert response_1["request_id"] == 20
    np.testing.assert_allclose(response_1["policy_logits"], np.array([15.0, -1.0, 6.0]))
    np.testing.assert_allclose(response_1["value"], 0.3)


def test_parallel_self_play_assignment_and_merge_helpers_preserve_totals():
    assert _assign_episode_indices(5, 2) == [[0, 2, 4], [1, 3]]

    aggregate = _empty_serialized_stats(num_players=2)
    turns = []
    _merge_serialized_result(
        aggregate,
        turns,
        {
            "stats": {
                "RED": {"wins": 1, "vps": [10, 8]},
                "BLUE": {"wins": 0, "vps": [6, 10]},
            },
            "turns": [12, 14],
        },
    )
    _merge_serialized_result(
        aggregate,
        turns,
        {
            "stats": {
                "RED": {"wins": 0, "vps": [7]},
                "BLUE": {"wins": 1, "vps": [10]},
            },
            "turns": [11],
        },
    )

    assert aggregate["RED"]["wins"] == 1
    assert aggregate["BLUE"]["wins"] == 1
    assert aggregate["RED"]["vps"] == [10, 8, 7]
    assert aggregate["BLUE"]["vps"] == [6, 10, 10]
    assert turns == [12, 14, 11]
