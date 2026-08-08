import multiprocessing as mp
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch

from catanrl.algorithms.alphazero import native_self_play
from catanrl.algorithms.alphazero.parallel_self_play import (
    _put_training_result_chunks,
    run_inference_server_workers,
)
from catanrl.models.heads import FlatPolicyHead, WDLValueHead
from catanrl.models.wrappers import PolicyValueNetworkWrapper
from catanrl.players.nn_mcts_player import (
    _CentralNNMCTSInferenceServer,
    _LocalNNMCTSInferenceBackend,
    _RemoteLeafEvaluationRequest,
    _RemoteNNMCTSInferenceBackend,
)
from scripts.eval_mcts_self_play import (
    _assign_episode_indices,
    _empty_serialized_stats,
    _merge_serialized_result,
)


class _PolicyModel(torch.nn.Module):
    """Deterministic 3-logit head for exact numeric assertions."""

    def forward(self, x):
        return torch.stack((x.sum(dim=1), x[:, 0] - x[:, 1], x[:, -1]), dim=1)


class _CriticModel(torch.nn.Module):
    """Deterministic scalar value for exact numeric assertions."""

    def forward(self, x):
        return x.sum(dim=1, keepdim=True) / 10.0


def test_native_self_play_multiplexes_games_inside_each_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = threading.Lock()
    release = threading.Event()
    active = 0
    maximum_active = 0
    started = 0

    def fake_game_result(*, episode_seed, args_dict, inference_backend):
        del args_dict, inference_backend
        nonlocal active, maximum_active, started
        with lock:
            active += 1
            started += 1
            maximum_active = max(maximum_active, active)
            if started >= 2:
                release.set()
        assert release.wait(timeout=2.0)
        time.sleep(0.01)
        with lock:
            active -= 1
        return [], {"games": 1, "seed": episode_seed}

    monkeypatch.setattr(
        native_self_play,
        "_build_native_training_game_result",
        fake_game_result,
    )
    results = list(
        native_self_play._iter_native_training_game_results(
            episode_seeds=(11, 13, 17),
            args_dict={},
            inference_backend=object(),
            game_concurrency=2,
        )
    )

    assert maximum_active == 2
    assert sorted(stats["seed"] for _experiences, stats in results) == [11, 13, 17]


def _silent_game_worker(
    _worker_id,
    _request_queue,
    _response_queue,
    _result_queue,
):
    time.sleep(30.0)


class _RecordingQueue:
    def __init__(self):
        self.messages = []

    def put(self, message):
        self.messages.append(message)


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
    assert result.wdl is None


def test_local_inference_backend_preserves_shared_wdl_probabilities():
    model = PolicyValueNetworkWrapper(
        torch.nn.Identity(),
        FlatPolicyHead(3, 3),
        WDLValueHead(3),
    )
    assert isinstance(model.value_head, WDLValueHead)
    assert model.value_head.value_head.bias is not None
    with torch.no_grad():
        model.value_head.value_head.weight.zero_()
        model.value_head.value_head.bias.copy_(torch.log(torch.tensor([0.5, 0.3, 0.2])))
    backend = _LocalNNMCTSInferenceBackend(
        policy_model=model,
        critic_model=None,
        model_type="flat",
        device="cpu",
    )

    result = backend.evaluate_leaf(
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
    )

    assert result.wdl is not None
    np.testing.assert_allclose(result.wdl, [0.5, 0.3, 0.2], atol=1e-7)
    assert result.value == pytest.approx(0.3)


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


def test_remote_inference_backend_times_out_instead_of_waiting_forever():
    ctx = mp.get_context("spawn")
    request_queue = ctx.Queue()
    response_queue = ctx.Queue()
    backend = _RemoteNNMCTSInferenceBackend(
        worker_id=0,
        request_queue=request_queue,
        response_queue=response_queue,
        response_timeout_s=0.05,
    )

    try:
        with pytest.raises(RuntimeError, match="Timed out waiting"):
            backend.evaluate_leaf(
                np.ones(3, dtype=np.float32),
                np.ones(2, dtype=np.float32),
            )
    finally:
        backend.close()


def test_self_play_coordinator_reports_stalled_workers():
    error = run_inference_server_workers(
        policy_model=_PolicyModel(),
        critic_model=_CriticModel(),
        model_type="flat",
        device="cpu",
        num_workers=1,
        inference_batch_size=4,
        inference_wait_ms=1.0,
        worker_target=_silent_game_worker,
        worker_args=[()],
        handle_result=lambda _message: None,
        total=1,
        show_tqdm=False,
        stall_timeout_s=0.1,
    )

    assert error is not None
    assert "no worker-result or neural-inference progress" in error
    assert "0:pid=" in error


def test_training_results_are_streamed_in_bounded_chunks():
    result_queue = _RecordingQueue()

    _put_training_result_chunks(
        result_queue=result_queue,  # type: ignore[arg-type]
        worker_id=3,
        experiences=list(range(5)),  # type: ignore[arg-type]
        stats={"games": 1, "wins_RED": 1},
        chunk_size=2,
    )

    assert [message["experiences"] for message in result_queue.messages] == [
        [0, 1],
        [2, 3],
        [4],
    ]
    assert [message["games"] for message in result_queue.messages] == [1, 0, 0]
    assert [message["stats"] for message in result_queue.messages] == [
        {"games": 1, "wins_RED": 1},
        {},
        {},
    ]


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
    assert server.stats() == (2, 1)


def test_central_inference_server_transports_shared_wdl_probabilities():
    model = PolicyValueNetworkWrapper(
        torch.nn.Identity(),
        FlatPolicyHead(3, 3),
        WDLValueHead(3),
    )
    assert isinstance(model.value_head, WDLValueHead)
    assert model.value_head.value_head.bias is not None
    with torch.no_grad():
        model.value_head.value_head.weight.zero_()
        model.value_head.value_head.bias.copy_(torch.log(torch.tensor([0.4, 0.4, 0.2])))
    ctx = mp.get_context("spawn")
    request_queue = ctx.Queue()
    response_queue = ctx.Queue()
    server = _CentralNNMCTSInferenceServer(
        policy_model=model,
        critic_model=None,
        model_type="flat",
        device="cpu",
        request_queue=request_queue,
        response_queues=[response_queue],
        max_batch_size=4,
        max_wait_ms=1.0,
    )

    server.start()
    try:
        request_queue.put(
            _RemoteLeafEvaluationRequest(
                request_id=30,
                worker_id=0,
                actor_features=np.array([1.0, 2.0, 3.0], dtype=np.float32),
                critic_features=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            )
        )
        response = response_queue.get(timeout=5.0)
    finally:
        server.stop()

    np.testing.assert_allclose(response["wdl"], [0.4, 0.4, 0.2], atol=1e-7)
    assert response["value"] == pytest.approx(0.2)


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
