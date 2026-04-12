"""Run a bot against OpenHex: queue, sync state via event replay, act on your turn.

Replace `RandomPlayer(your_color)` with e.g. `NNPolicyPlayer(...)` once you load a
checkpoint; keep the same `Color` as `your_color`.
"""

from __future__ import annotations

import argparse
import time

from catanatron.models.player import Color, RandomPlayer

from catanrl.openhex.board import build_catan_map_from_openhex_board
from catanrl.openhex.client import DEFAULT_BASE_URL, OpenHexClient
from catanrl.openhex.convert import oh_to_color, openhex_post_bodies_for_action
from catanrl.openhex.replay import build_game_from_openhex, replay_openhex_events


def run_daemon(
    api_key: str,
    *,
    base_url: str | None = None,
    mode: str = "1v1",
    poll_interval: float = 1.0,
    queue_poll: float = 5.0,
) -> None:
    client = OpenHexClient(api_key, base_url=base_url or DEFAULT_BASE_URL)

    client.queue_join(mode=mode)
    try:
        cached_board: dict | None = None
        last_game_id: str | None = None

        while True:
            games = client.games_active()
            if not games:
                time.sleep(queue_poll)
                continue

            ginfo = games[0]
            game_id = ginfo["id"]
            st = client.get_state(game_id)

            if cached_board is None or game_id != last_game_id:
                board = st.get("state", {}).get("board") or {}
                tiles = board.get("tiles")
                ports = board.get("ports")
                if tiles and ports:
                    cached_board = {"tiles": tiles, "ports": ports}
                elif cached_board is None:
                    time.sleep(poll_interval)
                    continue

            assert cached_board is not None
            catan_map = build_catan_map_from_openhex_board(cached_board)
            your_color = oh_to_color(st["your_color"])
            turn_order = st["state"]["turnOrder"]

            ev_resp = client.get_events(game_id, after=-1)
            events_wrapped = ev_resp.get("events", [])
            inner_events = [w["event"] for w in events_wrapped]

            policy = RandomPlayer(your_color, is_bot=True)
            game = build_game_from_openhex(turn_order, catan_map, policy, your_color)

            try:
                replay_openhex_events(game, catan_map, inner_events)
            except (NotImplementedError, ValueError) as e:
                raise RuntimeError(
                    "Event replay failed (often hidden opponent dev cards). "
                    f"Details: {e}"
                ) from e

            gi = st["game"]
            if gi.get("status") != "active":
                client.queue_join(mode=mode)
                cached_board = None
                last_game_id = None
                continue

            current = gi["current_player_color"]
            if current != st["your_color"]:
                time.sleep(poll_interval)
                last_game_id = game_id
                continue

            action = policy.decide(game, game.playable_actions)
            bodies = openhex_post_bodies_for_action(
                action,
                catan_map,
                duration_ms=None,
                is_initial_build_phase=game.state.is_initial_build_phase,
            )
            for body in bodies:
                client.post_action(game_id, body)

            last_game_id = game_id
            time.sleep(poll_interval)
    finally:
        try:
            client.queue_leave(mode=mode)
        except Exception:
            pass


def main() -> None:
    p = argparse.ArgumentParser(description="OpenHex bot daemon (RandomPlayer by default)")
    p.add_argument("--api-key", required=True, help="Bearer token from POST /api/bots")
    p.add_argument(
        "--base-url",
        default="https://open-hex.web.app/api",
        help="API base URL",
    )
    p.add_argument("--mode", default="1v1", choices=("1v1", "4p"))
    args = p.parse_args()
    run_daemon(args.api_key, base_url=args.base_url, mode=args.mode)


if __name__ == "__main__":
    main()
