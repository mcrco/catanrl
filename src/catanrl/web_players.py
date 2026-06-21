from __future__ import annotations

import logging

from catanrl.players.factory import validate_game_context
from catanrl.players.lazy_player import LazyConfiguredPlayer
from catanrl.players.player_config import load_all_player_specs


def register_players(registry) -> None:
    """Register CatanRL player agents from configs/players/ (+ optional auto-discover)."""
    specs = load_all_player_specs()
    if not specs:
        logging.info("No CatanRL player specs found; skipping web player registration.")
        return

    registered = 0
    for spec in specs:
        try:
            num_players = spec.game_num_players()
            map_type = spec.game_map_type()
        except (FileNotFoundError, ValueError) as exc:
            logging.warning("Skipping player spec '%s': %s", spec.id, exc)
            continue

        registry.register(
            key=f"CATANRL:{spec.id}",
            label=spec.display_label(),
            description=spec.display_description(),
            min_players=num_players,
            max_players=num_players,
            map_templates=(map_type,),
            factory=_make_factory(spec.id, num_players, map_type),
        )
        registered += 1
        logging.info("Registered CatanRL web player '%s' (%s)", spec.id, spec.type)

    logging.info("Registered %d CatanRL web player(s).", registered)


def _make_factory(spec_id: str, num_players: int, map_type: str):
    def factory(color, context):
        from catanrl.players.player_config import get_player_spec

        spec = get_player_spec(spec_id)
        validate_game_context(
            spec,
            num_players=context.num_players,
            map_type=context.map_template,
        )
        if context.num_players != num_players or context.map_template != map_type:
            raise ValueError(
                f"Player '{spec_id}' was registered for {num_players}p {map_type}."
            )
        return LazyConfiguredPlayer(spec_id, color)

    return factory
