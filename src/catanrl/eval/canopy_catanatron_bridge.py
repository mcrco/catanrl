"""Run a released cullback/Canopy checkpoint as a Catanatron player.

The Catanatron game remains authoritative.  This module serializes a complete
decision state into Canopy's fixed topology/action indexing, sends batches of
states to the companion Rust bridge, and maps the selected Canopy action back
to an action from Catanatron's own ``playable_actions`` list.
"""

from __future__ import annotations

import json
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Sequence

from catanatron.game import Game
from catanatron.models.enums import (
    CITY,
    KNIGHT,
    MONOPOLY,
    RESOURCES,
    ROAD,
    ROAD_BUILDING,
    SETTLEMENT,
    VICTORY_POINT,
    YEAR_OF_PLENTY,
    Action,
    ActionPrompt,
    ActionType,
    EdgeRef,
    NodeRef,
)
from catanatron.models.player import Color, Player, SimplePlayer
from catanatron.models.tiles import Port
from catanatron.state_functions import player_has_rolled, player_key

from catanrl.utils.catanatron_game import force_player_order

CANOPY_LAND_HEXES: tuple[tuple[int, int], ...] = (
    (0, -2),
    (1, -2),
    (2, -2),
    (2, -1),
    (2, 0),
    (1, 1),
    (0, 2),
    (-1, 2),
    (-2, 2),
    (-2, 1),
    (-2, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
    (1, 0),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (0, 0),
)

CANOPY_PORT_HEXES: tuple[tuple[int, int], ...] = (
    (3, 0),
    (1, 2),
    (-1, 3),
    (-3, 3),
    (-3, 1),
    (-2, -1),
    (0, -3),
    (2, -3),
    (3, -2),
)

CANOPY_CORNERS: tuple[NodeRef, ...] = (
    NodeRef.NORTH,
    NodeRef.NORTHEAST,
    NodeRef.SOUTHEAST,
    NodeRef.SOUTH,
    NodeRef.SOUTHWEST,
    NodeRef.NORTHWEST,
)

# Canopy edge i joins corner i and (i + 1) % 6.
CANOPY_EDGES: tuple[EdgeRef, ...] = (
    EdgeRef.NORTHEAST,
    EdgeRef.EAST,
    EdgeRef.SOUTHEAST,
    EdgeRef.SOUTHWEST,
    EdgeRef.WEST,
    EdgeRef.NORTHWEST,
)

CANOPY_DEV_ORDER: tuple[str, ...] = (
    KNIGHT,
    VICTORY_POINT,
    ROAD_BUILDING,
    YEAR_OF_PLENTY,
    MONOPOLY,
)


def _cube(q: int, r: int) -> tuple[int, int, int]:
    return (q, -q - r, r)


@dataclass(frozen=True)
class CanopyTopologyMapping:
    """Static Canopy IDs paired with the current Catanatron map IDs."""

    nodes: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]
    tiles: tuple[tuple[int, int, int], ...]
    node_to_canopy: dict[int, int]
    edge_to_canopy: dict[tuple[int, int], int]
    tile_to_canopy: dict[tuple[int, int, int], int]

    @classmethod
    def from_game(cls, game: Game) -> "CanopyTopologyMapping":
        catan_map = game.state.board.map
        nodes: list[int] = []
        node_to_canopy: dict[int, int] = {}
        edges: list[tuple[int, int]] = []
        edge_to_canopy: dict[tuple[int, int], int] = {}
        tiles = tuple(_cube(q, r) for q, r in CANOPY_LAND_HEXES)

        for coordinate in tiles:
            tile = catan_map.land_tiles[coordinate]
            for corner in CANOPY_CORNERS:
                node = int(tile.nodes[corner])
                if node not in node_to_canopy:
                    node_to_canopy[node] = len(nodes)
                    nodes.append(node)
            for edge_ref in CANOPY_EDGES:
                edge = tuple(sorted(tile.edges[edge_ref]))
                if edge not in edge_to_canopy:
                    edge_to_canopy[edge] = len(edges)
                    edges.append(edge)

        if len(nodes) != 54 or len(edges) != 72:
            raise ValueError(
                f"Canopy BASE topology requires 54 nodes/72 edges, got {len(nodes)}/{len(edges)}"
            )
        return cls(
            nodes=tuple(nodes),
            edges=tuple(edges),
            tiles=tiles,
            node_to_canopy=node_to_canopy,
            edge_to_canopy=edge_to_canopy,
            tile_to_canopy={coordinate: index for index, coordinate in enumerate(tiles)},
        )


def _turn_records(game: Game, color: Color) -> list[Any]:
    records: list[Any] = []
    for record in reversed(game.state.action_records):
        if record.action.color == color and record.action.action_type == ActionType.END_TURN:
            break
        records.append(record)
    records.reverse()
    return records


def _trade_ratios(game: Game, color: Color) -> list[int]:
    ratios = [4] * len(RESOURCES)
    owned_ports = game.state.board.get_player_port_resources(color)
    if None in owned_ports:
        ratios = [3] * len(RESOURCES)
    for port in owned_ports:
        if port is not None:
            ratios[RESOURCES.index(port)] = 2
    return ratios


def _player_snapshot(
    game: Game,
    color: Color,
    tested_non_knight: int,
) -> dict[str, Any]:
    state = game.state
    key = player_key(state, color)
    turn_records = _turn_records(game, color)
    bought = {card: 0 for card in CANOPY_DEV_ORDER}
    for record in turn_records:
        if (
            record.action.color == color
            and record.action.action_type == ActionType.BUY_DEVELOPMENT_CARD
        ):
            bought[record.result] += 1

    settlements = state.buildings_by_color[color][SETTLEMENT]
    cities = state.buildings_by_color[color][CITY]
    return {
        "hand": [int(state.player_state[f"{key}_{resource}_IN_HAND"]) for resource in RESOURCES],
        "dev_cards": [
            int(state.player_state[f"{key}_{card}_IN_HAND"]) for card in CANOPY_DEV_ORDER
        ],
        "dev_cards_bought_this_turn": [bought[card] for card in CANOPY_DEV_ORDER],
        "dev_cards_played": [
            0 if card == VICTORY_POINT else int(state.player_state[f"{key}_PLAYED_{card}"])
            for card in CANOPY_DEV_ORDER
        ],
        "knights_played": int(state.player_state[f"{key}_PLAYED_KNIGHT"]),
        "roads_placed": 15 - int(state.player_state[f"{key}_ROADS_AVAILABLE"]),
        "settlements_left": int(state.player_state[f"{key}_SETTLEMENTS_AVAILABLE"]),
        "cities_left": int(state.player_state[f"{key}_CITIES_AVAILABLE"]),
        "roads_left": int(state.player_state[f"{key}_ROADS_AVAILABLE"]),
        "has_played_dev_card_this_turn": bool(
            state.player_state[f"{key}_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"]
        ),
        "has_traded_this_turn": any(
            record.action.color == color and record.action.action_type == ActionType.MARITIME_TRADE
            for record in turn_records
        ),
        "building_vps": len(settlements) + 2 * len(cities),
        "trade_ratios": _trade_ratios(game, color),
        "tested_non_knight": tested_non_knight,
    }


def _robber_blocks_player(game: Game, color: Color) -> bool:
    state = game.state
    tile = state.board.map.land_tiles[state.board.robber_coordinate]
    if tile.resource is None:
        return False
    buildings = set(state.buildings_by_color[color][SETTLEMENT]) | set(
        state.buildings_by_color[color][CITY]
    )
    return bool(buildings.intersection(tile.nodes.values()))


def _dev_cards_in_hand(game: Game, color: Color) -> int:
    key = player_key(game.state, color)
    return sum(int(game.state.player_state[f"{key}_{card}_IN_HAND"]) for card in CANOPY_DEV_ORDER)


def _update_tested_non_knight_before_action(
    game: Game,
    action: Action,
    tested: list[int],
) -> None:
    if action.action_type != ActionType.END_TURN or not _robber_blocks_player(game, action.color):
        return
    index = game.state.color_to_index[action.color]
    bought_this_turn = sum(
        record.action.color == action.color
        and record.action.action_type == ActionType.BUY_DEVELOPMENT_CARD
        for record in _turn_records(game, action.color)
    )
    eligible_old = max(0, _dev_cards_in_hand(game, action.color) - bought_this_turn)
    tested[index] = max(tested[index], eligible_old)


def _update_tested_non_knight_after_action(
    game: Game,
    action: Action,
    tested: list[int],
) -> None:
    index = game.state.color_to_index[action.color]
    if action.action_type == ActionType.PLAY_KNIGHT_CARD:
        tested[index] = 0
    elif action.action_type in {
        ActionType.PLAY_ROAD_BUILDING,
        ActionType.PLAY_YEAR_OF_PLENTY,
        ActionType.PLAY_MONOPOLY,
    }:
        tested[index] = min(tested[index], _dev_cards_in_hand(game, action.color))


@dataclass
class _CanopyHistory:
    """Incremental reconstruction of Canopy's history-derived Nexus-v3 fields."""

    replay: Game
    tested_non_knight: list[int]
    record_count: int = 0

    @classmethod
    def from_game(cls, game: Game) -> "_CanopyHistory":
        # Upstream Catanatron uses module-global randomness. Replaying recorded
        # outcomes must never perturb the authoritative live games' dice RNG.
        random_state = random.getstate()
        try:
            players = [SimplePlayer(color) for color in game.state.colors]
            replay = Game(
                players=players,
                seed=game.seed,
                discard_limit=game.state.discard_limit,
                friendly_robber=game.friendly_robber,
                vps_to_win=game.vps_to_win,
                catan_map=game.state.board.map,
            )
            force_player_order(replay, players)
        finally:
            random.setstate(random_state)
        history = cls(replay=replay, tested_non_knight=[0, 0])
        history.sync(game)
        return history

    def sync(self, game: Game) -> tuple[int, int]:
        records = game.state.action_records
        if len(records) < self.record_count:
            raise ValueError("Catanatron action history moved backwards")
        random_state = random.getstate()
        try:
            for record in records[self.record_count :]:
                _update_tested_non_knight_before_action(
                    self.replay,
                    record.action,
                    self.tested_non_knight,
                )
                # Recorded roll actions carry the realized dice in ``value``
                # while the live legal intent is ROLL(None), so Catanatron's
                # replay API intentionally bypasses intent-level validation.
                self.replay.execute(
                    record.action,
                    validate_action=False,
                    action_record=record,
                )
                _update_tested_non_knight_after_action(
                    self.replay,
                    record.action,
                    self.tested_non_knight,
                )
            self.record_count = len(records)
        finally:
            random.setstate(random_state)
        if self.replay.state.current_color() != game.state.current_color():
            raise ValueError("Catanatron history replay current player diverged")
        if self.replay.state.board.robber_coordinate != game.state.board.robber_coordinate:
            raise ValueError("Catanatron history replay robber diverged")
        return tuple(self.tested_non_knight)


def canopy_tested_non_knight(game: Game) -> tuple[int, int]:
    """Derive Canopy's Nexus-v3 behavioral history signal from action records."""

    return _CanopyHistory.from_game(game).sync(game)


def _phase_snapshot(game: Game) -> dict[str, Any]:
    state = game.state
    if state.is_initial_build_phase:
        if state.current_prompt == ActionPrompt.BUILD_INITIAL_SETTLEMENT:
            return {"kind": "place_settlement"}
        if state.current_prompt == ActionPrompt.BUILD_INITIAL_ROAD:
            return {"kind": "place_road"}
    if state.is_discarding or state.current_prompt == ActionPrompt.DISCARD:
        return {
            "kind": "discard",
            "remaining": int(state.discard_counts[state.current_player_index]),
            "roller": int(state.current_turn_index),
        }
    # Catanatron leaves ``is_moving_knight`` set after the robber has moved;
    # current_prompt is the authoritative decision phase.
    if state.current_prompt == ActionPrompt.MOVE_ROBBER:
        return {"kind": "move_robber"}
    if state.is_road_building:
        return {"kind": "road_building", "roads_left": int(state.free_roads_available)}
    if state.current_prompt == ActionPrompt.PLAY_TURN:
        if player_has_rolled(state, state.current_color()):
            return {"kind": "main"}
        return {"kind": "pre_roll"}
    raise ValueError(f"Catanatron phase has no Canopy equivalent: {state.current_prompt}")


def _last_setup_node(game: Game, mapping: CanopyTopologyMapping) -> int | None:
    if not game.state.is_initial_build_phase:
        return None
    for record in reversed(game.state.action_records):
        if record.action.action_type == ActionType.BUILD_SETTLEMENT:
            return mapping.node_to_canopy[int(record.action.value)]
    return None


def game_to_canopy_snapshot(
    game: Game,
    *,
    tree_actions: list[int] | None = None,
    tested_non_knight: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Serialize one authoritative Catanatron decision state for Canopy."""

    state = game.state
    if len(state.colors) != 2:
        raise ValueError("The released Canopy Catan model supports exactly two players")
    mapping = CanopyTopologyMapping.from_game(game)
    catan_map = state.board.map

    terrains: list[str] = []
    numbers: list[int | None] = []
    for coordinate in mapping.tiles:
        tile = catan_map.land_tiles[coordinate]
        terrains.append(tile.resource if tile.resource is not None else "DESERT")
        numbers.append(tile.number)

    ports: list[str | None] = []
    for q, r in CANOPY_PORT_HEXES:
        tile = catan_map.tiles[_cube(q, r)]
        if not isinstance(tile, Port):
            raise ValueError(f"Expected Catanatron port at axial coordinate {(q, r)}")
        ports.append(tile.resource)

    tile_nodes = [
        [
            mapping.node_to_canopy[int(catan_map.land_tiles[coordinate].nodes[corner])]
            for corner in CANOPY_CORNERS
        ]
        for coordinate in mapping.tiles
    ]
    tile_edges = [
        [
            mapping.edge_to_canopy[tuple(sorted(catan_map.land_tiles[coordinate].edges[edge_ref]))]
            for edge_ref in CANOPY_EDGES
        ]
        for coordinate in mapping.tiles
    ]

    settlements: list[list[int]] = [[], []]
    cities: list[list[int]] = [[], []]
    roads: list[list[int]] = [[], []]
    for index, color in enumerate(state.colors):
        settlements[index] = [
            mapping.node_to_canopy[int(node)]
            for node in state.buildings_by_color[color][SETTLEMENT]
        ]
        cities[index] = [
            mapping.node_to_canopy[int(node)] for node in state.buildings_by_color[color][CITY]
        ]
        roads[index] = [
            mapping.edge_to_canopy[tuple(sorted(edge))]
            for edge in state.buildings_by_color[color][ROAD]
        ]

    longest_road = None
    if state.board.road_color is not None:
        winner = state.color_to_index[state.board.road_color]
        winner_key = f"P{winner}"
        longest_road = {
            "player": winner,
            "length": int(state.player_state[f"{winner_key}_LONGEST_ROAD_LENGTH"]),
        }
    largest_army = None
    for index in range(2):
        if state.player_state[f"P{index}_HAS_ARMY"]:
            largest_army = {
                "player": index,
                "length": int(state.player_state[f"P{index}_PLAYED_KNIGHT"]),
            }
            break

    setup_count = 4
    if state.is_initial_build_phase:
        setup_count = sum(
            record.action.action_type == ActionType.BUILD_SETTLEMENT
            for record in state.action_records
        )

    current_color = state.current_color()
    current_index = state.color_to_index[current_color]
    roller_color = state.colors[state.current_turn_index]
    current_settlements = settlements[current_index]
    tested = canopy_tested_non_knight(game) if tested_non_knight is None else tested_non_knight
    if len(tested) != 2 or any(value < 0 for value in tested):
        raise ValueError("tested_non_knight must contain two non-negative counts")
    return {
        "game_id": game.id,
        "tree_actions": [] if tree_actions is None else tree_actions,
        "reuse_tree": tree_actions is not None,
        "terrains": terrains,
        "numbers": numbers,
        "ports": ports,
        "tile_nodes": tile_nodes,
        "tile_edges": tile_edges,
        "players": [
            _player_snapshot(game, color, int(tested[index]))
            for index, color in enumerate(state.colors)
        ],
        "settlements": settlements,
        "cities": cities,
        "roads": roads,
        "bank": [int(value) for value in state.resource_freqdeck],
        "dev_deck_total": len(state.development_listdeck),
        "robber": CANOPY_LAND_HEXES.index(
            (state.board.robber_coordinate[0], state.board.robber_coordinate[2])
        ),
        "current_player": current_index,
        "phase": _phase_snapshot(game),
        "turn_number": int(state.num_turns),
        # During an opponent discard, current_color is not the roller.  Canopy
        # uses this flag to return to pre-roll after a knight robber move, so it
        # must follow the turn owner rather than the temporary decision actor.
        "pre_roll": not player_has_rolled(state, roller_color),
        "setup_count": setup_count,
        "last_setup_node": _last_setup_node(game, mapping),
        "longest_road": longest_road,
        "largest_army": largest_army,
        "vp_limit": int(game.vps_to_win),
        "discard_threshold": int(state.discard_limit),
        # Treat all current settlements as pre-existing at an imported root.
        # Canopy canonicalizes subsequent simulated actions from this root.
        "settlements_at_turn_start": current_settlements,
        "roads_placed_at_turn_start": 15
        - int(state.player_state[f"P{current_index}_ROADS_AVAILABLE"]),
    }


def _find_action(
    playable_actions: Sequence[Action],
    action_type: ActionType,
    predicate=lambda _action: True,
) -> Action:
    matches = [
        action
        for action in playable_actions
        if action.action_type == action_type and predicate(action)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Canopy action maps to {len(matches)} legal Catanatron actions "
            f"of type {action_type}: {matches}"
        )
    return matches[0]


def canopy_action_to_catanatron(
    action_id: int,
    game: Game,
    playable_actions: Sequence[Action],
) -> Action:
    """Map one of Canopy's 249 fixed actions to an actually legal action."""

    mapping = CanopyTopologyMapping.from_game(game)
    if 0 <= action_id < 54:
        node = mapping.nodes[action_id]
        return _find_action(
            playable_actions,
            ActionType.BUILD_SETTLEMENT,
            lambda action: action.value == node,
        )
    if 54 <= action_id < 126:
        edge = mapping.edges[action_id - 54]
        return _find_action(
            playable_actions,
            ActionType.BUILD_ROAD,
            lambda action: tuple(sorted(action.value)) == edge,
        )
    if 126 <= action_id < 180:
        node = mapping.nodes[action_id - 126]
        return _find_action(
            playable_actions,
            ActionType.BUILD_CITY,
            lambda action: action.value == node,
        )

    fixed = {
        180: ActionType.ROLL,
        181: ActionType.END_TURN,
        182: ActionType.BUY_DEVELOPMENT_CARD,
        183: ActionType.PLAY_KNIGHT_CARD,
        184: ActionType.PLAY_ROAD_BUILDING,
    }
    if action_id in fixed:
        return _find_action(playable_actions, fixed[action_id])
    if 185 <= action_id < 200:
        pairs = [(first, second) for i, first in enumerate(RESOURCES) for second in RESOURCES[i:]]
        resources = pairs[action_id - 185]
        return _find_action(
            playable_actions,
            ActionType.PLAY_YEAR_OF_PLENTY,
            lambda action: tuple(action.value) == resources,
        )
    if 200 <= action_id < 205:
        selected_resource = RESOURCES[action_id - 200]
        return _find_action(
            playable_actions,
            ActionType.PLAY_MONOPOLY,
            lambda action: action.value == selected_resource,
        )
    if 205 <= action_id < 224:
        coordinate = mapping.tiles[action_id - 205]
        matches = [
            action
            for action in playable_actions
            if action.action_type == ActionType.MOVE_ROBBER and action.value[0] == coordinate
        ]
        if not matches:
            raise ValueError(f"Canopy robber action has no legal Catanatron action: {coordinate}")
        opponent = next(color for color in game.state.colors if color != game.state.current_color())
        for action in matches:
            if action.value[1] == opponent:
                return action
        if len(matches) == 1:
            return matches[0]
        raise ValueError(f"Ambiguous Catanatron robber actions for {coordinate}: {matches}")
    if 224 <= action_id < 229:
        selected_resource = RESOURCES[action_id - 224]
        return _find_action(
            playable_actions,
            ActionType.DISCARD_RESOURCE,
            lambda action: action.value == selected_resource,
        )
    if 229 <= action_id < 249:
        pair_index = action_id - 229
        give_index = pair_index // 4
        receive_index = pair_index % 4
        if receive_index >= give_index:
            receive_index += 1
        give = RESOURCES[give_index]
        receive = RESOURCES[receive_index]
        return _find_action(
            playable_actions,
            ActionType.MARITIME_TRADE,
            lambda action: action.value[0] == give and action.value[-1] == receive,
        )
    raise ValueError(f"Invalid Canopy action id {action_id}")


def catanatron_action_to_canopy(
    action: Action,
    mapping: CanopyTopologyMapping,
) -> int:
    """Map a Catanatron action into Canopy's fixed 249-action space."""

    action_type = action.action_type
    if action_type == ActionType.BUILD_SETTLEMENT:
        return mapping.node_to_canopy[int(action.value)]
    if action_type == ActionType.BUILD_ROAD:
        return 54 + mapping.edge_to_canopy[tuple(sorted(action.value))]
    if action_type == ActionType.BUILD_CITY:
        return 126 + mapping.node_to_canopy[int(action.value)]
    fixed = {
        ActionType.ROLL: 180,
        ActionType.END_TURN: 181,
        ActionType.BUY_DEVELOPMENT_CARD: 182,
        ActionType.PLAY_KNIGHT_CARD: 183,
        ActionType.PLAY_ROAD_BUILDING: 184,
    }
    if action_type in fixed:
        return fixed[action_type]
    if action_type == ActionType.PLAY_YEAR_OF_PLENTY:
        resources = tuple(action.value)
        if len(resources) != 2:
            raise ValueError("Canopy cannot represent a one-card Year of Plenty action")
        first_index, second_index = sorted(RESOURCES.index(value) for value in resources)
        offsets = (0, 5, 9, 12, 14)
        return 185 + offsets[first_index] + second_index - first_index
    if action_type == ActionType.PLAY_MONOPOLY:
        return 200 + RESOURCES.index(action.value)
    if action_type == ActionType.MOVE_ROBBER:
        return 205 + mapping.tile_to_canopy[tuple(action.value[0])]
    if action_type == ActionType.DISCARD_RESOURCE:
        return 224 + RESOURCES.index(action.value)
    if action_type == ActionType.MARITIME_TRADE:
        give = RESOURCES.index(action.value[0])
        receive = RESOURCES.index(action.value[-1])
        receive_index = receive if receive < give else receive - 1
        return 229 + give * 4 + receive_index
    raise ValueError(f"Canopy cannot represent Catanatron action {action}")


def _action_record_to_canopy(record: Any, mapping: CanopyTopologyMapping) -> list[int]:
    actions = [catanatron_action_to_canopy(record.action, mapping)]
    if record.action.action_type == ActionType.ROLL:
        actions.append(sum(record.result) - 2)
    elif record.action.action_type == ActionType.BUY_DEVELOPMENT_CARD:
        actions.append(CANOPY_DEV_ORDER.index(record.result))
    elif record.action.action_type == ActionType.MOVE_ROBBER and record.result is not None:
        actions.append(RESOURCES.index(record.result))
    return actions


class CanopyBridgeProcess:
    """Persistent released-Canopy inference/search subprocess."""

    def __init__(
        self,
        binary: str | Path,
        checkpoint: str | Path,
        *,
        simulations: int = 1600,
        seed: int = 0,
        stderr: IO[str] | int | None = None,
    ) -> None:
        if simulations < 1:
            raise ValueError("simulations must be positive")
        self._next_id = 0
        self._record_counts: dict[str, int] = {}
        self._histories: dict[str, _CanopyHistory] = {}
        self._process = subprocess.Popen(
            [
                str(binary),
                "catanatron-bridge",
                "--checkpoint",
                str(checkpoint),
                "--simulations",
                str(simulations),
                "--seed",
                str(seed),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            text=True,
            bufsize=1,
        )

    def close(self) -> None:
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()

    def __enter__(self) -> "CanopyBridgeProcess":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def decide_many(
        self,
        games_and_actions: Sequence[tuple[Game, Sequence[Action]]],
    ) -> list[Action]:
        if not games_and_actions:
            return []
        if self._process.poll() is not None:
            raise RuntimeError(f"Canopy bridge exited with code {self._process.returncode}")
        assert self._process.stdin is not None
        assert self._process.stdout is not None
        request_id = self._next_id
        self._next_id += 1
        snapshots: list[dict[str, Any]] = []
        next_record_counts: dict[str, int] = {}
        active_ids = {game.id for game, _ in games_and_actions}
        self._record_counts = {
            game_id: count
            for game_id, count in self._record_counts.items()
            if game_id in active_ids
        }
        self._histories = {
            game_id: history
            for game_id, history in self._histories.items()
            if game_id in active_ids
        }
        for game, _ in games_and_actions:
            records = game.state.action_records
            previous_count = self._record_counts.get(game.id, len(records))
            mapping = CanopyTopologyMapping.from_game(game)
            try:
                tree_actions = [
                    canopy_action
                    for record in records[previous_count:]
                    for canopy_action in _action_record_to_canopy(record, mapping)
                ]
            except (KeyError, ValueError):
                # An action outside Canopy's fixed space invalidates reuse, but
                # the exact imported root remains fully evaluable.
                tree_actions = None
            history = self._histories.get(game.id)
            if history is None:
                history = _CanopyHistory.from_game(game)
                self._histories[game.id] = history
            tested_non_knight = history.sync(game)
            snapshots.append(
                game_to_canopy_snapshot(
                    game,
                    tree_actions=tree_actions,
                    tested_non_knight=tested_non_knight,
                )
            )
            next_record_counts[game.id] = len(records)
        payload = {
            "id": request_id,
            "states": snapshots,
        }
        self._process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self._process.stdin.flush()

        line = self._process.stdout.readline()
        if not line:
            raise RuntimeError(f"Canopy bridge closed stdout (exit={self._process.poll()})")
        response = json.loads(line)
        if response.get("id") != request_id:
            raise RuntimeError(f"Canopy bridge response id {response.get('id')} != {request_id}")
        if "error" in response:
            raise RuntimeError(f"Canopy bridge rejected state batch: {response['error']}")
        action_ids = response["actions"]
        if len(action_ids) != len(games_and_actions):
            raise RuntimeError(
                f"Canopy bridge returned {len(action_ids)} actions for "
                f"{len(games_and_actions)} games"
            )
        self._record_counts.update(next_record_counts)
        return [
            canopy_action_to_catanatron(action_id, game, playable_actions)
            for action_id, (game, playable_actions) in zip(action_ids, games_and_actions)
        ]


class CanopyCatanatronPlayer(Player):
    """Synchronous one-game Catanatron Player backed by released Canopy."""

    def __init__(self, color: Color, bridge: CanopyBridgeProcess) -> None:
        super().__init__(color)
        self.bridge = bridge

    def decide(self, game: Game, playable_actions: Sequence[Action]) -> Action:
        return self.bridge.decide_many([(game, playable_actions)])[0]


__all__ = [
    "CANOPY_LAND_HEXES",
    "CanopyBridgeProcess",
    "CanopyCatanatronPlayer",
    "CanopyTopologyMapping",
    "canopy_action_to_catanatron",
    "canopy_tested_non_knight",
    "catanatron_action_to_canopy",
    "game_to_canopy_snapshot",
]
