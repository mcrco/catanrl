from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

MapType = Literal["BASE", "MINI", "TOURNAMENT"]
NumberPlacement = Literal["official_spiral", "random"]


class _PlayerState(ctypes.Structure):
    _fields_ = [
        ("victory_points", ctypes.c_int32),
        ("actual_victory_points", ctypes.c_int32),
        ("roads_available", ctypes.c_int32),
        ("settlements_available", ctypes.c_int32),
        ("cities_available", ctypes.c_int32),
        ("has_road", ctypes.c_int32),
        ("has_army", ctypes.c_int32),
        ("has_rolled", ctypes.c_int32),
        ("has_played_development_card_in_turn", ctypes.c_int32),
        ("longest_road_length", ctypes.c_int32),
        ("resources", ctypes.c_int32 * 5),
        ("development_cards", ctypes.c_int32 * 5),
        ("played_development_cards", ctypes.c_int32 * 5),
        ("development_card_owned_at_start", ctypes.c_int32 * 4),
        ("turns_since_last_knight", ctypes.c_int32),
        ("turns_since_last_development_card_bought", ctypes.c_int32),
    ]


class _Building(ctypes.Structure):
    _fields_ = [
        ("node", ctypes.c_int32),
        ("color", ctypes.c_int32),
        ("building", ctypes.c_int32),
    ]


class _Road(ctypes.Structure):
    _fields_ = [
        ("a", ctypes.c_int32),
        ("b", ctypes.c_int32),
        ("color", ctypes.c_int32),
    ]


class _Tile(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_int32),
        ("y", ctypes.c_int32),
        ("z", ctypes.c_int32),
        ("id", ctypes.c_int32),
        ("kind", ctypes.c_int32),
        ("resource", ctypes.c_int32),
        ("number", ctypes.c_int32),
        ("port_direction", ctypes.c_int32),
        ("nodes", ctypes.c_int32 * 6),
    ]


class _NodePosition(ctypes.Structure):
    _fields_ = [
        ("node", ctypes.c_int32),
        ("x", ctypes.c_int32),
        ("y", ctypes.c_int32),
    ]


class _EdgePosition(ctypes.Structure):
    _fields_ = [
        ("a", ctypes.c_int32),
        ("b", ctypes.c_int32),
        ("x", ctypes.c_int32),
        ("y", ctypes.c_int32),
    ]


class _TilePosition(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_int32),
        ("y", ctypes.c_int32),
        ("z", ctypes.c_int32),
        ("board_x", ctypes.c_int32),
        ("board_y", ctypes.c_int32),
    ]


class _SearchMetrics(ctypes.Structure):
    _fields_ = [
        ("simulations", ctypes.c_uint64),
        ("principal_variation_depth", ctypes.c_uint32),
        ("maximum_depth", ctypes.c_uint32),
        ("mean_depth", ctypes.c_double),
        ("root_value", ctypes.c_double),
        ("retained_root_visits", ctypes.c_uint32),
        ("pruned_actions", ctypes.c_uint64),
        ("coalesced_outcomes", ctypes.c_uint64),
        ("tree_reused", ctypes.c_int32),
    ]


@dataclass(frozen=True)
class NativePlayerState:
    victory_points: int
    actual_victory_points: int
    roads_available: int
    settlements_available: int
    cities_available: int
    has_road: bool
    has_army: bool
    has_rolled: bool
    has_played_development_card_in_turn: bool
    longest_road_length: int
    resources: tuple[int, ...]
    development_cards: tuple[int, ...]
    played_development_cards: tuple[int, ...]
    development_card_owned_at_start: tuple[bool, ...]
    turns_since_last_knight: int
    turns_since_last_development_card_bought: int

    @classmethod
    def from_c(cls, state: _PlayerState) -> NativePlayerState:
        return cls(
            victory_points=int(state.victory_points),
            actual_victory_points=int(state.actual_victory_points),
            roads_available=int(state.roads_available),
            settlements_available=int(state.settlements_available),
            cities_available=int(state.cities_available),
            has_road=bool(state.has_road),
            has_army=bool(state.has_army),
            has_rolled=bool(state.has_rolled),
            has_played_development_card_in_turn=bool(
                state.has_played_development_card_in_turn
            ),
            longest_road_length=int(state.longest_road_length),
            resources=tuple(int(value) for value in state.resources),
            development_cards=tuple(int(value) for value in state.development_cards),
            played_development_cards=tuple(
                int(value) for value in state.played_development_cards
            ),
            development_card_owned_at_start=tuple(
                bool(value) for value in state.development_card_owned_at_start
            ),
            turns_since_last_knight=int(state.turns_since_last_knight),
            turns_since_last_development_card_bought=int(
                state.turns_since_last_development_card_bought
            ),
        )


def find_cudanatron_library() -> Path:
    override = os.environ.get("CUDANATRON_LIBRARY")
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"CUDANATRON_LIBRARY does not exist: {path}")
        return path

    repository = Path(__file__).resolve().parents[4]
    candidates = [
        repository / "cudanatron" / "build" / "libcudanatron.so",
        repository / "cudanatron" / "build" / "libcudanatron.dylib",
        repository / "cudanatron" / "build" / "cudanatron.dll",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "cudanatron shared library was not found. Build it with "
        "`cmake -S cudanatron -B cudanatron/build && "
        "cmake --build cudanatron/build` or set CUDANATRON_LIBRARY."
    )


def _load_library(path: Path | None = None) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path or find_cudanatron_library()))
    handle = ctypes.c_void_p
    library.cudanatron_last_error.restype = ctypes.c_char_p
    library.cudanatron_game_create_seeded_with_number_placement.argtypes = [
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    library.cudanatron_game_create_seeded_with_number_placement.restype = handle
    library.cudanatron_game_destroy.argtypes = [handle]
    library.cudanatron_game_reset_seeded.argtypes = [
        handle,
        ctypes.c_uint64,
        ctypes.c_uint64,
    ]
    library.cudanatron_game_action_space_size.argtypes = [handle]
    library.cudanatron_game_action_space_size.restype = ctypes.c_int32
    library.cudanatron_game_valid_action_mask.argtypes = [
        handle,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
    ]
    library.cudanatron_game_step.argtypes = [handle, ctypes.c_int32]
    library.cudanatron_game_step_replay.argtypes = [
        handle,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    library.cudanatron_game_num_players.argtypes = [handle]
    library.cudanatron_game_num_players.restype = ctypes.c_int32
    library.cudanatron_game_current_player.argtypes = [handle]
    library.cudanatron_game_current_player.restype = ctypes.c_int32
    library.cudanatron_game_current_prompt.argtypes = [handle]
    library.cudanatron_game_current_prompt.restype = ctypes.c_int32
    library.cudanatron_game_num_turns.argtypes = [handle]
    library.cudanatron_game_num_turns.restype = ctypes.c_int32
    library.cudanatron_game_winner.argtypes = [handle]
    library.cudanatron_game_winner.restype = ctypes.c_int32
    library.cudanatron_game_development_cards_remaining.argtypes = [handle]
    library.cudanatron_game_development_cards_remaining.restype = ctypes.c_int32
    library.cudanatron_game_resource_bank.argtypes = [
        handle,
        ctypes.POINTER(ctypes.c_int32),
    ]
    library.cudanatron_game_flags.argtypes = [handle, ctypes.POINTER(ctypes.c_int32)]
    library.cudanatron_game_robber_coordinate.argtypes = [
        handle,
        ctypes.POINTER(ctypes.c_int32),
    ]
    library.cudanatron_game_player_state.argtypes = [
        handle,
        ctypes.c_int32,
        ctypes.POINTER(_PlayerState),
    ]
    library.cudanatron_game_buildings.argtypes = [
        handle,
        ctypes.POINTER(_Building),
        ctypes.c_size_t,
    ]
    library.cudanatron_game_buildings.restype = ctypes.c_int32
    library.cudanatron_game_roads.argtypes = [
        handle,
        ctypes.POINTER(_Road),
        ctypes.c_size_t,
    ]
    library.cudanatron_game_roads.restype = ctypes.c_int32
    library.cudanatron_game_tiles.argtypes = [
        handle,
        ctypes.POINTER(_Tile),
        ctypes.c_size_t,
    ]
    library.cudanatron_game_tiles.restype = ctypes.c_int32
    library.cudanatron_game_action_key.argtypes = [
        handle,
        ctypes.c_int32,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    library.cudanatron_game_set_observation_layout.argtypes = [
        handle,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(_NodePosition),
        ctypes.c_size_t,
        ctypes.POINTER(_EdgePosition),
        ctypes.c_size_t,
        ctypes.POINTER(_TilePosition),
        ctypes.c_size_t,
    ]
    library.cudanatron_game_observation_size.argtypes = [handle]
    library.cudanatron_game_observation_size.restype = ctypes.c_int32
    library.cudanatron_game_write_observation.argtypes = [
        handle,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
    ]
    library.cudanatron_search_pool_create.argtypes = [
        ctypes.POINTER(handle),
        ctypes.c_size_t,
        ctypes.c_double,
        ctypes.c_uint64,
        ctypes.c_int32,
    ]
    library.cudanatron_search_pool_create.restype = handle
    library.cudanatron_search_pool_destroy.argtypes = [handle]
    library.cudanatron_search_pool_size.argtypes = [handle]
    library.cudanatron_search_pool_size.restype = ctypes.c_int32
    library.cudanatron_search_pool_observation_size.argtypes = [handle]
    library.cudanatron_search_pool_observation_size.restype = ctypes.c_int32
    library.cudanatron_search_pool_initialize_roots.argtypes = [
        handle,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    library.cudanatron_search_pool_set_root_wdls.argtypes = [
        handle,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
    ]
    library.cudanatron_search_pool_enable_completed_q.argtypes = [
        handle,
        ctypes.c_double,
        ctypes.c_double,
    ]
    library.cudanatron_search_pool_add_dirichlet_noise.argtypes = [
        handle,
        ctypes.c_double,
        ctypes.c_double,
    ]
    library.cudanatron_search_pool_add_simulations_all.argtypes = [handle, ctypes.c_int32]
    library.cudanatron_search_pool_remaining_simulations.argtypes = [handle]
    library.cudanatron_search_pool_remaining_simulations.restype = ctypes.c_int32
    library.cudanatron_search_pool_select_leaves.argtypes = [
        handle,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
    ]
    library.cudanatron_search_pool_select_leaves.restype = ctypes.c_int32
    library.cudanatron_search_pool_evaluate_leaves.argtypes = [
        handle,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
    ]
    library.cudanatron_search_pool_root_visits.argtypes = [
        handle,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_size_t,
    ]
    library.cudanatron_search_pool_root_wdl.argtypes = [
        handle,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
    ]
    library.cudanatron_search_pool_metrics.argtypes = [
        handle,
        ctypes.c_int32,
        ctypes.POINTER(_SearchMetrics),
    ]
    library.cudanatron_search_pool_advance.argtypes = [
        handle,
        ctypes.c_int32,
        ctypes.c_size_t,
    ]
    library.cudanatron_search_pool_advance.restype = ctypes.c_int32
    library.cudanatron_search_pool_advance_to_game.argtypes = [
        handle,
        ctypes.c_int32,
        ctypes.c_size_t,
        handle,
    ]
    library.cudanatron_search_pool_advance_to_game.restype = ctypes.c_int32
    return library


def observation_layout_arrays():
    from catanatron.gym.board_tensor_features import (
        HEIGHT,
        WIDTH,
        get_node_and_edge_maps,
        get_tile_coordinate_map,
    )

    node_map, edge_map = get_node_and_edge_maps()
    nodes = (_NodePosition * len(node_map))(
        *(_NodePosition(node, x, y) for node, (x, y) in node_map.items())
    )
    unique_edges: dict[tuple[int, int], tuple[int, int]] = {}
    for (a, b), position in edge_map.items():
        unique_edges.setdefault(tuple(sorted((a, b))), position)
    edges = (_EdgePosition * len(unique_edges))(
        *(_EdgePosition(a, b, x, y) for (a, b), (x, y) in unique_edges.items())
    )
    tile_map = get_tile_coordinate_map()
    tiles = (_TilePosition * len(tile_map))(
        *(
            _TilePosition(int(x), int(y), int(z), board_x, board_y)
            for (x, y, z), (board_y, board_x) in tile_map.items()
        )
    )
    return WIDTH, HEIGHT, nodes, edges, tiles


class NativeGame:
    """Owned Python handle for one cudanatron game."""

    _MAP_TYPES = {"BASE": 0, "MINI": 1, "TOURNAMENT": 2}
    _NUMBER_PLACEMENTS = {"official_spiral": 0, "random": 1}

    def __init__(
        self,
        num_players: int = 2,
        map_type: MapType = "BASE",
        seed: int = 0,
        map_seed: int | None = None,
        number_placement: NumberPlacement = "random",
        discard_limit: int = 7,
        friendly_robber: bool = False,
        vps_to_win: int = 10,
        library_path: Path | None = None,
    ) -> None:
        self._library = _load_library(library_path)
        try:
            native_map_type = self._MAP_TYPES[map_type]
        except KeyError as exc:
            raise ValueError(f"Unknown map type: {map_type}") from exc
        try:
            native_number_placement = self._NUMBER_PLACEMENTS[number_placement]
        except KeyError as exc:
            raise ValueError(f"Unknown number placement: {number_placement}") from exc
        self._handle = self._library.cudanatron_game_create_seeded_with_number_placement(
            num_players,
            native_map_type,
            seed if map_seed is None else map_seed,
            seed,
            discard_limit,
            int(friendly_robber),
            vps_to_win,
            native_number_placement,
        )
        if not self._handle:
            self._raise_last_error()
        self.num_players = int(self._library.cudanatron_game_num_players(self._handle))
        self.action_space_size = int(
            self._library.cudanatron_game_action_space_size(self._handle)
        )
        self._layout_ready = False

    def _raise_last_error(self) -> None:
        message = self._library.cudanatron_last_error()
        raise RuntimeError(message.decode() if message else "unknown cudanatron error")

    def _check(self, result: int) -> None:
        if result != 0:
            self._raise_last_error()

    @property
    def current_player(self) -> int:
        return int(self._library.cudanatron_game_current_player(self._handle))

    @property
    def current_prompt(self) -> int:
        return int(self._library.cudanatron_game_current_prompt(self._handle))

    @property
    def num_turns(self) -> int:
        return int(self._library.cudanatron_game_num_turns(self._handle))

    @property
    def winner(self) -> int | None:
        value = int(self._library.cudanatron_game_winner(self._handle))
        return None if value == -1 else value

    @property
    def development_cards_remaining(self) -> int:
        return int(self._library.cudanatron_game_development_cards_remaining(self._handle))

    @property
    def resource_bank(self) -> tuple[int, ...]:
        values = (ctypes.c_int32 * 5)()
        self._check(self._library.cudanatron_game_resource_bank(self._handle, values))
        return tuple(int(value) for value in values)

    @property
    def flags(self) -> tuple[bool, bool, bool, bool, int, int, int]:
        values = (ctypes.c_int32 * 7)()
        self._check(self._library.cudanatron_game_flags(self._handle, values))
        return (
            bool(values[0]),
            bool(values[1]),
            bool(values[2]),
            bool(values[3]),
            int(values[4]),
            int(values[5]),
            int(values[6]),
        )

    @property
    def robber_coordinate(self) -> tuple[int, int, int]:
        values = (ctypes.c_int32 * 3)()
        self._check(self._library.cudanatron_game_robber_coordinate(self._handle, values))
        return tuple(int(value) for value in values)  # type: ignore[return-value]

    def reset(self, seed: int, *, map_seed: int | None = None) -> None:
        self._check(
            self._library.cudanatron_game_reset_seeded(
                self._handle,
                seed if map_seed is None else map_seed,
                seed,
            )
        )

    def valid_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_space_size, dtype=np.uint8)
        self._check(
            self._library.cudanatron_game_valid_action_mask(
                self._handle,
                mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                mask.size,
            )
        )
        return mask.astype(np.bool_, copy=False)

    def step(
        self,
        action: int,
        *,
        dice: tuple[int, int] | None = None,
        development_card: int | None = None,
        stolen_resource: int | None = None,
    ) -> None:
        if dice is None and development_card is None and stolen_resource is None:
            self._check(self._library.cudanatron_game_step(self._handle, int(action)))
            return
        die_one, die_two = dice or (-1, -1)
        self._check(
            self._library.cudanatron_game_step_replay(
                self._handle,
                int(action),
                die_one,
                die_two,
                -1 if development_card is None else development_card,
                -1 if stolen_resource is None else stolen_resource,
            )
        )

    def player_state(self, player: int) -> NativePlayerState:
        state = _PlayerState()
        self._check(
            self._library.cudanatron_game_player_state(
                self._handle, player, ctypes.byref(state)
            )
        )
        return NativePlayerState.from_c(state)

    def buildings(self) -> tuple[tuple[int, int, int], ...]:
        count = int(self._library.cudanatron_game_buildings(self._handle, None, 0))
        if count < 0:
            self._raise_last_error()
        if count == 0:
            return ()
        values = (_Building * count)()
        result = int(self._library.cudanatron_game_buildings(self._handle, values, count))
        if result < 0:
            self._raise_last_error()
        return tuple((int(item.node), int(item.color), int(item.building)) for item in values)

    def roads(self) -> tuple[tuple[int, int, int], ...]:
        count = int(self._library.cudanatron_game_roads(self._handle, None, 0))
        if count < 0:
            self._raise_last_error()
        if count == 0:
            return ()
        values = (_Road * count)()
        result = int(self._library.cudanatron_game_roads(self._handle, values, count))
        if result < 0:
            self._raise_last_error()
        return tuple((int(item.a), int(item.b), int(item.color)) for item in values)

    def tiles(
        self,
    ) -> tuple[tuple[int, int, int, int, int, int, int, int, tuple[int, ...]], ...]:
        count = int(self._library.cudanatron_game_tiles(self._handle, None, 0))
        if count < 0:
            self._raise_last_error()
        values = (_Tile * count)()
        result = int(self._library.cudanatron_game_tiles(self._handle, values, count))
        if result < 0:
            self._raise_last_error()
        return tuple(
            (
                int(item.x),
                int(item.y),
                int(item.z),
                int(item.id),
                int(item.kind),
                int(item.resource),
                int(item.number),
                int(item.port_direction),
                tuple(int(node) for node in item.nodes),
            )
            for item in values
        )

    def action_key(self, index: int) -> str:
        buffer = ctypes.create_string_buffer(256)
        self._check(
            self._library.cudanatron_game_action_key(
                self._handle, int(index), buffer, len(buffer)
            )
        )
        return buffer.value.decode()

    def ensure_observation_layout(self) -> None:
        if self._layout_ready:
            return
        width, height, nodes, edges, tiles = observation_layout_arrays()
        self._check(
            self._library.cudanatron_game_set_observation_layout(
                self._handle,
                width,
                height,
                nodes,
                len(nodes),
                edges,
                len(edges),
                tiles,
                len(tiles),
            )
        )
        self._layout_ready = True

    def observation_size(self) -> int:
        self.ensure_observation_layout()
        size = int(self._library.cudanatron_game_observation_size(self._handle))
        if size < 0:
            self._raise_last_error()
        return size

    def observation(self, base_player: int | None = None) -> np.ndarray:
        self.ensure_observation_layout()
        player = self.current_player if base_player is None else int(base_player)
        output = np.zeros(self.observation_size(), dtype=np.float32)
        self._check(
            self._library.cudanatron_game_write_observation(
                self._handle,
                player,
                output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                output.size,
            )
        )
        return output

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._library.cudanatron_game_destroy(self._handle)
            self._handle = None

    def __enter__(self) -> NativeGame:
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


@dataclass(frozen=True)
class NativeSearchMetrics:
    simulations: int
    principal_variation_depth: int
    maximum_depth: int
    mean_depth: float
    root_value: float
    retained_root_visits: int
    pruned_actions: int
    coalesced_outcomes: int
    tree_reused: bool


class NativeSearchPool:
    """Coordinator over independent native MCTS searches.

    Python only supplies batched neural evaluations. Terminal backups stay
    inside C++ and do not occupy leaf-batch slots.
    """

    def __init__(
        self,
        games: list[NativeGame],
        *,
        c_puct: float = 1.5,
        seed: int = 0,
        canonical_pruning: bool = False,
    ) -> None:
        if not games:
            raise ValueError("search pool requires at least one game")
        for game in games:
            game.ensure_observation_layout()
        self._library = games[0]._library
        self._games = games
        handles = (ctypes.c_void_p * len(games))(*[game._handle for game in games])
        self._handle = self._library.cudanatron_search_pool_create(
            handles,
            len(games),
            float(c_puct),
            int(seed),
            int(canonical_pruning),
        )
        if not self._handle:
            message = self._library.cudanatron_last_error()
            raise RuntimeError(message.decode() if message else "unknown cudanatron error")
        self.size = int(self._library.cudanatron_search_pool_size(self._handle))
        self.observation_size = int(
            self._library.cudanatron_search_pool_observation_size(self._handle)
        )
        self.action_space_size = games[0].action_space_size

    def _raise_last_error(self) -> None:
        message = self._library.cudanatron_last_error()
        raise RuntimeError(message.decode() if message else "unknown cudanatron error")

    def _check(self, result: int) -> None:
        if result != 0:
            self._raise_last_error()

    def initialize_roots(self, policy_logits: np.ndarray) -> None:
        logits = np.ascontiguousarray(policy_logits, dtype=np.float32)
        if logits.ndim != 2 or logits.shape[0] != self.size:
            raise ValueError("root policy logits must have shape (num_searches, action_space)")
        self._check(
            self._library.cudanatron_search_pool_initialize_roots(
                self._handle,
                logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                logits.shape[1],
                logits.shape[1],
            )
        )

    def set_root_wdls(self, wdls: np.ndarray) -> None:
        values = np.ascontiguousarray(wdls, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] != self.size or values.shape[1] < 3:
            raise ValueError("root WDLs must have shape (num_searches, 3)")
        self._check(
            self._library.cudanatron_search_pool_set_root_wdls(
                self._handle,
                values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                values.shape[1],
            )
        )

    def enable_completed_q(self, c_visit: float = 50.0, c_scale: float = 1.0) -> None:
        self._check(
            self._library.cudanatron_search_pool_enable_completed_q(
                self._handle, float(c_visit), float(c_scale)
            )
        )

    def add_dirichlet_noise(self, alpha: float, fraction: float) -> None:
        self._check(
            self._library.cudanatron_search_pool_add_dirichlet_noise(
                self._handle, float(alpha), float(fraction)
            )
        )

    def add_simulations_all(self, count: int) -> None:
        self._check(self._library.cudanatron_search_pool_add_simulations_all(self._handle, int(count)))

    @property
    def remaining_simulations(self) -> int:
        return int(self._library.cudanatron_search_pool_remaining_simulations(self._handle))

    def select_leaves(self, capacity: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        observations = np.zeros((capacity, self.observation_size), dtype=np.float32)
        players = np.full(capacity, -1, dtype=np.int32)
        tokens = np.full(capacity, -1, dtype=np.int32)
        filled = int(
            self._library.cudanatron_search_pool_select_leaves(
                self._handle,
                int(capacity),
                observations.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                self.observation_size,
                players.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                tokens.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            )
        )
        if filled < 0:
            self._raise_last_error()
        return observations[:filled], players[:filled], tokens[:filled]

    def evaluate_leaves(
        self,
        tokens: np.ndarray,
        policy_logits: np.ndarray,
        wdls: np.ndarray,
    ) -> None:
        token_values = np.ascontiguousarray(tokens, dtype=np.int32)
        logits = np.ascontiguousarray(policy_logits, dtype=np.float32)
        values = np.ascontiguousarray(wdls, dtype=np.float64)
        if logits.ndim != 2 or values.ndim != 2:
            raise ValueError("leaf policy and WDL must be 2-D")
        if logits.shape[0] != token_values.size or values.shape[0] != token_values.size:
            raise ValueError("leaf batch dimensions do not match tokens")
        self._check(
            self._library.cudanatron_search_pool_evaluate_leaves(
                self._handle,
                token_values.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                token_values.size,
                logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                logits.shape[1],
                logits.shape[1],
                values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                values.shape[1],
            )
        )

    def root_visits(self, index: int) -> np.ndarray:
        visits = np.zeros(self.action_space_size, dtype=np.uint32)
        self._check(
            self._library.cudanatron_search_pool_root_visits(
                self._handle,
                int(index),
                visits.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
                visits.size,
            )
        )
        return visits

    def root_wdl(self, index: int) -> np.ndarray:
        wdl = np.zeros(3, dtype=np.float64)
        self._check(
            self._library.cudanatron_search_pool_root_wdl(
                self._handle,
                int(index),
                wdl.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                wdl.size,
            )
        )
        return wdl

    def metrics(self, index: int) -> NativeSearchMetrics:
        raw = _SearchMetrics()
        self._check(
            self._library.cudanatron_search_pool_metrics(
                self._handle, int(index), ctypes.byref(raw)
            )
        )
        return NativeSearchMetrics(
            simulations=int(raw.simulations),
            principal_variation_depth=int(raw.principal_variation_depth),
            maximum_depth=int(raw.maximum_depth),
            mean_depth=float(raw.mean_depth),
            root_value=float(raw.root_value),
            retained_root_visits=int(raw.retained_root_visits),
            pruned_actions=int(raw.pruned_actions),
            coalesced_outcomes=int(raw.coalesced_outcomes),
            tree_reused=bool(raw.tree_reused),
        )

    def advance(self, index: int, action: int) -> bool:
        result = int(self._library.cudanatron_search_pool_advance(self._handle, int(index), int(action)))
        if result < 0:
            self._raise_last_error()
        return result == 1

    def advance_to(self, index: int, action: int, observed: NativeGame) -> bool:
        result = int(
            self._library.cudanatron_search_pool_advance_to_game(
                self._handle, int(index), int(action), observed._handle
            )
        )
        if result < 0:
            self._raise_last_error()
        return result == 1

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._library.cudanatron_search_pool_destroy(self._handle)
            self._handle = None

    def __enter__(self) -> NativeSearchPool:
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

