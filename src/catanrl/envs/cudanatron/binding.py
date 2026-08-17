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
    return library


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
