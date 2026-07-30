from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

MapType = Literal["BASE", "MINI", "TOURNAMENT"]


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
        )


def find_cppanatron_library() -> Path:
    override = os.environ.get("CPPANATRON_LIBRARY")
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"CPPANATRON_LIBRARY does not exist: {path}")
        return path

    repository = Path(__file__).resolve().parents[4]
    candidates = [
        repository / "cppanatron" / "build" / "libcppanatron.so",
        repository / "cppanatron" / "build-make" / "libcppanatron.so",
        repository / "cppanatron" / "build" / "libcppanatron.dylib",
        repository / "cppanatron" / "build-make" / "libcppanatron.dylib",
        repository / "cppanatron" / "build" / "cppanatron.dll",
        repository / "cppanatron" / "build-make" / "cppanatron.dll",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "cppanatron shared library was not found. Build it with "
        "`cmake -S cppanatron -B cppanatron/build && "
        "cmake --build cppanatron/build` or set CPPANATRON_LIBRARY."
    )


def _load_library(path: Path | None = None) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path or find_cppanatron_library()))
    handle = ctypes.c_void_p
    library.cppanatron_last_error.restype = ctypes.c_char_p
    library.cppanatron_game_create.argtypes = [
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_uint64,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    library.cppanatron_game_create.restype = handle
    library.cppanatron_game_destroy.argtypes = [handle]
    library.cppanatron_game_reset.argtypes = [handle, ctypes.c_uint64]
    library.cppanatron_game_action_space_size.argtypes = [handle]
    library.cppanatron_game_action_space_size.restype = ctypes.c_int32
    library.cppanatron_game_valid_action_mask.argtypes = [
        handle,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
    ]
    library.cppanatron_game_step.argtypes = [handle, ctypes.c_int32]
    library.cppanatron_game_step_replay.argtypes = [
        handle,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    library.cppanatron_game_num_players.argtypes = [handle]
    library.cppanatron_game_num_players.restype = ctypes.c_int32
    library.cppanatron_game_current_player.argtypes = [handle]
    library.cppanatron_game_current_player.restype = ctypes.c_int32
    library.cppanatron_game_current_prompt.argtypes = [handle]
    library.cppanatron_game_current_prompt.restype = ctypes.c_int32
    library.cppanatron_game_num_turns.argtypes = [handle]
    library.cppanatron_game_num_turns.restype = ctypes.c_int32
    library.cppanatron_game_winner.argtypes = [handle]
    library.cppanatron_game_winner.restype = ctypes.c_int32
    library.cppanatron_game_player_state.argtypes = [
        handle,
        ctypes.c_int32,
        ctypes.POINTER(_PlayerState),
    ]
    return library


class NativeGame:
    """Owned Python handle for one cppanatron game."""

    _MAP_TYPES = {"BASE": 0, "MINI": 1, "TOURNAMENT": 2}

    def __init__(
        self,
        num_players: int = 2,
        map_type: MapType = "BASE",
        seed: int = 0,
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
        self._handle = self._library.cppanatron_game_create(
            num_players,
            native_map_type,
            seed,
            discard_limit,
            int(friendly_robber),
            vps_to_win,
        )
        if not self._handle:
            self._raise_last_error()
        self.num_players = int(self._library.cppanatron_game_num_players(self._handle))
        self.action_space_size = int(
            self._library.cppanatron_game_action_space_size(self._handle)
        )

    def _raise_last_error(self) -> None:
        message = self._library.cppanatron_last_error()
        raise RuntimeError(message.decode() if message else "unknown cppanatron error")

    def _check(self, result: int) -> None:
        if result != 0:
            self._raise_last_error()

    @property
    def current_player(self) -> int:
        return int(self._library.cppanatron_game_current_player(self._handle))

    @property
    def current_prompt(self) -> int:
        return int(self._library.cppanatron_game_current_prompt(self._handle))

    @property
    def num_turns(self) -> int:
        return int(self._library.cppanatron_game_num_turns(self._handle))

    @property
    def winner(self) -> int | None:
        value = int(self._library.cppanatron_game_winner(self._handle))
        return None if value == -1 else value

    def reset(self, seed: int) -> None:
        self._check(self._library.cppanatron_game_reset(self._handle, seed))

    def valid_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_space_size, dtype=np.uint8)
        self._check(
            self._library.cppanatron_game_valid_action_mask(
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
            self._check(self._library.cppanatron_game_step(self._handle, int(action)))
            return
        die_one, die_two = dice or (-1, -1)
        self._check(
            self._library.cppanatron_game_step_replay(
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
            self._library.cppanatron_game_player_state(
                self._handle, player, ctypes.byref(state)
            )
        )
        return NativePlayerState.from_c(state)

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._library.cppanatron_game_destroy(self._handle)
            self._handle = None

    def __enter__(self) -> NativeGame:
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
