#include "cudanatron/c_api.h"

#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "cudanatron/action_space.hpp"
#include "cudanatron/game.hpp"
#include "cudanatron/map.hpp"

using cudanatron::ActionPrompt;
using cudanatron::FlatActionSpace;
using cudanatron::GameConfig;
using cudanatron::MapType;
using cudanatron::NumberPlacement;
using cudanatron::PackedAction;
using cudanatron::PackedGame;
using cudanatron::PackedMap;
using cudanatron::Replay;
using cudanatron::Status;
using cudanatron::build_flat_action_space;
using cudanatron::build_packed_map;
using cudanatron::decode_flat_action;
using cudanatron::execute_action;
using cudanatron::initialize_game;
using cudanatron::turns_since;
using cudanatron::winning_player;
using cudanatron::write_legal_mask;

struct cudanatron_game {
    GameConfig config{};
    PackedMap map{};
    PackedGame game{};
    FlatActionSpace action_space{};
};

namespace {

thread_local std::string last_error;

MapType parse_map_type(int value) {
    switch (value) {
        case CUDANATRON_MAP_BASE:
            return MapType::base;
        case CUDANATRON_MAP_MINI:
            return MapType::mini;
        case CUDANATRON_MAP_TOURNAMENT:
            return MapType::tournament;
        default:
            throw std::invalid_argument("invalid map type");
    }
}

NumberPlacement parse_number_placement(int value) {
    switch (value) {
        case CUDANATRON_NUMBER_PLACEMENT_OFFICIAL_SPIRAL:
            return NumberPlacement::official_spiral;
        case CUDANATRON_NUMBER_PLACEMENT_RANDOM:
            return NumberPlacement::random;
        default:
            throw std::invalid_argument("invalid number placement");
    }
}

void set_error(const std::exception& error) {
    last_error = error.what();
}

void require_ok(Status status, const char* what) {
    if (status != Status::ok) {
        throw std::runtime_error(std::string(what) + " failed");
    }
}

std::unique_ptr<cudanatron_game> make_game(
    int32_t num_players,
    int32_t map_type,
    uint64_t map_seed,
    uint64_t game_seed,
    int32_t discard_limit,
    int32_t friendly_robber,
    int32_t victory_points_to_win,
    int32_t number_placement) {
    if (num_players < 1 || num_players > 4) {
        throw std::invalid_argument("num_players must be between one and four");
    }
    auto handle = std::make_unique<cudanatron_game>();
    handle->config.num_players = num_players;
    handle->config.map_type = parse_map_type(map_type);
    handle->config.number_placement = parse_number_placement(number_placement);
    handle->config.map_seed = map_seed;
    handle->config.game_seed = game_seed;
    handle->config.discard_limit = discard_limit;
    handle->config.friendly_robber = friendly_robber != 0;
    handle->config.victory_points_to_win = victory_points_to_win;
    require_ok(
        build_packed_map(
            &handle->map,
            handle->config.map_type,
            map_seed,
            handle->config.number_placement),
        "map");
    require_ok(initialize_game(handle->map, handle->config, &handle->game), "game");
    require_ok(
        build_flat_action_space(&handle->action_space, handle->map, num_players),
        "action space");
    return handle;
}

Replay make_replay(
    int32_t die_one,
    int32_t die_two,
    int32_t development_card,
    int32_t stolen_resource) {
    Replay replay{};
    if (die_one >= 1 && die_two >= 1) {
        replay.has_dice = true;
        replay.die0 = static_cast<std::int8_t>(die_one);
        replay.die1 = static_cast<std::int8_t>(die_two);
    }
    if (development_card >= 0) {
        replay.has_development_card = true;
        replay.development_card = static_cast<std::int8_t>(development_card);
    }
    if (stolen_resource >= 0) {
        replay.has_stolen_resource = true;
        replay.stolen_resource = static_cast<std::int8_t>(stolen_resource);
    }
    return replay;
}

}  // namespace

extern "C" {

const char* cudanatron_version(void) { return "0.1.0"; }

const char* cudanatron_last_error(void) { return last_error.c_str(); }

cudanatron_game* cudanatron_game_create_seeded_with_number_placement(
    int32_t num_players,
    int32_t map_type,
    uint64_t map_seed,
    uint64_t game_seed,
    int32_t discard_limit,
    int32_t friendly_robber,
    int32_t victory_points_to_win,
    int32_t number_placement) {
    try {
        return make_game(
                   num_players,
                   map_type,
                   map_seed,
                   game_seed,
                   discard_limit,
                   friendly_robber,
                   victory_points_to_win,
                   number_placement)
            .release();
    } catch (const std::exception& error) {
        set_error(error);
        return nullptr;
    }
}

void cudanatron_game_destroy(cudanatron_game* handle) { delete handle; }

int32_t cudanatron_game_reset_seeded(
    cudanatron_game* handle,
    uint64_t map_seed,
    uint64_t game_seed) {
    if (handle == nullptr) {
        last_error = "null game";
        return -1;
    }
    try {
        handle->config.map_seed = map_seed;
        handle->config.game_seed = game_seed;
        require_ok(
            build_packed_map(
                &handle->map,
                handle->config.map_type,
                map_seed,
                handle->config.number_placement),
            "map");
        require_ok(initialize_game(handle->map, handle->config, &handle->game), "game");
        return 0;
    } catch (const std::exception& error) {
        set_error(error);
        return -1;
    }
}

int32_t cudanatron_game_action_space_size(const cudanatron_game* handle) {
    return handle == nullptr ? -1 : handle->action_space.size;
}

int32_t cudanatron_game_valid_action_mask(
    const cudanatron_game* handle,
    uint8_t* mask,
    size_t mask_size) {
    if (handle == nullptr || mask == nullptr) {
        last_error = "null game or mask";
        return -1;
    }
    if (static_cast<int>(mask_size) < handle->action_space.size) {
        last_error = "mask is too small";
        return -1;
    }
    write_legal_mask(
        handle->map,
        handle->game,
        handle->action_space,
        mask,
        static_cast<int>(mask_size));
    return 0;
}

int32_t cudanatron_game_step(cudanatron_game* handle, int32_t flat_action) {
    return cudanatron_game_step_replay(handle, flat_action, -1, -1, -1, -1);
}

int32_t cudanatron_game_step_replay(
    cudanatron_game* handle,
    int32_t flat_action,
    int32_t die_one,
    int32_t die_two,
    int32_t development_card,
    int32_t stolen_resource) {
    if (handle == nullptr) {
        last_error = "null game";
        return -1;
    }
    if (flat_action < 0 || flat_action >= handle->action_space.size) {
        last_error = "flat action is out of range";
        return -1;
    }
    try {
        PackedAction action = decode_flat_action(
            handle->action_space, handle->map, handle->game, flat_action);
        Replay replay = make_replay(die_one, die_two, development_card, stolen_resource);
        const Status status = execute_action(handle->map, &handle->game, action, &replay);
        if (status == Status::illegal_action) {
            throw std::runtime_error("action is not currently playable");
        }
        require_ok(status, "step");
        return 0;
    } catch (const std::exception& error) {
        set_error(error);
        return -1;
    }
}

int32_t cudanatron_game_num_players(const cudanatron_game* handle) {
    return handle == nullptr ? -1 : handle->game.num_players;
}

int32_t cudanatron_game_current_player(const cudanatron_game* handle) {
    return handle == nullptr ? -1 : handle->game.current_player_index;
}

int32_t cudanatron_game_current_prompt(const cudanatron_game* handle) {
    return handle == nullptr ? -1 : static_cast<int32_t>(handle->game.current_prompt);
}

int32_t cudanatron_game_num_turns(const cudanatron_game* handle) {
    return handle == nullptr ? -1 : handle->game.num_turns;
}

int32_t cudanatron_game_winner(const cudanatron_game* handle) {
    if (handle == nullptr) {
        return -1;
    }
    return winning_player(handle->game);
}

int32_t cudanatron_game_flags(const cudanatron_game* handle, int32_t output[7]) {
    if (handle == nullptr || output == nullptr) {
        last_error = "null game or flags";
        return -1;
    }
    output[0] = handle->game.is_initial_build_phase;
    output[1] = handle->game.is_discarding;
    output[2] = handle->game.current_prompt == ActionPrompt::move_robber;
    output[3] = handle->game.is_road_building;
    output[4] = handle->game.current_player_index;
    output[5] = handle->game.current_turn_index;
    output[6] = handle->game.completed_turns;
    return 0;
}

int32_t cudanatron_game_robber_coordinate(const cudanatron_game* handle, int32_t output[3]) {
    if (handle == nullptr || output == nullptr) {
        last_error = "null game or coordinate";
        return -1;
    }
    const auto coordinate = handle->map.tiles[handle->game.robber_tile].coordinate;
    output[0] = coordinate.x;
    output[1] = coordinate.y;
    output[2] = coordinate.z;
    return 0;
}

int32_t cudanatron_game_development_cards_remaining(const cudanatron_game* handle) {
    return handle == nullptr ? -1 : handle->game.development_deck_size;
}

int32_t cudanatron_game_resource_bank(const cudanatron_game* handle, int32_t output[5]) {
    if (handle == nullptr || output == nullptr) {
        last_error = "null game or bank";
        return -1;
    }
    for (int i = 0; i < 5; ++i) {
        output[i] = handle->game.resource_bank[i];
    }
    return 0;
}

int32_t cudanatron_game_player_state(
    const cudanatron_game* handle,
    int32_t player,
    cudanatron_player_state* output) {
    if (handle == nullptr || output == nullptr) {
        last_error = "null game or player state";
        return -1;
    }
    if (player < 0 || player >= handle->game.num_players) {
        last_error = "player is out of range";
        return -1;
    }
    const auto& state = handle->game.players[player];
    std::memset(output, 0, sizeof(*output));
    output->victory_points = state.victory_points;
    output->actual_victory_points = state.actual_victory_points;
    output->roads_available = state.roads_available;
    output->settlements_available = state.settlements_available;
    output->cities_available = state.cities_available;
    output->has_road = state.has_road;
    output->has_army = state.has_army;
    output->has_rolled = state.has_rolled;
    output->has_played_development_card_in_turn = state.has_played_development_card_in_turn;
    output->longest_road_length = state.longest_road_length;
    for (int i = 0; i < 5; ++i) {
        output->resources[i] = state.resources[i];
        output->development_cards[i] = state.development_cards[i];
        output->played_development_cards[i] = state.played_development_cards[i];
    }
    for (int i = 0; i < 4; ++i) {
        output->development_card_owned_at_start[i] =
            (state.development_card_owned_at_start >> i) & 1;
    }
    output->turns_since_last_knight =
        turns_since(handle->game.completed_turns, state.last_knight_completed_turn);
    output->turns_since_last_development_card_bought =
        turns_since(handle->game.completed_turns, state.last_dev_bought_completed_turn);
    return 0;
}

int32_t cudanatron_game_buildings(
    const cudanatron_game* handle,
    cudanatron_building* output,
    size_t capacity) {
    if (handle == nullptr) {
        last_error = "null game";
        return -1;
    }
    int32_t count = 0;
    for (int node = 0; node < handle->map.num_nodes; ++node) {
        if (handle->game.node_owner[node] != cudanatron::kEmpty) {
            ++count;
        }
    }
    if (output == nullptr) {
        return count;
    }
    if (capacity < static_cast<size_t>(count)) {
        last_error = "building buffer is too small";
        return -1;
    }
    int32_t written = 0;
    for (int node = 0; node < handle->map.num_nodes; ++node) {
        if (handle->game.node_owner[node] == cudanatron::kEmpty) {
            continue;
        }
        output[written].node = node;
        output[written].color = handle->game.node_owner[node];
        output[written].building = handle->game.node_building[node];
        ++written;
    }
    return written;
}

int32_t cudanatron_game_roads(
    const cudanatron_game* handle,
    cudanatron_road* output,
    size_t capacity) {
    if (handle == nullptr) {
        last_error = "null game";
        return -1;
    }
    int32_t count = 0;
    for (int edge = 0; edge < handle->map.num_edges; ++edge) {
        if (handle->game.edge_owner[edge] != cudanatron::kEmpty) {
            ++count;
        }
    }
    if (output == nullptr) {
        return count;
    }
    if (capacity < static_cast<size_t>(count)) {
        last_error = "road buffer is too small";
        return -1;
    }
    int32_t written = 0;
    for (int edge = 0; edge < handle->map.num_edges; ++edge) {
        if (handle->game.edge_owner[edge] == cudanatron::kEmpty) {
            continue;
        }
        output[written].a = handle->map.edge_a[edge];
        output[written].b = handle->map.edge_b[edge];
        output[written].color = handle->game.edge_owner[edge];
        ++written;
    }
    return written;
}

int32_t cudanatron_game_tiles(
    const cudanatron_game* handle,
    cudanatron_tile* output,
    size_t capacity) {
    if (handle == nullptr) {
        last_error = "null game";
        return -1;
    }
    if (output == nullptr) {
        return handle->map.num_tiles;
    }
    if (capacity < static_cast<size_t>(handle->map.num_tiles)) {
        last_error = "tile buffer is too small";
        return -1;
    }
    for (int i = 0; i < handle->map.num_tiles; ++i) {
        const auto& tile = handle->map.tiles[i];
        output[i].x = tile.coordinate.x;
        output[i].y = tile.coordinate.y;
        output[i].z = tile.coordinate.z;
        output[i].id = tile.id;
        output[i].kind = static_cast<int32_t>(tile.kind);
        output[i].resource = tile.resource;
        output[i].number = tile.number;
        output[i].port_direction = tile.port_direction;
        for (int n = 0; n < 6; ++n) {
            output[i].nodes[n] = tile.nodes[n];
        }
    }
    return handle->map.num_tiles;
}

int32_t cudanatron_game_action_key(
    const cudanatron_game* handle,
    int32_t index,
    char* buffer,
    size_t buffer_size) {
    if (handle == nullptr || buffer == nullptr) {
        last_error = "null game or buffer";
        return -1;
    }
    if (index < 0 || index >= handle->action_space.size) {
        last_error = "flat action is out of range";
        return -1;
    }
    cudanatron::write_flat_action_key(
        handle->action_space.actions[index],
        buffer,
        static_cast<int>(buffer_size));
    return 0;
}

}  // extern "C"
