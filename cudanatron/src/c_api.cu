#include "cudanatron/c_api.h"

#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "cudanatron/action_space.hpp"
#include "cudanatron/batch.hpp"
#include "cudanatron/game.hpp"
#include "cudanatron/map.hpp"
#include "cudanatron/mcts.hpp"
#include "cudanatron/observation.hpp"

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
using cudanatron::EdgePosition;
using cudanatron::MCTSSearch;
using cudanatron::NodePosition;
using cudanatron::ObservationLayout;
using cudanatron::SearchPool;
using cudanatron::TilePosition;
using cudanatron::WDL;
using cudanatron::fill_observation_layout;
using cudanatron::full_observation_size;
using cudanatron::write_full_observation;
using cudanatron::write_legal_mask;
using cudanatron::BatchBuffers;
using cudanatron::BatchConfig;
using cudanatron::GameBatch;
using cudanatron::RewardFunction;

struct cudanatron_game {
    GameConfig config{};
    PackedMap map{};
    PackedGame game{};
    FlatActionSpace action_space{};
    ObservationLayout layout{};
    bool has_layout{false};
};

struct cudanatron_search_pool {
    std::unique_ptr<SearchPool> pool;
};

struct cudanatron_batch {
    std::unique_ptr<GameBatch> batch;
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

void apply_observation_layout(
    cudanatron_game* handle,
    int32_t width,
    int32_t height,
    const cudanatron_node_position* nodes,
    size_t node_count,
    const cudanatron_edge_position* edges,
    size_t edge_count,
    const cudanatron_tile_position* tiles,
    size_t tile_count) {
    if (handle == nullptr) {
        throw std::invalid_argument("null game");
    }
    std::vector<NodePosition> node_positions(node_count);
    for (size_t i = 0; i < node_count; ++i) {
        node_positions[i] = NodePosition{nodes[i].node, nodes[i].x, nodes[i].y};
    }
    std::vector<EdgePosition> edge_positions(edge_count);
    for (size_t i = 0; i < edge_count; ++i) {
        edge_positions[i] = EdgePosition{edges[i].a, edges[i].b, edges[i].x, edges[i].y};
    }
    std::vector<TilePosition> tile_positions(tile_count);
    for (size_t i = 0; i < tile_count; ++i) {
        tile_positions[i] = TilePosition{
            tiles[i].x,
            tiles[i].y,
            tiles[i].z,
            tiles[i].board_x,
            tiles[i].board_y,
        };
    }
    require_ok(
        fill_observation_layout(
            &handle->layout,
            width,
            height,
            node_positions.data(),
            static_cast<int>(node_positions.size()),
            edge_positions.data(),
            static_cast<int>(edge_positions.size()),
            tile_positions.data(),
            static_cast<int>(tile_positions.size()),
            handle->map),
        "observation layout");
    handle->has_layout = true;
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

int32_t cudanatron_game_set_observation_layout(
    cudanatron_game* handle,
    int32_t width,
    int32_t height,
    const cudanatron_node_position* nodes,
    size_t node_count,
    const cudanatron_edge_position* edges,
    size_t edge_count,
    const cudanatron_tile_position* tiles,
    size_t tile_count) {
    try {
        if (nodes == nullptr || edges == nullptr || tiles == nullptr) {
            throw std::invalid_argument("observation layout pointers are null");
        }
        apply_observation_layout(
            handle,
            width,
            height,
            nodes,
            node_count,
            edges,
            edge_count,
            tiles,
            tile_count);
        return 0;
    } catch (const std::exception& error) {
        set_error(error);
        return -1;
    }
}

int32_t cudanatron_game_observation_size(const cudanatron_game* handle) {
    if (handle == nullptr || !handle->has_layout) {
        return -1;
    }
    return full_observation_size(
        handle->game.num_players, handle->layout.width, handle->layout.height);
}

int32_t cudanatron_game_write_observation(
    const cudanatron_game* handle,
    int32_t base_player,
    float* output,
    size_t output_size) {
    if (handle == nullptr || output == nullptr) {
        last_error = "null game or observation";
        return -1;
    }
    if (!handle->has_layout) {
        last_error = "observation layout has not been set";
        return -1;
    }
    const Status status = write_full_observation(
        handle->map,
        handle->game,
        base_player,
        handle->layout,
        output,
        static_cast<int>(output_size));
    if (status != Status::ok) {
        last_error = "failed to write observation";
        return -1;
    }
    return 0;
}

cudanatron_search_pool* cudanatron_search_pool_create(
    cudanatron_game* const* games,
    size_t game_count,
    double c_puct,
    uint64_t seed,
    int32_t canonical_pruning) {
    try {
        if (games == nullptr || game_count == 0) {
            throw std::invalid_argument("search pool requires at least one game");
        }
        ObservationLayout layout{};
        bool has_layout = false;
        std::vector<std::unique_ptr<MCTSSearch>> searches;
        searches.reserve(game_count);
        for (size_t i = 0; i < game_count; ++i) {
            auto* game = games[i];
            if (game == nullptr) {
                throw std::invalid_argument("search pool contains a null game");
            }
            if (!game->has_layout) {
                throw std::invalid_argument("search pool games need an observation layout");
            }
            if (!has_layout) {
                layout = game->layout;
                has_layout = true;
            }
            searches.push_back(std::make_unique<MCTSSearch>(
                game->map,
                game->game,
                game->action_space,
                c_puct,
                seed + static_cast<uint64_t>(i),
                canonical_pruning != 0));
        }
        auto handle = std::make_unique<cudanatron_search_pool>();
        handle->pool = std::make_unique<SearchPool>(layout, std::move(searches));
        return handle.release();
    } catch (const std::exception& error) {
        set_error(error);
        return nullptr;
    }
}

void cudanatron_search_pool_destroy(cudanatron_search_pool* handle) { delete handle; }

int32_t cudanatron_search_pool_size(const cudanatron_search_pool* handle) {
    return handle == nullptr || handle->pool == nullptr ? -1 : handle->pool->size();
}

int32_t cudanatron_search_pool_observation_size(const cudanatron_search_pool* handle) {
    return handle == nullptr || handle->pool == nullptr ? -1
                                                        : handle->pool->observation_size();
}

int32_t cudanatron_search_pool_initialize_roots(
    cudanatron_search_pool* handle,
    const float* policy_logits,
    size_t policy_stride,
    size_t policy_size) {
    if (handle == nullptr || handle->pool == nullptr) {
        last_error = "null search pool";
        return -1;
    }
    try {
        handle->pool->initialize_roots(
            policy_logits,
            static_cast<int>(policy_stride),
            static_cast<int>(policy_size));
        return 0;
    } catch (const std::exception& error) {
        set_error(error);
        return -1;
    }
}

int32_t cudanatron_search_pool_set_root_wdls(
    cudanatron_search_pool* handle,
    const double* wdls,
    size_t wdl_stride) {
    if (handle == nullptr || handle->pool == nullptr) {
        last_error = "null search pool";
        return -1;
    }
    try {
        handle->pool->set_root_network_wdls(wdls, static_cast<int>(wdl_stride));
        return 0;
    } catch (const std::exception& error) {
        set_error(error);
        return -1;
    }
}

int32_t cudanatron_search_pool_enable_completed_q(
    cudanatron_search_pool* handle,
    double c_visit,
    double c_scale) {
    if (handle == nullptr || handle->pool == nullptr) {
        last_error = "null search pool";
        return -1;
    }
    try {
        handle->pool->enable_completed_q_selection(c_visit, c_scale);
        return 0;
    } catch (const std::exception& error) {
        set_error(error);
        return -1;
    }
}

int32_t cudanatron_search_pool_add_dirichlet_noise(
    cudanatron_search_pool* handle,
    double alpha,
    double fraction) {
    if (handle == nullptr || handle->pool == nullptr) {
        last_error = "null search pool";
        return -1;
    }
    try {
        handle->pool->add_root_dirichlet_noise(alpha, fraction);
        return 0;
    } catch (const std::exception& error) {
        set_error(error);
        return -1;
    }
}

int32_t cudanatron_search_pool_add_simulations_all(
    cudanatron_search_pool* handle,
    int32_t count) {
    if (handle == nullptr || handle->pool == nullptr) {
        last_error = "null search pool";
        return -1;
    }
    try {
        handle->pool->add_simulations_all(count);
        return 0;
    } catch (const std::exception& error) {
        set_error(error);
        return -1;
    }
}

int32_t cudanatron_search_pool_remaining_simulations(const cudanatron_search_pool* handle) {
    return handle == nullptr || handle->pool == nullptr ? -1
                                                        : handle->pool->remaining_simulations();
}

int32_t cudanatron_search_pool_select_leaves(
    cudanatron_search_pool* handle,
    int32_t capacity,
    float* observations,
    size_t observation_stride,
    int32_t* players,
    int32_t* tokens) {
    if (handle == nullptr || handle->pool == nullptr) {
        last_error = "null search pool";
        return -1;
    }
    try {
        return handle->pool->select_leaves(
            capacity,
            observations,
            static_cast<int>(observation_stride),
            players,
            tokens);
    } catch (const std::exception& error) {
        set_error(error);
        return -1;
    }
}

int32_t cudanatron_search_pool_evaluate_leaves(
    cudanatron_search_pool* handle,
    const int32_t* tokens,
    size_t count,
    const float* policy_logits,
    size_t policy_stride,
    size_t policy_size,
    const double* wdls,
    size_t wdl_stride) {
    if (handle == nullptr || handle->pool == nullptr) {
        last_error = "null search pool";
        return -1;
    }
    try {
        handle->pool->evaluate_leaves(
            tokens,
            static_cast<int>(count),
            policy_logits,
            static_cast<int>(policy_stride),
            static_cast<int>(policy_size),
            wdls,
            static_cast<int>(wdl_stride));
        return 0;
    } catch (const std::exception& error) {
        set_error(error);
        return -1;
    }
}

int32_t cudanatron_search_pool_root_visits(
    const cudanatron_search_pool* handle,
    int32_t index,
    uint32_t* visits,
    size_t visit_count) {
    if (handle == nullptr || handle->pool == nullptr) {
        last_error = "null search pool";
        return -1;
    }
    try {
        handle->pool->search(index).root_visits(visits, static_cast<int>(visit_count));
        return 0;
    } catch (const std::exception& error) {
        set_error(error);
        return -1;
    }
}

int32_t cudanatron_search_pool_root_wdl(
    const cudanatron_search_pool* handle,
    int32_t index,
    double* wdl,
    size_t wdl_size) {
    if (handle == nullptr || handle->pool == nullptr || wdl == nullptr || wdl_size < 3) {
        last_error = "null search pool or WDL buffer";
        return -1;
    }
    try {
        const WDL value = handle->pool->search(index).root_wdl();
        wdl[0] = value.win;
        wdl[1] = value.draw;
        wdl[2] = value.loss;
        return 0;
    } catch (const std::exception& error) {
        set_error(error);
        return -1;
    }
}

int32_t cudanatron_search_pool_metrics(
    const cudanatron_search_pool* handle,
    int32_t index,
    cudanatron_search_metrics* output) {
    if (handle == nullptr || handle->pool == nullptr || output == nullptr) {
        last_error = "null search pool or metrics";
        return -1;
    }
    try {
        const auto metrics = handle->pool->search(index).metrics();
        output->simulations = metrics.simulations;
        output->principal_variation_depth = metrics.principal_variation_depth;
        output->maximum_depth = metrics.maximum_depth;
        output->mean_depth = metrics.mean_depth;
        output->root_value = metrics.root_value;
        output->retained_root_visits = metrics.retained_root_visits;
        output->pruned_actions = metrics.pruned_actions;
        output->coalesced_outcomes = metrics.coalesced_outcomes;
        output->tree_reused = metrics.tree_reused ? 1 : 0;
        return 0;
    } catch (const std::exception& error) {
        set_error(error);
        return -1;
    }
}

int32_t cudanatron_search_pool_advance(
    cudanatron_search_pool* handle,
    int32_t index,
    size_t action_index) {
    if (handle == nullptr || handle->pool == nullptr) {
        last_error = "null search pool";
        return -1;
    }
    try {
        return handle->pool->advance(index, static_cast<int>(action_index)) ? 1 : 0;
    } catch (const std::exception& error) {
        set_error(error);
        return -1;
    }
}

int32_t cudanatron_search_pool_advance_to_game(
    cudanatron_search_pool* handle,
    int32_t index,
    size_t action_index,
    const cudanatron_game* observed_game) {
    if (handle == nullptr || handle->pool == nullptr || observed_game == nullptr) {
        last_error = "null search pool or observed game";
        return -1;
    }
    try {
        return handle->pool->advance_to(
                   index, static_cast<int>(action_index), observed_game->game)
                   ? 1
                   : 0;
    } catch (const std::exception& error) {
        set_error(error);
        return -1;
    }
}

cudanatron_batch* cudanatron_batch_create(
    int32_t num_envs,
    int32_t num_players,
    int32_t map_type,
    int32_t discard_limit,
    int32_t friendly_robber,
    int32_t victory_points_to_win,
    int32_t number_placement,
    int32_t reward_function,
    int32_t turns_limit,
    int32_t board_width,
    int32_t board_height,
    const cudanatron_node_position* node_positions,
    size_t node_position_count,
    const cudanatron_edge_position* edge_positions,
    size_t edge_position_count,
    const cudanatron_tile_position* tile_positions,
    size_t tile_position_count) {
    try {
        if (node_positions == nullptr || edge_positions == nullptr ||
            tile_positions == nullptr) {
            throw std::invalid_argument("batch observation layout pointers are null");
        }
        PackedMap template_map{};
        require_ok(
            build_packed_map(
                &template_map,
                parse_map_type(map_type),
                0,
                parse_number_placement(number_placement)),
            "batch template map");
        ObservationLayout layout{};
        std::vector<NodePosition> nodes(node_position_count);
        for (size_t i = 0; i < node_position_count; ++i) {
            nodes[i] = NodePosition{
                node_positions[i].node, node_positions[i].x, node_positions[i].y};
        }
        std::vector<EdgePosition> edges(edge_position_count);
        for (size_t i = 0; i < edge_position_count; ++i) {
            edges[i] = EdgePosition{
                edge_positions[i].a,
                edge_positions[i].b,
                edge_positions[i].x,
                edge_positions[i].y};
        }
        std::vector<TilePosition> tiles(tile_position_count);
        for (size_t i = 0; i < tile_position_count; ++i) {
            tiles[i] = TilePosition{
                tile_positions[i].x,
                tile_positions[i].y,
                tile_positions[i].z,
                tile_positions[i].board_x,
                tile_positions[i].board_y};
        }
        require_ok(
            fill_observation_layout(
                &layout,
                board_width,
                board_height,
                nodes.data(),
                static_cast<int>(nodes.size()),
                edges.data(),
                static_cast<int>(edges.size()),
                tiles.data(),
                static_cast<int>(tiles.size()),
                template_map),
            "batch observation layout");
        BatchConfig config{};
        config.num_envs = num_envs;
        config.num_players = num_players;
        config.map_type = parse_map_type(map_type);
        config.number_placement = parse_number_placement(number_placement);
        config.discard_limit = discard_limit;
        config.friendly_robber = friendly_robber != 0;
        config.victory_points_to_win = victory_points_to_win;
        config.reward_function = reward_function == 1 ? RewardFunction::win
                                                     : RewardFunction::shaped;
        config.turns_limit = turns_limit;
        auto handle = std::make_unique<cudanatron_batch>();
        handle->batch = std::make_unique<GameBatch>(config, layout);
        return handle.release();
    } catch (const std::exception& error) {
        set_error(error);
        return nullptr;
    }
}

void cudanatron_batch_destroy(cudanatron_batch* handle) { delete handle; }

int32_t cudanatron_batch_bind_buffers(
    cudanatron_batch* handle,
    uint8_t* observations,
    size_t observation_row_stride,
    size_t action_mask_offset,
    size_t observation_offset,
    int32_t* actions,
    float* rewards,
    uint8_t* terminals,
    uint8_t* truncations,
    uint8_t* masks) {
    if (handle == nullptr || handle->batch == nullptr) {
        last_error = "null batch";
        return -1;
    }
    try {
        BatchBuffers buffers{};
        buffers.observations = observations;
        buffers.observation_row_stride = observation_row_stride;
        buffers.action_mask_offset = action_mask_offset;
        buffers.observation_offset = observation_offset;
        buffers.actions = actions;
        buffers.rewards = rewards;
        buffers.terminals = terminals;
        buffers.truncations = truncations;
        buffers.masks = masks;
        handle->batch->bind(buffers);
        return 0;
    } catch (const std::exception& error) {
        set_error(error);
        return -1;
    }
}

int32_t cudanatron_batch_reset_all(
    cudanatron_batch* handle,
    const uint64_t* map_seeds,
    const uint64_t* game_seeds,
    size_t seed_count) {
    if (handle == nullptr || handle->batch == nullptr) {
        last_error = "null batch";
        return -1;
    }
    try {
        handle->batch->reset_all(map_seeds, game_seeds, seed_count);
        return 0;
    } catch (const std::exception& error) {
        set_error(error);
        return -1;
    }
}

int32_t cudanatron_batch_reset_at(
    cudanatron_batch* handle,
    int32_t env_index,
    uint64_t map_seed,
    uint64_t game_seed,
    int32_t preserve_transition) {
    if (handle == nullptr || handle->batch == nullptr) {
        last_error = "null batch";
        return -1;
    }
    try {
        handle->batch->reset_at(env_index, map_seed, game_seed, preserve_transition != 0);
        return 0;
    } catch (const std::exception& error) {
        set_error(error);
        return -1;
    }
}

int32_t cudanatron_batch_step(cudanatron_batch* handle) {
    if (handle == nullptr || handle->batch == nullptr) {
        last_error = "null batch";
        return -1;
    }
    try {
        handle->batch->step();
        return 0;
    } catch (const std::exception& error) {
        set_error(error);
        return -1;
    }
}

int32_t cudanatron_batch_action_space_size(const cudanatron_batch* handle) {
    return handle == nullptr || handle->batch == nullptr ? -1
                                                         : handle->batch->action_space_size();
}

int32_t cudanatron_batch_observation_size(const cudanatron_batch* handle) {
    return handle == nullptr || handle->batch == nullptr ? -1
                                                         : handle->batch->observation_size();
}

}  // extern "C"
