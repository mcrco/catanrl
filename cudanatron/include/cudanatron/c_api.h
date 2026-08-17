#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define CUDANATRON_API __declspec(dllexport)
#else
#define CUDANATRON_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cudanatron_game cudanatron_game;
typedef struct cudanatron_batch cudanatron_batch;
typedef struct cudanatron_search_pool cudanatron_search_pool;

enum cudanatron_map_type {
    CUDANATRON_MAP_BASE = 0,
    CUDANATRON_MAP_MINI = 1,
    CUDANATRON_MAP_TOURNAMENT = 2,
};

enum cudanatron_number_placement {
    CUDANATRON_NUMBER_PLACEMENT_OFFICIAL_SPIRAL = 0,
    CUDANATRON_NUMBER_PLACEMENT_RANDOM = 1,
};

typedef struct cudanatron_player_state {
    int32_t victory_points;
    int32_t actual_victory_points;
    int32_t roads_available;
    int32_t settlements_available;
    int32_t cities_available;
    int32_t has_road;
    int32_t has_army;
    int32_t has_rolled;
    int32_t has_played_development_card_in_turn;
    int32_t longest_road_length;
    int32_t resources[5];
    int32_t development_cards[5];
    int32_t played_development_cards[5];
    int32_t development_card_owned_at_start[4];
    int32_t turns_since_last_knight;
    int32_t turns_since_last_development_card_bought;
} cudanatron_player_state;

typedef struct cudanatron_building {
    int32_t node;
    int32_t color;
    int32_t building;
} cudanatron_building;

typedef struct cudanatron_road {
    int32_t a;
    int32_t b;
    int32_t color;
} cudanatron_road;

typedef struct cudanatron_tile {
    int32_t x;
    int32_t y;
    int32_t z;
    int32_t id;
    int32_t kind;
    int32_t resource;
    int32_t number;
    int32_t port_direction;
    int32_t nodes[6];
} cudanatron_tile;

typedef struct cudanatron_search_metrics {
    uint64_t simulations;
    uint32_t principal_variation_depth;
    uint32_t maximum_depth;
    double mean_depth;
    double root_value;
    uint32_t retained_root_visits;
    uint64_t pruned_actions;
    uint64_t coalesced_outcomes;
    int32_t tree_reused;
} cudanatron_search_metrics;

CUDANATRON_API const char* cudanatron_version(void);
CUDANATRON_API const char* cudanatron_last_error(void);

CUDANATRON_API cudanatron_game* cudanatron_game_create_seeded_with_number_placement(
    int32_t num_players,
    int32_t map_type,
    uint64_t map_seed,
    uint64_t game_seed,
    int32_t discard_limit,
    int32_t friendly_robber,
    int32_t victory_points_to_win,
    int32_t number_placement);
CUDANATRON_API void cudanatron_game_destroy(cudanatron_game* handle);
CUDANATRON_API int32_t cudanatron_game_reset_seeded(
    cudanatron_game* handle,
    uint64_t map_seed,
    uint64_t game_seed);
CUDANATRON_API int32_t cudanatron_game_action_space_size(const cudanatron_game* handle);
CUDANATRON_API int32_t cudanatron_game_valid_action_mask(
    const cudanatron_game* handle,
    uint8_t* mask,
    size_t mask_size);
CUDANATRON_API int32_t cudanatron_game_step(cudanatron_game* handle, int32_t flat_action);
CUDANATRON_API int32_t cudanatron_game_step_replay(
    cudanatron_game* handle,
    int32_t flat_action,
    int32_t die_one,
    int32_t die_two,
    int32_t development_card,
    int32_t stolen_resource);
CUDANATRON_API int32_t cudanatron_game_num_players(const cudanatron_game* handle);
CUDANATRON_API int32_t cudanatron_game_current_player(const cudanatron_game* handle);
CUDANATRON_API int32_t cudanatron_game_current_prompt(const cudanatron_game* handle);
CUDANATRON_API int32_t cudanatron_game_num_turns(const cudanatron_game* handle);
CUDANATRON_API int32_t cudanatron_game_winner(const cudanatron_game* handle);
CUDANATRON_API int32_t cudanatron_game_flags(const cudanatron_game* handle, int32_t output[7]);
CUDANATRON_API int32_t cudanatron_game_robber_coordinate(
    const cudanatron_game* handle,
    int32_t output[3]);
CUDANATRON_API int32_t cudanatron_game_development_cards_remaining(const cudanatron_game* handle);
CUDANATRON_API int32_t cudanatron_game_resource_bank(
    const cudanatron_game* handle,
    int32_t output[5]);
CUDANATRON_API int32_t cudanatron_game_player_state(
    const cudanatron_game* handle,
    int32_t player,
    cudanatron_player_state* output);
CUDANATRON_API int32_t cudanatron_game_buildings(
    const cudanatron_game* handle,
    cudanatron_building* output,
    size_t capacity);
CUDANATRON_API int32_t cudanatron_game_roads(
    const cudanatron_game* handle,
    cudanatron_road* output,
    size_t capacity);
CUDANATRON_API int32_t cudanatron_game_tiles(
    const cudanatron_game* handle,
    cudanatron_tile* output,
    size_t capacity);

#ifdef __cplusplus
}
#endif
