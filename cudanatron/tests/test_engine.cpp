#include "cudanatron/action_space.hpp"
#include "cudanatron/game.hpp"
#include "cudanatron/map.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}

void test_maps_have_catanatron_sizes() {
    cudanatron::PackedMap base{};
    require(
        cudanatron::build_packed_map(
            &base, cudanatron::MapType::base, 0, cudanatron::NumberPlacement::official_spiral) ==
            cudanatron::Status::ok,
        "base map");
    require(base.num_land_tiles == 19, "base land tiles");
    require(base.num_nodes == 54, "base nodes");
    require(base.num_edges == 72, "base edges");
    require(base.tiles[base.desert_tile].resource < 0, "base desert");

    cudanatron::PackedMap mini{};
    require(
        cudanatron::build_packed_map(
            &mini, cudanatron::MapType::mini, 1, cudanatron::NumberPlacement::official_spiral) ==
            cudanatron::Status::ok,
        "mini map");
    require(mini.num_land_tiles == 7, "mini land tiles");
    require(mini.num_nodes == 24, "mini nodes");

    cudanatron::PackedMap tournament{};
    require(
        cudanatron::build_packed_map(
            &tournament,
            cudanatron::MapType::tournament,
            0,
            cudanatron::NumberPlacement::random) == cudanatron::Status::ok,
        "tournament map");
    require(tournament.num_land_tiles == 19, "tournament land tiles");
}

void test_initial_setup_completes() {
    cudanatron::PackedMap map{};
    require(
        cudanatron::build_packed_map(
            &map,
            cudanatron::MapType::tournament,
            42,
            cudanatron::NumberPlacement::official_spiral) == cudanatron::Status::ok,
        "tournament map for setup");

    cudanatron::GameConfig config{};
    config.num_players = 2;
    config.map_type = cudanatron::MapType::tournament;
    config.game_seed = 42;
    config.map_seed = 42;

    cudanatron::PackedGame game{};
    require(cudanatron::initialize_game(map, config, &game) == cudanatron::Status::ok, "init");
    require(game.current_prompt == cudanatron::ActionPrompt::build_initial_settlement, "prompt");

    cudanatron::FlatActionSpace space{};
    require(cudanatron::build_flat_action_space(&space, map, 2) == cudanatron::Status::ok, "space");
    require(space.size == 313, "2p tournament action space size");

    int steps = 0;
    while (game.current_prompt == cudanatron::ActionPrompt::build_initial_settlement ||
           game.current_prompt == cudanatron::ActionPrompt::build_initial_road) {
        cudanatron::PackedAction legal[cudanatron::kMaxLegalActions];
        const int count = cudanatron::generate_legal_actions(map, game, legal, cudanatron::kMaxLegalActions);
        require(count > 0, "setup has legal actions");
        require(
            cudanatron::execute_action(map, &game, legal[0], nullptr) == cudanatron::Status::ok,
            "setup step");
        ++steps;
        require(steps < 16, "setup terminated");
    }
    require(game.is_initial_build_phase == 0, "left setup");
    require(game.current_prompt == cudanatron::ActionPrompt::play_turn, "play turn");
    require(game.current_player_index == 0, "first player to roll");
    require(game.num_turns == 2, "2p setup advances twice");
}

void test_random_rollout_does_not_dead_end() {
    cudanatron::PackedMap map{};
    require(
        cudanatron::build_packed_map(
            &map, cudanatron::MapType::mini, 7, cudanatron::NumberPlacement::random) ==
            cudanatron::Status::ok,
        "mini map");
    cudanatron::GameConfig config{};
    config.num_players = 2;
    config.map_type = cudanatron::MapType::mini;
    config.game_seed = 99;
    config.victory_points_to_win = 10;
    cudanatron::PackedGame game{};
    require(cudanatron::initialize_game(map, config, &game) == cudanatron::Status::ok, "init mini");

    for (int step = 0; step < 4000; ++step) {
        if (cudanatron::winning_player(game) >= 0 || game.num_turns >= 1000) {
            break;
        }
        cudanatron::PackedAction legal[cudanatron::kMaxLegalActions];
        const int count = cudanatron::generate_legal_actions(map, game, legal, cudanatron::kMaxLegalActions);
        require(count > 0, "rollout has legal actions");
        const int choice = game.rng.uniform_int(0, count - 1);
        require(
            cudanatron::execute_action(map, &game, legal[choice], nullptr) == cudanatron::Status::ok,
            "rollout step");
    }
}

}  // namespace

int main() {
    test_maps_have_catanatron_sizes();
    test_initial_setup_completes();
    test_random_rollout_does_not_dead_end();
    std::puts("ok");
    return 0;
}
