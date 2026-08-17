#include "cudanatron/action_space.hpp"
#include "cudanatron/chance.hpp"
#include "cudanatron/game.hpp"
#include "cudanatron/map.hpp"
#include "cudanatron/mcts.hpp"
#include "cudanatron/observation.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
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

void play_setup(const cudanatron::PackedMap& map, cudanatron::PackedGame* game) {
    int steps = 0;
    while (game->current_prompt == cudanatron::ActionPrompt::build_initial_settlement ||
           game->current_prompt == cudanatron::ActionPrompt::build_initial_road) {
        cudanatron::PackedAction legal[cudanatron::kMaxLegalActions];
        const int count = cudanatron::generate_legal_actions(
            map, *game, legal, cudanatron::kMaxLegalActions);
        require(count > 0, "setup has legal actions");
        require(
            cudanatron::execute_action(map, game, legal[0], nullptr) == cudanatron::Status::ok,
            "setup step");
        ++steps;
        require(steps < 16, "setup terminated");
    }
}

void test_observation_numeric_prefix() {
    cudanatron::PackedMap map{};
    require(
        cudanatron::build_packed_map(
            &map, cudanatron::MapType::base, 0, cudanatron::NumberPlacement::official_spiral) ==
            cudanatron::Status::ok,
        "base map");
    cudanatron::GameConfig config{};
    config.num_players = 2;
    cudanatron::PackedGame game{};
    require(cudanatron::initialize_game(map, config, &game) == cudanatron::Status::ok, "init");

    cudanatron::ObservationLayout layout{};
    layout.width = cudanatron::kBoardWidth;
    layout.height = cudanatron::kBoardHeight;
    const int size = cudanatron::full_observation_size(2, layout.width, layout.height);
    require(size == 32 * 2 + 10 + 21 * 11 * (4 + 12), "2p observation size");
    std::vector<float> observation(static_cast<std::size_t>(size), -1.0F);
    require(
        cudanatron::write_full_observation(
            map, game, 0, layout, observation.data(), size) == cudanatron::Status::ok,
        "write observation");
    require(observation[0] == 19.0F, "bank brick");
    require(observation[1] == static_cast<float>(game.development_deck_size), "bank dev");
    require(observation[6] == 0.0F, "not discarding");
    require(observation[7] == 1.0F, "initial build");
    require(observation[8] == 0.0F, "not moving robber");
}

void test_roll_chance_outcomes_sum_to_one() {
    cudanatron::PackedMap map{};
    require(
        cudanatron::build_packed_map(
            &map, cudanatron::MapType::mini, 3, cudanatron::NumberPlacement::random) ==
            cudanatron::Status::ok,
        "mini map");
    cudanatron::GameConfig config{};
    config.num_players = 2;
    config.map_type = cudanatron::MapType::mini;
    cudanatron::PackedGame game{};
    require(cudanatron::initialize_game(map, config, &game) == cudanatron::Status::ok, "init");
    play_setup(map, &game);

    cudanatron::PackedAction legal[cudanatron::kMaxLegalActions];
    const int count =
        cudanatron::generate_legal_actions(map, game, legal, cudanatron::kMaxLegalActions);
    require(count > 0, "post-setup actions");
    require(legal[0].type == cudanatron::ActionType::roll, "first action is roll");

    cudanatron::ChanceOutcome outcomes[cudanatron::kMaxChanceOutcomes];
    const int outcome_count = cudanatron::enumerate_chance_outcomes(
        map, game, legal[0], outcomes, cudanatron::kMaxChanceOutcomes);
    require(outcome_count == 11, "eleven dice totals");
    float total = 0.0F;
    for (int i = 0; i < outcome_count; ++i) {
        total += outcomes[i].probability;
        require(outcomes[i].replay.has_dice, "dice replay");
    }
    require(std::fabs(total - 1.0F) < 1e-6F, "dice probabilities sum to one");
}

void test_search_pool_runs_simulations() {
    cudanatron::PackedMap map{};
    require(
        cudanatron::build_packed_map(
            &map, cudanatron::MapType::mini, 5, cudanatron::NumberPlacement::random) ==
            cudanatron::Status::ok,
        "mini map");
    cudanatron::GameConfig config{};
    config.num_players = 2;
    config.map_type = cudanatron::MapType::mini;
    cudanatron::PackedGame game{};
    require(cudanatron::initialize_game(map, config, &game) == cudanatron::Status::ok, "init");
    play_setup(map, &game);

    cudanatron::FlatActionSpace space{};
    require(cudanatron::build_flat_action_space(&space, map, 2) == cudanatron::Status::ok, "space");
    cudanatron::ObservationLayout layout{};
    layout.width = cudanatron::kBoardWidth;
    layout.height = cudanatron::kBoardHeight;

    std::vector<std::unique_ptr<cudanatron::MCTSSearch>> searches;
    searches.push_back(std::make_unique<cudanatron::MCTSSearch>(map, game, space, 1.5, 1, false));
    cudanatron::SearchPool pool(layout, std::move(searches));

    std::vector<float> policy(static_cast<std::size_t>(space.size), 0.0F);
    pool.initialize_roots(policy.data(), space.size, space.size);
    const double wdl[3] = {0.4, 0.2, 0.4};
    pool.set_root_network_wdls(wdl, 3);
    pool.add_simulations_all(24);

    const int obs_size = pool.observation_size();
    std::vector<float> observations(static_cast<std::size_t>(obs_size));
    std::vector<int> players(1);
    std::vector<int> tokens(1);
    int steps = 0;
    while (pool.remaining_simulations() > 0) {
        const int n = pool.select_leaves(
            1, observations.data(), obs_size, players.data(), tokens.data());
        if (n == 0) {
            break;
        }
        require(n == 1, "one pending leaf");
        std::vector<float> leaf_policy(static_cast<std::size_t>(space.size), 0.0F);
        const double leaf_wdl[3] = {0.5, 0.0, 0.5};
        pool.evaluate_leaves(tokens.data(), n, leaf_policy.data(), space.size, space.size, leaf_wdl, 3);
        ++steps;
        require(steps <= 24, "simulation budget");
    }
    require(pool.remaining_simulations() == 0, "budget exhausted");
    require(pool.search(0).metrics().simulations == 24, "completed simulations");
    std::vector<std::uint32_t> visits(static_cast<std::size_t>(space.size), 0);
    pool.search(0).root_visits(visits.data(), space.size);
    std::uint32_t visit_sum = 0;
    for (std::uint32_t visit : visits) {
        visit_sum += visit;
    }
    require(visit_sum == 24, "root visits match simulations");
}

}  // namespace

int main() {
    test_maps_have_catanatron_sizes();
    test_initial_setup_completes();
    test_random_rollout_does_not_dead_end();
    test_observation_numeric_prefix();
    test_roll_chance_outcomes_sum_to_one();
    test_search_pool_runs_simulations();
    std::puts("ok");
    return 0;
}
