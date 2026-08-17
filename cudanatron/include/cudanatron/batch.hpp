#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "cudanatron/action_space.hpp"
#include "cudanatron/game.hpp"
#include "cudanatron/observation.hpp"

namespace cudanatron {

enum class RewardFunction : std::int32_t {
    shaped = 0,
    win = 1,
};

struct BatchConfig {
    int num_envs{1};
    int num_players{2};
    MapType map_type{MapType::base};
    NumberPlacement number_placement{NumberPlacement::random};
    int discard_limit{7};
    bool friendly_robber{false};
    int victory_points_to_win{10};
    RewardFunction reward_function{RewardFunction::shaped};
    int turns_limit{1000};
};

struct BatchBuffers {
    std::uint8_t* observations{nullptr};
    std::size_t observation_row_stride{0};
    std::size_t action_mask_offset{0};
    std::size_t observation_offset{0};
    std::int32_t* actions{nullptr};
    float* rewards{nullptr};
    std::uint8_t* terminals{nullptr};
    std::uint8_t* truncations{nullptr};
    std::uint8_t* masks{nullptr};
};

class GameBatch {
public:
    GameBatch(BatchConfig config, ObservationLayout layout);
    ~GameBatch();

    GameBatch(const GameBatch&) = delete;
    GameBatch& operator=(const GameBatch&) = delete;

    void bind(BatchBuffers buffers);
    void reset_all(
        const std::uint64_t* map_seeds,
        const std::uint64_t* game_seeds,
        std::size_t seed_count);
    void reset_at(
        int env_index,
        std::uint64_t map_seed,
        std::uint64_t game_seed,
        bool preserve_transition);
    void step();

    int num_envs() const { return config_.num_envs; }
    int num_players() const { return config_.num_players; }
    int action_space_size() const { return action_space_.size; }
    int observation_size() const {
        return full_observation_size(config_.num_players, layout_.width, layout_.height);
    }

private:
    struct DeviceState;

    std::size_t row_index(int env_index, int player) const;
    void create_env(int env_index, std::uint64_t map_seed, std::uint64_t game_seed);
    void upload_env(int env_index);
    void write_outputs();
    void clear_transition(int env_index);
    void validate_bound() const;

    BatchConfig config_{};
    ObservationLayout layout_{};
    FlatActionSpace action_space_{};
    BatchBuffers buffers_{};
    std::vector<PackedMap> host_maps_{};
    std::vector<PackedGame> host_games_{};
    std::vector<int> previous_victory_points_{};
    std::vector<double> previous_production_{};
    std::unique_ptr<DeviceState> device_{};
};

}  // namespace cudanatron
