#include "cudanatron/batch.hpp"

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace cudanatron {
namespace {

void cuda_check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

int kernel_blocks(int count, int threads) {
    return (count + threads - 1) / threads;
}

__global__ void step_kernel(
    PackedMap* maps,
    PackedGame* games,
    const FlatActionSpace* space,
    const int* actions,
    int* previous_victory_points,
    double* previous_production,
    float* rewards,
    std::uint8_t* terminals,
    std::uint8_t* truncations,
    int num_envs,
    int num_players,
    int reward_function,
    int turns_limit,
    int victory_points_to_win,
    int* statuses) {
    const int env = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (env >= num_envs) {
        return;
    }

    PackedGame& game = games[env];
    const int current = game.current_player_index;
    const int action_index = actions[env * num_players + current];
    if (action_index < 0 || action_index >= space->size) {
        statuses[env] = static_cast<int>(Status::out_of_range);
        return;
    }
    const PackedAction action = decode_flat_action(*space, maps[env], game, action_index);
    const Status status = execute_action(maps[env], &game, action, nullptr);
    statuses[env] = static_cast<int>(status);
    if (status != Status::ok) {
        return;
    }

    const int winner = winning_player(game);
    const bool truncated = game.num_turns >= turns_limit;
    for (int player = 0; player < num_players; ++player) {
        const int row = env * num_players + player;
        float reward = 0.0F;
        if (winner >= 0 && winner == player) {
            reward = 1.0F;
        } else if (reward_function == static_cast<int>(RewardFunction::win) && winner >= 0) {
            reward = -1.0F;
        } else if (reward_function == static_cast<int>(RewardFunction::shaped)) {
            const double production = production_sum(maps[env], game, player);
            reward = static_cast<float>(
                0.01 *
                    static_cast<double>(
                        game.players[player].actual_victory_points - previous_victory_points[row]) /
                    static_cast<double>(victory_points_to_win) +
                0.0025 * (production - previous_production[row]));
            previous_victory_points[row] = game.players[player].actual_victory_points;
            previous_production[row] = production;
        }
        rewards[row] = reward;
        terminals[row] = winner >= 0 ? 1 : 0;
        truncations[row] = truncated ? 1 : 0;
    }
}

__global__ void write_kernel(
    const PackedMap* maps,
    const PackedGame* games,
    const FlatActionSpace* space,
    const ObservationLayout* layout,
    std::uint8_t* observations,
    std::size_t observation_row_stride,
    std::size_t action_mask_offset,
    std::size_t observation_offset,
    std::uint8_t* masks,
    int num_envs,
    int num_players,
    int turns_limit,
    int observation_size) {
    const int env = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (env >= num_envs) {
        return;
    }
    const PackedGame& game = games[env];
    const int winner = winning_player(game);
    const bool done = winner >= 0 || game.num_turns >= turns_limit;
    const int current = game.current_player_index;
    for (int player = 0; player < num_players; ++player) {
        const int row = env * num_players + player;
        std::uint8_t* row_bytes = observations + static_cast<std::size_t>(row) * observation_row_stride;
        if (done) {
            for (std::size_t i = 0; i < observation_row_stride; ++i) {
                row_bytes[i] = 0;
            }
            masks[row] = 0;
            continue;
        }
        masks[row] = 1;
        write_full_observation(
            maps[env],
            game,
            player,
            *layout,
            reinterpret_cast<float*>(row_bytes + observation_offset),
            observation_size);
        std::uint8_t* action_mask = row_bytes + action_mask_offset;
        if (player != current) {
            for (int i = 0; i < space->size; ++i) {
                action_mask[i] = 0;
            }
            if (space->size > 0) {
                action_mask[space->size - 1] = 1;
            }
        } else {
            write_legal_mask(maps[env], game, *space, action_mask, space->size);
        }
    }
}

}  // namespace

struct GameBatch::DeviceState {
    PackedMap* maps{nullptr};
    PackedGame* games{nullptr};
    FlatActionSpace* space{nullptr};
    ObservationLayout* layout{nullptr};
    int* actions{nullptr};
    int* previous_victory_points{nullptr};
    double* previous_production{nullptr};
    std::uint8_t* observations{nullptr};
    float* rewards{nullptr};
    std::uint8_t* terminals{nullptr};
    std::uint8_t* truncations{nullptr};
    std::uint8_t* masks{nullptr};
    int* statuses{nullptr};
    std::size_t observation_bytes{0};
    int rows{0};

    ~DeviceState() {
        cudaFree(maps);
        cudaFree(games);
        cudaFree(space);
        cudaFree(layout);
        cudaFree(actions);
        cudaFree(previous_victory_points);
        cudaFree(previous_production);
        cudaFree(observations);
        cudaFree(rewards);
        cudaFree(terminals);
        cudaFree(truncations);
        cudaFree(masks);
        cudaFree(statuses);
    }
};

GameBatch::GameBatch(BatchConfig config, ObservationLayout layout)
    : config_(config), layout_(layout) {
    if (config.num_envs <= 0) {
        throw std::invalid_argument("num_envs must be positive");
    }
    if (config.num_players < 1 || config.num_players > kMaxPlayers) {
        throw std::invalid_argument("num_players must be between one and four");
    }
    if (config.turns_limit <= 0) {
        throw std::invalid_argument("turns_limit must be positive");
    }

    PackedMap template_map{};
    if (build_packed_map(
            &template_map, config.map_type, 0, config.number_placement) != Status::ok) {
        throw std::runtime_error("failed to build batch template map");
    }
    if (build_flat_action_space(&action_space_, template_map, config.num_players) != Status::ok) {
        throw std::runtime_error("failed to build batch action space");
    }

    host_maps_.assign(static_cast<std::size_t>(config.num_envs), PackedMap{});
    host_games_.assign(static_cast<std::size_t>(config.num_envs), PackedGame{});
    previous_victory_points_.assign(
        static_cast<std::size_t>(config.num_envs * config.num_players), 0);
    previous_production_.assign(
        static_cast<std::size_t>(config.num_envs * config.num_players), 0.0);

    device_ = std::make_unique<DeviceState>();
    device_->rows = config.num_envs * config.num_players;
    cuda_check(cudaDeviceSetLimit(cudaLimitStackSize, 64 * 1024), "stack size");
    cuda_check(cudaMalloc(&device_->maps, sizeof(PackedMap) * config.num_envs), "maps");
    cuda_check(cudaMalloc(&device_->games, sizeof(PackedGame) * config.num_envs), "games");
    cuda_check(cudaMalloc(&device_->space, sizeof(FlatActionSpace)), "action space");
    cuda_check(cudaMalloc(&device_->layout, sizeof(ObservationLayout)), "layout");
    cuda_check(cudaMalloc(&device_->actions, sizeof(int) * device_->rows), "actions");
    cuda_check(
        cudaMalloc(&device_->previous_victory_points, sizeof(int) * device_->rows), "prev vp");
    cuda_check(
        cudaMalloc(&device_->previous_production, sizeof(double) * device_->rows), "prev production");
    cuda_check(cudaMalloc(&device_->rewards, sizeof(float) * device_->rows), "rewards");
    cuda_check(cudaMalloc(&device_->terminals, device_->rows), "terminals");
    cuda_check(cudaMalloc(&device_->truncations, device_->rows), "truncations");
    cuda_check(cudaMalloc(&device_->masks, device_->rows), "masks");
    cuda_check(cudaMalloc(&device_->statuses, sizeof(int) * config.num_envs), "statuses");
    cuda_check(
        cudaMemcpy(
            device_->space, &action_space_, sizeof(FlatActionSpace), cudaMemcpyHostToDevice),
        "upload action space");
    cuda_check(
        cudaMemcpy(device_->layout, &layout_, sizeof(ObservationLayout), cudaMemcpyHostToDevice),
        "upload layout");
}

GameBatch::~GameBatch() = default;

std::size_t GameBatch::row_index(int env_index, int player) const {
    return static_cast<std::size_t>(env_index * config_.num_players + player);
}

void GameBatch::validate_bound() const {
    if (buffers_.observations == nullptr) {
        throw std::logic_error("batch buffers have not been bound");
    }
}

void GameBatch::bind(BatchBuffers buffers) {
    if (buffers.observations == nullptr || buffers.actions == nullptr ||
        buffers.rewards == nullptr || buffers.terminals == nullptr ||
        buffers.truncations == nullptr || buffers.masks == nullptr) {
        throw std::invalid_argument("batch buffer pointer is null");
    }
    if (buffers.observation_row_stride == 0 ||
        buffers.observation_offset % alignof(float) != 0) {
        throw std::invalid_argument("invalid observation buffer layout");
    }
    buffers_ = buffers;
    device_->observation_bytes =
        buffers.observation_row_stride * static_cast<std::size_t>(device_->rows);
    if (device_->observations != nullptr) {
        cudaFree(device_->observations);
        device_->observations = nullptr;
    }
    cuda_check(cudaMalloc(&device_->observations, device_->observation_bytes), "observations");
}

void GameBatch::create_env(int env_index, std::uint64_t map_seed, std::uint64_t game_seed) {
    PackedMap& map = host_maps_[static_cast<std::size_t>(env_index)];
    PackedGame& game = host_games_[static_cast<std::size_t>(env_index)];
    GameConfig config{};
    config.num_players = config_.num_players;
    config.map_type = config_.map_type;
    config.number_placement = config_.number_placement;
    config.map_seed = map_seed;
    config.game_seed = game_seed;
    config.discard_limit = config_.discard_limit;
    config.friendly_robber = config_.friendly_robber;
    config.victory_points_to_win = config_.victory_points_to_win;
    if (build_packed_map(&map, config.map_type, map_seed, config.number_placement) != Status::ok) {
        throw std::runtime_error("batch map construction failed");
    }
    if (initialize_game(map, config, &game) != Status::ok) {
        throw std::runtime_error("batch game initialization failed");
    }
    for (int player = 0; player < config_.num_players; ++player) {
        const auto row = row_index(env_index, player);
        previous_victory_points_[row] = 0;
        previous_production_[row] = 0.0;
    }
}

void GameBatch::upload_env(int env_index) {
    cuda_check(
        cudaMemcpy(
            device_->maps + env_index,
            &host_maps_[static_cast<std::size_t>(env_index)],
            sizeof(PackedMap),
            cudaMemcpyHostToDevice),
        "upload map");
    cuda_check(
        cudaMemcpy(
            device_->games + env_index,
            &host_games_[static_cast<std::size_t>(env_index)],
            sizeof(PackedGame),
            cudaMemcpyHostToDevice),
        "upload game");
    cuda_check(
        cudaMemcpy(
            device_->previous_victory_points + row_index(env_index, 0),
            previous_victory_points_.data() + row_index(env_index, 0),
            sizeof(int) * config_.num_players,
            cudaMemcpyHostToDevice),
        "upload prev vp");
    cuda_check(
        cudaMemcpy(
            device_->previous_production + row_index(env_index, 0),
            previous_production_.data() + row_index(env_index, 0),
            sizeof(double) * config_.num_players,
            cudaMemcpyHostToDevice),
        "upload prev production");
}

void GameBatch::clear_transition(int env_index) {
    for (int player = 0; player < config_.num_players; ++player) {
        const auto index = row_index(env_index, player);
        buffers_.rewards[index] = 0.0F;
        buffers_.terminals[index] = 0;
        buffers_.truncations[index] = 0;
        buffers_.masks[index] = 1;
    }
}

void GameBatch::write_outputs() {
    constexpr int kThreads = 128;
    write_kernel<<<kernel_blocks(config_.num_envs, kThreads), kThreads>>>(
        device_->maps,
        device_->games,
        device_->space,
        device_->layout,
        device_->observations,
        buffers_.observation_row_stride,
        buffers_.action_mask_offset,
        buffers_.observation_offset,
        device_->masks,
        config_.num_envs,
        config_.num_players,
        config_.turns_limit,
        observation_size());
    cuda_check(cudaGetLastError(), "write kernel");
    cuda_check(cudaDeviceSynchronize(), "write sync");
    cuda_check(
        cudaMemcpy(
            buffers_.observations,
            device_->observations,
            device_->observation_bytes,
            cudaMemcpyDeviceToHost),
        "download observations");
    cuda_check(
        cudaMemcpy(
            buffers_.masks, device_->masks, device_->rows, cudaMemcpyDeviceToHost),
        "download masks");
}

void GameBatch::reset_all(
    const std::uint64_t* map_seeds,
    const std::uint64_t* game_seeds,
    std::size_t seed_count) {
    validate_bound();
    if (map_seeds == nullptr || game_seeds == nullptr ||
        seed_count != static_cast<std::size_t>(config_.num_envs)) {
        throw std::invalid_argument("batch reset seed array has incorrect size");
    }
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int env_index = 0; env_index < config_.num_envs; ++env_index) {
        create_env(env_index, map_seeds[env_index], game_seeds[env_index]);
    }
    cuda_check(
        cudaMemcpy(
            device_->maps,
            host_maps_.data(),
            sizeof(PackedMap) * config_.num_envs,
            cudaMemcpyHostToDevice),
        "upload maps");
    cuda_check(
        cudaMemcpy(
            device_->games,
            host_games_.data(),
            sizeof(PackedGame) * config_.num_envs,
            cudaMemcpyHostToDevice),
        "upload games");
    cuda_check(
        cudaMemcpy(
            device_->previous_victory_points,
            previous_victory_points_.data(),
            sizeof(int) * device_->rows,
            cudaMemcpyHostToDevice),
        "upload prev vp");
    cuda_check(
        cudaMemcpy(
            device_->previous_production,
            previous_production_.data(),
            sizeof(double) * device_->rows,
            cudaMemcpyHostToDevice),
        "upload prev production");
    for (int env_index = 0; env_index < config_.num_envs; ++env_index) {
        clear_transition(env_index);
    }
    write_outputs();
    std::memset(buffers_.rewards, 0, sizeof(float) * static_cast<std::size_t>(device_->rows));
    std::memset(buffers_.terminals, 0, static_cast<std::size_t>(device_->rows));
    std::memset(buffers_.truncations, 0, static_cast<std::size_t>(device_->rows));
}

void GameBatch::reset_at(
    int env_index,
    std::uint64_t map_seed,
    std::uint64_t game_seed,
    bool preserve_transition) {
    validate_bound();
    if (env_index < 0 || env_index >= config_.num_envs) {
        throw std::out_of_range("batch environment index is out of range");
    }
    create_env(env_index, map_seed, game_seed);
    upload_env(env_index);
    if (!preserve_transition) {
        clear_transition(env_index);
    } else {
        for (int player = 0; player < config_.num_players; ++player) {
            buffers_.masks[row_index(env_index, player)] = 1;
        }
    }
    write_outputs();
}

void GameBatch::step() {
    validate_bound();
    cuda_check(
        cudaMemcpy(
            device_->actions,
            buffers_.actions,
            sizeof(int) * device_->rows,
            cudaMemcpyHostToDevice),
        "upload actions");
    constexpr int kThreads = 128;
    step_kernel<<<kernel_blocks(config_.num_envs, kThreads), kThreads>>>(
        device_->maps,
        device_->games,
        device_->space,
        device_->actions,
        device_->previous_victory_points,
        device_->previous_production,
        device_->rewards,
        device_->terminals,
        device_->truncations,
        config_.num_envs,
        config_.num_players,
        static_cast<int>(config_.reward_function),
        config_.turns_limit,
        config_.victory_points_to_win,
        device_->statuses);
    cuda_check(cudaGetLastError(), "step kernel");
    cuda_check(cudaDeviceSynchronize(), "step sync");

    std::vector<int> statuses(static_cast<std::size_t>(config_.num_envs));
    cuda_check(
        cudaMemcpy(
            statuses.data(),
            device_->statuses,
            sizeof(int) * config_.num_envs,
            cudaMemcpyDeviceToHost),
        "download statuses");
    for (int env_index = 0; env_index < config_.num_envs; ++env_index) {
        if (statuses[static_cast<std::size_t>(env_index)] != static_cast<int>(Status::ok)) {
            throw std::runtime_error(
                "invalid action for environment " + std::to_string(env_index));
        }
    }
    write_outputs();
    cuda_check(
        cudaMemcpy(
            buffers_.rewards,
            device_->rewards,
            sizeof(float) * device_->rows,
            cudaMemcpyDeviceToHost),
        "download rewards");
    cuda_check(
        cudaMemcpy(
            buffers_.terminals,
            device_->terminals,
            device_->rows,
            cudaMemcpyDeviceToHost),
        "download terminals");
    cuda_check(
        cudaMemcpy(
            buffers_.truncations,
            device_->truncations,
            device_->rows,
            cudaMemcpyDeviceToHost),
        "download truncations");
}

}  // namespace cudanatron
