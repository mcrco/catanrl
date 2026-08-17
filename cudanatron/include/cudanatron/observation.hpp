#pragma once

#include "cudanatron/game.hpp"

namespace cudanatron {

struct NodePosition {
    int node{};
    int x{};
    int y{};
};

struct EdgePosition {
    int a{};
    int b{};
    int x{};
    int y{};
};

struct TilePosition {
    int x{};
    int y{};
    int z{};
    int board_x{};
    int board_y{};
};

struct ObservationLayout {
    int width{kBoardWidth};
    int height{kBoardHeight};
    int node_x[kMaxNodes]{};
    int node_y[kMaxNodes]{};
    std::uint8_t has_node[kMaxNodes]{};
    int edge_x[kMaxEdges]{};
    int edge_y[kMaxEdges]{};
    std::uint8_t has_edge[kMaxEdges]{};
    int tile_x[kMaxTiles]{};
    int tile_y[kMaxTiles]{};
    std::uint8_t has_tile[kMaxTiles]{};
};

CUDANATRON_HD inline int board_channel_count(int num_players) {
    return 2 * num_players + kBoardChannelsWithoutPlayers;
}

CUDANATRON_HD inline int full_numeric_observation_size(int num_players) {
    return kPlayerFullFeatureCount * num_players + kSharedNumericFeatureCount;
}

CUDANATRON_HD inline int board_observation_size(int num_players, int width, int height) {
    return width * height * board_channel_count(num_players);
}

CUDANATRON_HD inline int full_observation_size(int num_players, int width, int height) {
    return full_numeric_observation_size(num_players) +
           board_observation_size(num_players, width, height);
}

Status fill_observation_layout(
    ObservationLayout* layout,
    int width,
    int height,
    const NodePosition* nodes,
    int node_count,
    const EdgePosition* edges,
    int edge_count,
    const TilePosition* tiles,
    int tile_count,
    const PackedMap& map);

CUDANATRON_HD Status write_full_observation(
    const PackedMap& map,
    const PackedGame& game,
    int base_player,
    const ObservationLayout& layout,
    float* output,
    int output_size);

CUDANATRON_HD double production_sum(const PackedMap& map, const PackedGame& game, int player);

}  // namespace cudanatron
