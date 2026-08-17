#pragma once

#include <cstdint>

#include "cudanatron/types.hpp"

namespace cudanatron {

struct PackedTile {
    Coordinate coordinate{};
    TileKind kind{TileKind::water};
    std::int8_t id{-1};
    std::int8_t resource{-1};
    std::int8_t number{-1};
    std::int8_t port_direction{-1};
    std::int16_t nodes[6]{};
};

struct PackedMap {
    MapType map_type{MapType::base};
    NumberPlacement number_placement{NumberPlacement::official_spiral};
    std::uint8_t num_tiles{0};
    std::uint8_t num_land_tiles{0};
    std::uint8_t num_nodes{0};
    std::uint8_t num_edges{0};
    std::uint8_t desert_tile{0};
    PackedTile tiles[kMaxTiles]{};
    std::int16_t land_tile_indices[kMaxLandTiles]{};
    std::int16_t edge_a[kMaxEdges]{};
    std::int16_t edge_b[kMaxEdges]{};
    std::int16_t node_neighbors[kMaxNodes][kMaxNodeDegree]{};
    std::uint8_t node_neighbor_count[kMaxNodes]{};
    std::int16_t node_edge[kMaxNodes][kMaxNodeDegree]{};
};

// Host-only: shuffle terrain/ports/numbers and assign Catanatron node IDs.
Status build_packed_map(
    PackedMap* map,
    MapType map_type,
    std::uint64_t seed,
    NumberPlacement number_placement);

CUDANATRON_HD inline int find_tile_index(const PackedMap& map, Coordinate coordinate) {
    for (int i = 0; i < map.num_tiles; ++i) {
        if (map.tiles[i].coordinate == coordinate) {
            return i;
        }
    }
    return -1;
}

CUDANATRON_HD inline int find_land_tile_index(const PackedMap& map, Coordinate coordinate) {
    for (int i = 0; i < map.num_land_tiles; ++i) {
        const int tile_index = map.land_tile_indices[i];
        if (map.tiles[tile_index].coordinate == coordinate) {
            return tile_index;
        }
    }
    return -1;
}

CUDANATRON_HD inline int find_edge(const PackedMap& map, int a, int b) {
    if (a > b) {
        const int tmp = a;
        a = b;
        b = tmp;
    }
    for (int i = 0; i < map.num_edges; ++i) {
        if (map.edge_a[i] == a && map.edge_b[i] == b) {
            return i;
        }
    }
    return -1;
}

}  // namespace cudanatron
