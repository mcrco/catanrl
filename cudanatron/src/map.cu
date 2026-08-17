#include "cudanatron/map.hpp"

#include <algorithm>
#include <array>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <tuple>
#include <vector>

namespace cudanatron {
namespace {

using TopologyEntry = std::tuple<Coordinate, TileKind, std::optional<Direction>>;

constexpr std::array<int, 18> kBaseNumbersInSpiralOrder{
    5, 2, 6, 3, 8, 10, 9, 12, 11, 4, 8, 10, 9, 4, 5, 6, 3, 11,
};

constexpr std::size_t index(NodeRef ref) {
    return static_cast<std::size_t>(ref);
}

constexpr std::size_t index(EdgeRef ref) {
    return static_cast<std::size_t>(ref);
}

std::pair<NodeRef, NodeRef> edge_nodes(EdgeRef edge_ref) {
    switch (edge_ref) {
        case EdgeRef::east:
            return {NodeRef::northeast, NodeRef::southeast};
        case EdgeRef::southeast:
            return {NodeRef::southeast, NodeRef::south};
        case EdgeRef::southwest:
            return {NodeRef::south, NodeRef::southwest};
        case EdgeRef::west:
            return {NodeRef::southwest, NodeRef::northwest};
        case EdgeRef::northwest:
            return {NodeRef::northwest, NodeRef::north};
        case EdgeRef::northeast:
            return {NodeRef::north, NodeRef::northeast};
    }
    return {NodeRef::north, NodeRef::northeast};
}

struct NodeEdgeResult {
    std::array<int, 6> nodes{};
    std::array<std::pair<int, int>, 6> edges{};
    int next_node_id{};
};

NodeEdgeResult get_nodes_and_edges(
    const std::vector<PackedTile>& tiles,
    const std::map<std::tuple<int, int, int>, int>& coordinate_to_tile,
    Coordinate coordinate,
    int next_node_id) {
    std::array<int, 6> nodes{};
    nodes.fill(-1);
    std::array<std::pair<int, int>, 6> edges{};
    edges.fill({-1, -1});

    for (Direction direction : {
             Direction::east,
             Direction::southeast,
             Direction::southwest,
             Direction::west,
             Direction::northwest,
             Direction::northeast,
         }) {
        const Coordinate neighbor_coord = coordinate + direction_vector(direction);
        const auto key = std::tuple<int, int, int>{
            neighbor_coord.x, neighbor_coord.y, neighbor_coord.z};
        const auto neighbor_it = coordinate_to_tile.find(key);
        if (neighbor_it == coordinate_to_tile.end()) {
            continue;
        }
        const PackedTile& neighbor = tiles[static_cast<std::size_t>(neighbor_it->second)];
        switch (direction) {
            case Direction::east:
                nodes[index(NodeRef::northeast)] = neighbor.nodes[index(NodeRef::northwest)];
                nodes[index(NodeRef::southeast)] = neighbor.nodes[index(NodeRef::southwest)];
                edges[index(EdgeRef::east)] = {
                    neighbor.nodes[index(NodeRef::northwest)],
                    neighbor.nodes[index(NodeRef::southwest)],
                };
                break;
            case Direction::southeast:
                nodes[index(NodeRef::south)] = neighbor.nodes[index(NodeRef::northwest)];
                nodes[index(NodeRef::southeast)] = neighbor.nodes[index(NodeRef::north)];
                edges[index(EdgeRef::southeast)] = {
                    neighbor.nodes[index(NodeRef::northwest)],
                    neighbor.nodes[index(NodeRef::north)],
                };
                break;
            case Direction::southwest:
                nodes[index(NodeRef::south)] = neighbor.nodes[index(NodeRef::northeast)];
                nodes[index(NodeRef::southwest)] = neighbor.nodes[index(NodeRef::north)];
                edges[index(EdgeRef::southwest)] = {
                    neighbor.nodes[index(NodeRef::northeast)],
                    neighbor.nodes[index(NodeRef::north)],
                };
                break;
            case Direction::west:
                nodes[index(NodeRef::northwest)] = neighbor.nodes[index(NodeRef::northeast)];
                nodes[index(NodeRef::southwest)] = neighbor.nodes[index(NodeRef::southeast)];
                edges[index(EdgeRef::west)] = {
                    neighbor.nodes[index(NodeRef::northeast)],
                    neighbor.nodes[index(NodeRef::southeast)],
                };
                break;
            case Direction::northwest:
                nodes[index(NodeRef::north)] = neighbor.nodes[index(NodeRef::southeast)];
                nodes[index(NodeRef::northwest)] = neighbor.nodes[index(NodeRef::south)];
                edges[index(EdgeRef::northwest)] = {
                    neighbor.nodes[index(NodeRef::southeast)],
                    neighbor.nodes[index(NodeRef::south)],
                };
                break;
            case Direction::northeast:
                nodes[index(NodeRef::north)] = neighbor.nodes[index(NodeRef::southwest)];
                nodes[index(NodeRef::northeast)] = neighbor.nodes[index(NodeRef::south)];
                edges[index(EdgeRef::northeast)] = {
                    neighbor.nodes[index(NodeRef::southwest)],
                    neighbor.nodes[index(NodeRef::south)],
                };
                break;
        }
    }

    for (int& node : nodes) {
        if (node < 0) {
            node = next_node_id++;
        }
    }
    for (EdgeRef ref : {
             EdgeRef::east,
             EdgeRef::southeast,
             EdgeRef::southwest,
             EdgeRef::west,
             EdgeRef::northwest,
             EdgeRef::northeast,
         }) {
        auto& edge = edges[index(ref)];
        if (edge.first < 0) {
            const auto [a_ref, b_ref] = edge_nodes(ref);
            edge = {nodes[index(a_ref)], nodes[index(b_ref)]};
        }
    }
    return {nodes, edges, next_node_id};
}

std::vector<Coordinate> land_coordinates(bool mini) {
    std::vector<Coordinate> result{
        {0, 0, 0},
        {1, -1, 0},
        {0, -1, 1},
        {-1, 0, 1},
        {-1, 1, 0},
        {0, 1, -1},
        {1, 0, -1},
    };
    if (!mini) {
        result.insert(
            result.end(),
            {
                {2, -2, 0},
                {1, -2, 1},
                {0, -2, 2},
                {-1, -1, 2},
                {-2, 0, 2},
                {-2, 1, 1},
                {-2, 2, 0},
                {-1, 2, -1},
                {0, 2, -2},
                {1, 1, -2},
                {2, 0, -2},
                {2, -1, -1},
            });
    }
    return result;
}

std::vector<TopologyEntry> make_topology(bool mini) {
    std::vector<TopologyEntry> result;
    for (Coordinate coordinate : land_coordinates(mini)) {
        result.emplace_back(coordinate, TileKind::land, std::nullopt);
    }
    if (mini) {
        for (Coordinate coordinate : {
                 Coordinate{2, -2, 0},
                 {1, -2, 1},
                 {0, -2, 2},
                 {-1, -1, 2},
                 {-2, 0, 2},
                 {-2, 1, 1},
                 {-2, 2, 0},
                 {-1, 2, -1},
                 {0, 2, -2},
                 {1, 1, -2},
                 {2, 0, -2},
                 {2, -1, -1},
             }) {
            result.emplace_back(coordinate, TileKind::water, std::nullopt);
        }
        return result;
    }

    const std::array<TopologyEntry, 18> water_ring{{
        {{3, -3, 0}, TileKind::port, Direction::west},
        {{2, -3, 1}, TileKind::water, std::nullopt},
        {{1, -3, 2}, TileKind::port, Direction::northwest},
        {{0, -3, 3}, TileKind::water, std::nullopt},
        {{-1, -2, 3}, TileKind::port, Direction::northwest},
        {{-2, -1, 3}, TileKind::water, std::nullopt},
        {{-3, 0, 3}, TileKind::port, Direction::northeast},
        {{-3, 1, 2}, TileKind::water, std::nullopt},
        {{-3, 2, 1}, TileKind::port, Direction::east},
        {{-3, 3, 0}, TileKind::water, std::nullopt},
        {{-2, 3, -1}, TileKind::port, Direction::east},
        {{-1, 3, -2}, TileKind::water, std::nullopt},
        {{0, 3, -3}, TileKind::port, Direction::southeast},
        {{1, 2, -3}, TileKind::water, std::nullopt},
        {{2, 1, -3}, TileKind::port, Direction::southwest},
        {{3, 0, -3}, TileKind::water, std::nullopt},
        {{3, -1, -2}, TileKind::port, Direction::southwest},
        {{3, -2, -1}, TileKind::water, std::nullopt},
    }};
    result.insert(result.end(), water_ring.begin(), water_ring.end());
    return result;
}

std::vector<Coordinate> official_spiral_coordinates(bool mini) {
    if (mini) {
        return {
            {1, -1, 0},
            {1, 0, -1},
            {0, 1, -1},
            {-1, 1, 0},
            {-1, 0, 1},
            {0, -1, 1},
            {0, 0, 0},
        };
    }
    return {
        {2, -2, 0},
        {2, -1, -1},
        {2, 0, -2},
        {1, 1, -2},
        {0, 2, -2},
        {-1, 2, -1},
        {-2, 2, 0},
        {-2, 1, 1},
        {-2, 0, 2},
        {-1, -1, 2},
        {0, -2, 2},
        {1, -2, 1},
        {1, -1, 0},
        {1, 0, -1},
        {0, 1, -1},
        {-1, 1, 0},
        {-1, 0, 1},
        {0, -1, 1},
        {0, 0, 0},
    };
}

int resource_or_neg(std::optional<Resource> resource) {
    return resource.has_value() ? static_cast<int>(*resource) : -1;
}

}  // namespace

Status build_packed_map(
    PackedMap* map,
    MapType map_type,
    std::uint64_t seed,
    NumberPlacement number_placement) {
    if (map == nullptr) {
        return Status::invalid_argument;
    }
    *map = PackedMap{};
    map->map_type = map_type;
    map->number_placement = number_placement;

    const bool mini = map_type == MapType::mini;
    const auto topology = make_topology(mini);

    std::vector<std::optional<Resource>> resources;
    std::vector<int> numbers;
    std::vector<std::optional<Resource>> ports;
    if (mini) {
        resources = {
            Resource::wood,
            std::nullopt,
            Resource::brick,
            Resource::sheep,
            Resource::wheat,
            Resource::wheat,
            Resource::ore,
        };
        numbers = {3, 4, 5, 6, 8, 9, 10};
    } else {
        resources = {
            Resource::wood,
            Resource::wood,
            Resource::wood,
            Resource::wood,
            Resource::brick,
            Resource::brick,
            Resource::brick,
            Resource::sheep,
            Resource::sheep,
            Resource::sheep,
            Resource::sheep,
            Resource::wheat,
            Resource::wheat,
            Resource::wheat,
            Resource::wheat,
            Resource::ore,
            Resource::ore,
            Resource::ore,
            std::nullopt,
        };
        numbers = {2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12};
        ports = {
            Resource::wood,
            Resource::brick,
            Resource::sheep,
            Resource::wheat,
            Resource::ore,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
        };
    }

    std::mt19937_64 random(seed);
    std::shuffle(resources.begin(), resources.end(), random);
    std::shuffle(numbers.begin(), numbers.end(), random);
    std::shuffle(ports.begin(), ports.end(), random);

    std::vector<PackedTile> tiles;
    std::map<std::tuple<int, int, int>, int> coordinate_to_tile;
    int next_node_id = 0;
    int next_land_id = 0;
    int next_port_id = 0;
    std::vector<int> land_tile_indices;

    for (const auto& [coordinate, kind, port_direction] : topology) {
        NodeEdgeResult node_edges =
            get_nodes_and_edges(tiles, coordinate_to_tile, coordinate, next_node_id);
        next_node_id = node_edges.next_node_id;

        PackedTile tile{};
        tile.coordinate = coordinate;
        tile.kind = kind;
        tile.port_direction =
            port_direction.has_value() ? static_cast<std::int8_t>(*port_direction) : -1;
        for (int i = 0; i < 6; ++i) {
            tile.nodes[i] = static_cast<std::int16_t>(node_edges.nodes[static_cast<std::size_t>(i)]);
        }
        if (kind == TileKind::land) {
            tile.id = static_cast<std::int8_t>(next_land_id++);
            const auto resource = resources.back();
            resources.pop_back();
            tile.resource = static_cast<std::int8_t>(resource_or_neg(resource));
            if (resource.has_value()) {
                tile.number = static_cast<std::int8_t>(numbers.back());
                numbers.pop_back();
            }
        } else if (kind == TileKind::port) {
            tile.id = static_cast<std::int8_t>(next_port_id++);
            tile.resource = static_cast<std::int8_t>(resource_or_neg(ports.back()));
            ports.pop_back();
        }

        const int tile_index = static_cast<int>(tiles.size());
        coordinate_to_tile.emplace(
            std::tuple<int, int, int>{coordinate.x, coordinate.y, coordinate.z},
            tile_index);
        tiles.push_back(tile);
        if (kind == TileKind::land) {
            land_tile_indices.push_back(tile_index);
        }
    }

    if (number_placement == NumberPlacement::official_spiral) {
        std::size_t number_index = 0;
        for (Coordinate coordinate : official_spiral_coordinates(mini)) {
            const auto key =
                std::tuple<int, int, int>{coordinate.x, coordinate.y, coordinate.z};
            PackedTile& tile = tiles[static_cast<std::size_t>(coordinate_to_tile.at(key))];
            if (tile.resource < 0) {
                tile.number = -1;
                continue;
            }
            tile.number = static_cast<std::int8_t>(kBaseNumbersInSpiralOrder[number_index++]);
        }
    }

    if (map_type == MapType::tournament) {
        constexpr std::array<std::int8_t, 19> kTournamentResources{
            -1, 4, 0, 4, 1, 4, 3, 3, 2, 1, 2, 1, 3, 0, 3, 0, 2, 2, 0,
        };
        constexpr std::array<std::int8_t, 19> kTournamentNumbers{
            -1, 6, 3, 11, 9, 4, 5, 9, 12, 11, 4, 8, 10, 5, 2, 6, 3, 8, 10,
        };
        constexpr std::array<std::int8_t, 9> kTournamentPorts{
            -1, 1, 0, -1, 3, 4, -1, 2, -1,
        };
        for (PackedTile& tile : tiles) {
            if (tile.kind == TileKind::land) {
                tile.resource = kTournamentResources.at(static_cast<std::size_t>(tile.id));
                tile.number = kTournamentNumbers.at(static_cast<std::size_t>(tile.id));
            } else if (tile.kind == TileKind::port) {
                tile.resource = kTournamentPorts.at(static_cast<std::size_t>(tile.id));
            }
        }
    }

    std::set<std::pair<int, int>> unique_edges;
    int desert_tile = -1;
    int max_land_node = -1;
    for (int tile_index : land_tile_indices) {
        const PackedTile& tile = tiles[static_cast<std::size_t>(tile_index)];
        if (tile.resource < 0 && desert_tile < 0) {
            desert_tile = tile_index;
        }
        for (int n = 0; n < 6; ++n) {
            max_land_node = std::max(max_land_node, static_cast<int>(tile.nodes[n]));
            const int a = tile.nodes[n];
            const int b = tile.nodes[(n + 1) % 6];
            unique_edges.emplace(std::min(a, b), std::max(a, b));
        }
    }
    if (desert_tile < 0) {
        return Status::logic_error;
    }

    const int num_land_nodes = max_land_node + 1;
    if (tiles.size() > static_cast<std::size_t>(kMaxTiles) ||
        land_tile_indices.size() > static_cast<std::size_t>(kMaxLandTiles) ||
        unique_edges.size() > static_cast<std::size_t>(kMaxEdges) ||
        num_land_nodes > kMaxNodes) {
        return Status::logic_error;
    }

    map->num_tiles = static_cast<std::uint8_t>(tiles.size());
    map->num_land_tiles = static_cast<std::uint8_t>(land_tile_indices.size());
    map->num_nodes = static_cast<std::uint8_t>(num_land_nodes);
    map->num_edges = static_cast<std::uint8_t>(unique_edges.size());
    map->desert_tile = static_cast<std::uint8_t>(desert_tile);
    for (std::size_t i = 0; i < tiles.size(); ++i) {
        map->tiles[i] = tiles[i];
    }
    for (std::size_t i = 0; i < land_tile_indices.size(); ++i) {
        map->land_tile_indices[i] = static_cast<std::int16_t>(land_tile_indices[i]);
    }
    int edge_index = 0;
    for (const auto& [a, b] : unique_edges) {
        map->edge_a[edge_index] = static_cast<std::int16_t>(a);
        map->edge_b[edge_index] = static_cast<std::int16_t>(b);
        ++edge_index;
    }

    for (int node = 0; node < map->num_nodes; ++node) {
        map->node_neighbor_count[node] = 0;
        for (int d = 0; d < kMaxNodeDegree; ++d) {
            map->node_neighbors[node][d] = -1;
            map->node_edge[node][d] = -1;
        }
    }
    for (int e = 0; e < map->num_edges; ++e) {
        const int a = map->edge_a[e];
        const int b = map->edge_b[e];
        auto add_adj = [&](int from, int to) {
            const int count = map->node_neighbor_count[from];
            map->node_neighbors[from][count] = static_cast<std::int16_t>(to);
            map->node_edge[from][count] = static_cast<std::int16_t>(e);
            ++map->node_neighbor_count[from];
        };
        add_adj(a, b);
        add_adj(b, a);
    }
    return Status::ok;
}

}  // namespace cudanatron
