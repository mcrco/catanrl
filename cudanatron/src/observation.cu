#include "cudanatron/observation.hpp"

namespace cudanatron {
namespace {

CUDANATRON_HD int resource_count(const PackedPlayer& state, Resource resource) {
    return state.resources[resource_index(resource)];
}

CUDANATRON_HD int card_count(const PackedPlayer& state, DevelopmentCard card) {
    return state.development_cards[static_cast<int>(card)];
}

CUDANATRON_HD int played_card_count(const PackedPlayer& state, DevelopmentCard card) {
    return state.played_development_cards[static_cast<int>(card)];
}

CUDANATRON_HD int playable_card(const PackedPlayer& state, DevelopmentCard card) {
    const int index = static_cast<int>(card);
    return state.development_cards[index] > 0 &&
                   ((state.development_card_owned_at_start >> index) & 1) != 0 &&
                   state.has_played_development_card_in_turn == 0
               ? 1
               : 0;
}

CUDANATRON_HD int total_resources(const PackedPlayer& state) {
    int total = 0;
    for (int i = 0; i < kResourceCount; ++i) {
        total += state.resources[i];
    }
    return total;
}

CUDANATRON_HD int total_development_cards(const PackedPlayer& state) {
    int total = 0;
    for (int i = 0; i < kDevCardCount; ++i) {
        total += state.development_cards[i];
    }
    return total;
}

CUDANATRON_HD float dice_probability(int number) {
    const int distance = number > 7 ? number - 7 : 7 - number;
    return static_cast<float>(6 - distance) / 36.0F;
}

CUDANATRON_HD int board_tensor_index(
    int numeric_size,
    int height,
    int channels,
    int x,
    int y,
    int channel) {
    return numeric_size + (x * height + y) * channels + channel;
}

CUDANATRON_HD void write_full_player_features(
    const PackedGame& game,
    const PackedPlayer& state,
    float*& output) {
    const int completed = game.completed_turns;
    *output++ = static_cast<float>(state.actual_victory_points);
    *output++ = static_cast<float>(resource_count(state, Resource::brick));
    *output++ = static_cast<float>(state.cities_available);
    *output++ = static_cast<float>(state.has_army);
    *output++ = static_cast<float>(state.has_played_development_card_in_turn);
    *output++ = static_cast<float>(state.has_road);
    *output++ = static_cast<float>(state.has_rolled);
    *output++ = static_cast<float>(card_count(state, DevelopmentCard::knight));
    *output++ = static_cast<float>(playable_card(state, DevelopmentCard::knight));
    *output++ = static_cast<float>(played_card_count(state, DevelopmentCard::knight));
    *output++ = static_cast<float>(state.longest_road_length);
    *output++ = static_cast<float>(card_count(state, DevelopmentCard::monopoly));
    *output++ = static_cast<float>(playable_card(state, DevelopmentCard::monopoly));
    *output++ = static_cast<float>(played_card_count(state, DevelopmentCard::monopoly));
    *output++ = static_cast<float>(total_development_cards(state));
    *output++ = static_cast<float>(total_resources(state));
    *output++ = static_cast<float>(resource_count(state, Resource::ore));
    *output++ = static_cast<float>(state.victory_points);
    *output++ = static_cast<float>(state.roads_available);
    *output++ = static_cast<float>(card_count(state, DevelopmentCard::road_building));
    *output++ = static_cast<float>(playable_card(state, DevelopmentCard::road_building));
    *output++ = static_cast<float>(played_card_count(state, DevelopmentCard::road_building));
    *output++ = static_cast<float>(state.settlements_available);
    *output++ = static_cast<float>(resource_count(state, Resource::sheep));
    *output++ = static_cast<float>(
        turns_since(completed, state.last_dev_bought_completed_turn));
    *output++ = static_cast<float>(turns_since(completed, state.last_knight_completed_turn));
    *output++ = static_cast<float>(card_count(state, DevelopmentCard::victory_point));
    *output++ = static_cast<float>(resource_count(state, Resource::wheat));
    *output++ = static_cast<float>(resource_count(state, Resource::wood));
    *output++ = static_cast<float>(card_count(state, DevelopmentCard::year_of_plenty));
    *output++ = static_cast<float>(playable_card(state, DevelopmentCard::year_of_plenty));
    *output++ = static_cast<float>(played_card_count(state, DevelopmentCard::year_of_plenty));
}

CUDANATRON_HD void write_public_player_features(
    const PackedGame& game,
    const PackedPlayer& state,
    float*& output) {
    const int completed = game.completed_turns;
    *output++ = static_cast<float>(state.cities_available);
    *output++ = static_cast<float>(state.has_army);
    *output++ = static_cast<float>(state.has_road);
    *output++ = static_cast<float>(state.has_rolled);
    *output++ = static_cast<float>(played_card_count(state, DevelopmentCard::knight));
    *output++ = static_cast<float>(state.longest_road_length);
    *output++ = static_cast<float>(played_card_count(state, DevelopmentCard::monopoly));
    *output++ = static_cast<float>(total_development_cards(state));
    *output++ = static_cast<float>(total_resources(state));
    *output++ = static_cast<float>(state.victory_points);
    *output++ = static_cast<float>(state.roads_available);
    *output++ = static_cast<float>(played_card_count(state, DevelopmentCard::road_building));
    *output++ = static_cast<float>(state.settlements_available);
    *output++ = static_cast<float>(
        turns_since(completed, state.last_dev_bought_completed_turn));
    *output++ = static_cast<float>(turns_since(completed, state.last_knight_completed_turn));
    *output++ = static_cast<float>(played_card_count(state, DevelopmentCard::year_of_plenty));
}

CUDANATRON_HD void write_private_player_features(const PackedPlayer& state, float*& output) {
    *output++ = static_cast<float>(state.actual_victory_points);
    *output++ = static_cast<float>(resource_count(state, Resource::brick));
    *output++ = static_cast<float>(state.has_played_development_card_in_turn);
    *output++ = static_cast<float>(card_count(state, DevelopmentCard::knight));
    *output++ = static_cast<float>(playable_card(state, DevelopmentCard::knight));
    *output++ = static_cast<float>(card_count(state, DevelopmentCard::monopoly));
    *output++ = static_cast<float>(playable_card(state, DevelopmentCard::monopoly));
    *output++ = static_cast<float>(resource_count(state, Resource::ore));
    *output++ = static_cast<float>(card_count(state, DevelopmentCard::road_building));
    *output++ = static_cast<float>(playable_card(state, DevelopmentCard::road_building));
    *output++ = static_cast<float>(resource_count(state, Resource::sheep));
    *output++ = static_cast<float>(card_count(state, DevelopmentCard::victory_point));
    *output++ = static_cast<float>(resource_count(state, Resource::wheat));
    *output++ = static_cast<float>(resource_count(state, Resource::wood));
    *output++ = static_cast<float>(card_count(state, DevelopmentCard::year_of_plenty));
    *output++ = static_cast<float>(playable_card(state, DevelopmentCard::year_of_plenty));
}

}  // namespace

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
    const PackedMap& map) {
    if (layout == nullptr || width <= 0 || height <= 0) {
        return Status::invalid_argument;
    }
    *layout = ObservationLayout{};
    layout->width = width;
    layout->height = height;
    for (int i = 0; i < node_count; ++i) {
        const int node = nodes[i].node;
        if (node < 0 || node >= kMaxNodes) {
            continue;
        }
        layout->node_x[node] = nodes[i].x;
        layout->node_y[node] = nodes[i].y;
        layout->has_node[node] = 1;
    }
    for (int i = 0; i < edge_count; ++i) {
        const int edge = find_edge(map, edges[i].a, edges[i].b);
        if (edge < 0 || edge >= kMaxEdges) {
            continue;
        }
        layout->edge_x[edge] = edges[i].x;
        layout->edge_y[edge] = edges[i].y;
        layout->has_edge[edge] = 1;
    }
    for (int i = 0; i < tile_count; ++i) {
        const int tile = find_tile_index(
            map,
            Coordinate{
                static_cast<std::int8_t>(tiles[i].x),
                static_cast<std::int8_t>(tiles[i].y),
                static_cast<std::int8_t>(tiles[i].z),
            });
        if (tile < 0 || tile >= kMaxTiles) {
            continue;
        }
        layout->tile_x[tile] = tiles[i].board_x;
        layout->tile_y[tile] = tiles[i].board_y;
        layout->has_tile[tile] = 1;
    }
    return Status::ok;
}

CUDANATRON_HD Status write_full_observation(
    const PackedMap& map,
    const PackedGame& game,
    int base_player,
    const ObservationLayout& layout,
    float* output,
    int output_size) {
    const int num_players = game.num_players;
    if (output == nullptr || num_players < 1 || num_players > kMaxPlayers) {
        return Status::invalid_argument;
    }
    if (base_player < 0 || base_player >= num_players) {
        return Status::out_of_range;
    }
    const int expected = full_observation_size(num_players, layout.width, layout.height);
    if (output_size != expected) {
        return Status::invalid_argument;
    }
    for (int i = 0; i < output_size; ++i) {
        output[i] = 0.0F;
    }

    float* cursor = output;
    *cursor++ = static_cast<float>(game.resource_bank[resource_index(Resource::brick)]);
    *cursor++ = static_cast<float>(game.development_deck_size);
    *cursor++ = static_cast<float>(game.resource_bank[resource_index(Resource::ore)]);
    *cursor++ = static_cast<float>(game.resource_bank[resource_index(Resource::sheep)]);
    *cursor++ = static_cast<float>(game.resource_bank[resource_index(Resource::wheat)]);
    *cursor++ = static_cast<float>(game.resource_bank[resource_index(Resource::wood)]);
    *cursor++ = static_cast<float>(game.is_discarding);
    *cursor++ = static_cast<float>(game.is_initial_build_phase);
    *cursor++ = static_cast<float>(game.current_prompt == ActionPrompt::move_robber);

    write_full_player_features(game, game.players[base_player], cursor);
    for (int offset = 1; offset < num_players; ++offset) {
        const int player = (base_player + offset) % num_players;
        write_public_player_features(game, game.players[player], cursor);
    }
    *cursor++ = static_cast<float>(game.num_turns);
    for (int offset = 1; offset < num_players; ++offset) {
        const int player = (base_player + offset) % num_players;
        write_private_player_features(game.players[player], cursor);
    }

    const int numeric_size = full_numeric_observation_size(num_players);
    if (cursor != output + numeric_size) {
        return Status::logic_error;
    }

    const int channels = board_channel_count(num_players);
    const int x_deltas[3] = {0, 2, 4};
    const int y_deltas[2] = {0, 2};

    for (int node = 0; node < map.num_nodes; ++node) {
        if (game.node_owner[node] == kEmpty || layout.has_node[node] == 0) {
            continue;
        }
        const int relative =
            (static_cast<int>(game.node_owner[node]) - base_player + num_players) %
            num_players;
        const int x = layout.node_x[node];
        const int y = layout.node_y[node];
        output[board_tensor_index(numeric_size, layout.height, channels, x, y, 2 * relative)] =
            game.node_building[node] == static_cast<std::uint8_t>(Building::settlement)
                ? 1.0F
                : 2.0F;
    }
    for (int edge = 0; edge < map.num_edges; ++edge) {
        if (game.edge_owner[edge] == kEmpty || layout.has_edge[edge] == 0) {
            continue;
        }
        const int relative =
            (static_cast<int>(game.edge_owner[edge]) - base_player + num_players) %
            num_players;
        output[board_tensor_index(
            numeric_size,
            layout.height,
            channels,
            layout.edge_x[edge],
            layout.edge_y[edge],
            2 * relative + 1)] = 1.0F;
    }

    constexpr int kPortNodeIndices[6][2] = {{2, 1}, {3, 2}, {4, 3}, {5, 4}, {0, 5}, {1, 0}};
    for (int tile_index = 0; tile_index < map.num_tiles; ++tile_index) {
        const PackedTile& tile = map.tiles[tile_index];
        if (tile.kind == TileKind::land && layout.has_tile[tile_index] != 0) {
            const int x = layout.tile_x[tile_index];
            const int y = layout.tile_y[tile_index];
            if (tile.resource >= 0) {
                const float probability = dice_probability(tile.number);
                const int channel = 2 * num_players + tile.resource;
                for (int dx = 0; dx < 3; ++dx) {
                    for (int dy = 0; dy < 2; ++dy) {
                        output[board_tensor_index(
                            numeric_size,
                            layout.height,
                            channels,
                            x + x_deltas[dx],
                            y + y_deltas[dy],
                            channel)] += probability;
                    }
                }
            }
            if (tile_index == game.robber_tile) {
                const int channel = 2 * num_players + 5;
                for (int dx = 0; dx < 3; ++dx) {
                    for (int dy = 0; dy < 2; ++dy) {
                        output[board_tensor_index(
                            numeric_size,
                            layout.height,
                            channels,
                            x + x_deltas[dx],
                            y + y_deltas[dy],
                            channel)] = 1.0F;
                    }
                }
            }
        }
        if (tile.kind == TileKind::port && tile.port_direction >= 0) {
            const int resource = tile.resource >= 0 ? tile.resource : 5;
            const int channel = 2 * num_players + 6 + resource;
            const int direction = tile.port_direction;
            for (int n = 0; n < 2; ++n) {
                const int node = tile.nodes[kPortNodeIndices[direction][n]];
                if (node < 0 || node >= kMaxNodes || layout.has_node[node] == 0) {
                    continue;
                }
                output[board_tensor_index(
                    numeric_size,
                    layout.height,
                    channels,
                    layout.node_x[node],
                    layout.node_y[node],
                    channel)] = 1.0F;
            }
        }
    }
    return Status::ok;
}

CUDANATRON_HD double production_sum(
    const PackedMap& map,
    const PackedGame& game,
    int player) {
    double total = 0.0;
    for (int i = 0; i < map.num_land_tiles; ++i) {
        const int tile_index = map.land_tile_indices[i];
        const PackedTile& tile = map.tiles[tile_index];
        if (tile.resource < 0 || tile.number < 0 || tile_index == game.robber_tile) {
            continue;
        }
        const double probability = static_cast<double>(dice_probability(tile.number));
        for (int n = 0; n < 6; ++n) {
            const int node = tile.nodes[n];
            if (game.node_owner[node] != static_cast<std::uint8_t>(player)) {
                continue;
            }
            total += probability *
                     (game.node_building[node] == static_cast<std::uint8_t>(Building::city)
                          ? 2.0
                          : 1.0);
        }
    }
    return total;
}

}  // namespace cudanatron
