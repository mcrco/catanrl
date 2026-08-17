#include "cudanatron/game.hpp"

#include <cstddef>

namespace cudanatron {
namespace {

CUDANATRON_HD bool bit_get(std::uint64_t mask, int index) {
    return ((mask >> index) & 1ULL) != 0;
}

CUDANATRON_HD std::uint64_t bit_set(std::uint64_t mask, int index) {
    return mask | (1ULL << index);
}

CUDANATRON_HD std::uint64_t bit_clear(std::uint64_t mask, int index) {
    return mask & ~(1ULL << index);
}

CUDANATRON_HD bool can_afford(const PackedPlayer& player, const int* cost) {
    for (int i = 0; i < kResourceCount; ++i) {
        if (player.resources[i] < cost[i]) {
            return false;
        }
    }
    return true;
}

CUDANATRON_HD int resource_total(const PackedPlayer& player) {
    int total = 0;
    for (int i = 0; i < kResourceCount; ++i) {
        total += player.resources[i];
    }
    return total;
}

CUDANATRON_HD bool can_play_dev(const PackedPlayer& player, DevelopmentCard card) {
    if (card == DevelopmentCard::victory_point) {
        return false;
    }
    const int index = static_cast<int>(card);
    return player.has_played_development_card_in_turn == 0 &&
           player.development_cards[index] > 0 &&
           ((player.development_card_owned_at_start >> index) & 1u) != 0;
}

CUDANATRON_HD bool is_enemy_node(const PackedGame& game, int node, int player) {
    const std::uint8_t owner = game.node_owner[node];
    return owner != kEmpty && owner != static_cast<std::uint8_t>(player);
}

CUDANATRON_HD bool is_friendly_node(const PackedGame& game, int node, int player) {
    return game.node_owner[node] == static_cast<std::uint8_t>(player);
}

CUDANATRON_HD bool is_friendly_road(const PackedGame& game, int edge, int player) {
    return game.edge_owner[edge] == static_cast<std::uint8_t>(player);
}

CUDANATRON_HD bool node_in_network(const PackedGame& game, int player, int node) {
    for (int c = 0; c < game.num_components[player]; ++c) {
        if (bit_get(game.components[player][c], node)) {
            return true;
        }
    }
    return false;
}

CUDANATRON_HD int component_index(const PackedGame& game, int player, int node) {
    for (int c = 0; c < game.num_components[player]; ++c) {
        if (bit_get(game.components[player][c], node)) {
            return c;
        }
    }
    return -1;
}

CUDANATRON_HD std::uint64_t dfs_walk(
    const PackedMap& map,
    const PackedGame& game,
    int start,
    int player) {
    std::uint64_t visited = 0;
    int stack[kMaxNodes];
    int sp = 0;
    stack[sp++] = start;
    while (sp > 0) {
        const int node = stack[--sp];
        if (bit_get(visited, node)) {
            continue;
        }
        visited = bit_set(visited, node);
        if (is_enemy_node(game, node, player)) {
            continue;
        }
        for (int i = 0; i < map.node_neighbor_count[node]; ++i) {
            const int neighbor = map.node_neighbors[node][i];
            const int edge = map.node_edge[node][i];
            if (!bit_get(visited, neighbor) && is_friendly_road(game, edge, player)) {
                stack[sp++] = neighbor;
            }
        }
    }
    return visited;
}

CUDANATRON_HD int longest_acyclic_path(
    const PackedMap& map,
    const PackedGame& game,
    std::uint64_t component,
    int player) {
    int best = 0;
    std::uint64_t used0 = 0;
    std::uint64_t used1 = 0;
    auto edge_used = [&](int edge) {
        return edge < 64 ? bit_get(used0, edge) : bit_get(used1, edge - 64);
    };
    auto mark_edge = [&](int edge, bool value) {
        if (edge < 64) {
            used0 = value ? bit_set(used0, edge) : bit_clear(used0, edge);
        } else {
            used1 = value ? bit_set(used1, edge - 64) : bit_clear(used1, edge - 64);
        }
    };

    struct Frame {
        std::int16_t node;
        std::int16_t length;
        std::int8_t neighbor_i;
        std::int16_t taken_edge;
    };
    Frame stack[32];

    for (int start = 0; start < map.num_nodes; ++start) {
        if (!bit_get(component, start)) {
            continue;
        }
        int sp = 0;
        stack[sp++] = Frame{
            static_cast<std::int16_t>(start),
            0,
            0,
            -1,
        };
        while (sp > 0) {
            Frame& frame = stack[sp - 1];
            bool pushed = false;
            while (frame.neighbor_i < map.node_neighbor_count[frame.node]) {
                const int i = frame.neighbor_i++;
                const int neighbor = map.node_neighbors[frame.node][i];
                const int edge = map.node_edge[frame.node][i];
                if (!is_friendly_road(game, edge, player) || is_enemy_node(game, neighbor, player) ||
                    edge_used(edge)) {
                    continue;
                }
                mark_edge(edge, true);
                frame.taken_edge = static_cast<std::int16_t>(edge);
                stack[sp++] = Frame{
                    static_cast<std::int16_t>(neighbor),
                    static_cast<std::int16_t>(frame.length + 1),
                    0,
                    -1,
                };
                pushed = true;
                break;
            }
            if (pushed) {
                continue;
            }
            if (frame.length > best) {
                best = frame.length;
            }
            --sp;
            if (sp > 0 && stack[sp - 1].taken_edge >= 0) {
                mark_edge(stack[sp - 1].taken_edge, false);
                stack[sp - 1].taken_edge = -1;
            }
        }
    }
    return best;
}

CUDANATRON_HD void sync_player_road_lengths(PackedGame* game) {
    for (int player = 0; player < game->num_players; ++player) {
        game->players[player].longest_road_length = game->board_road_lengths[player];
    }
}

CUDANATRON_HD void maintain_longest_road(
    PackedGame* game,
    int previous_owner,
    int next_owner) {
    // Catanatron copies board.road_lengths into player_state only here, not
    // during initial placement.
    sync_player_road_lengths(game);
    if (next_owner < 0 || previous_owner == next_owner) {
        return;
    }
    if (previous_owner >= 0) {
        PackedPlayer& previous = game->players[previous_owner];
        previous.has_road = 0;
        previous.victory_points = static_cast<std::int8_t>(previous.victory_points - 2);
        previous.actual_victory_points =
            static_cast<std::int8_t>(previous.actual_victory_points - 2);
    }
    PackedPlayer& winner = game->players[next_owner];
    winner.has_road = 1;
    winner.victory_points = static_cast<std::int8_t>(winner.victory_points + 2);
    winner.actual_victory_points =
        static_cast<std::int8_t>(winner.actual_victory_points + 2);
}

CUDANATRON_HD int max_road_owner(const PackedGame& game) {
    int best_player = -1;
    int best_length = -1;
    for (int player = 0; player < game.num_players; ++player) {
        const int length = game.board_road_lengths[player];
        if (length > best_length) {
            best_length = length;
            best_player = player;
        }
    }
    return best_player;
}

CUDANATRON_HD void recompute_player_road_length(
    const PackedMap& map,
    PackedGame* game,
    int player) {
    int best = 0;
    for (int c = 0; c < game->num_components[player]; ++c) {
        const int length =
            longest_acyclic_path(map, *game, game->components[player][c], player);
        if (length > best) {
            best = length;
        }
    }
    game->board_road_lengths[player] = static_cast<std::int8_t>(best);
}

CUDANATRON_HD void pay(PackedGame* game, int player, const int* cost) {
    for (int i = 0; i < kResourceCount; ++i) {
        game->players[player].resources[i] =
            static_cast<std::int8_t>(game->players[player].resources[i] - cost[i]);
        game->resource_bank[i] = static_cast<std::int8_t>(game->resource_bank[i] + cost[i]);
    }
}

CUDANATRON_HD Status consume_dev(PackedGame* game, int player, DevelopmentCard card) {
    PackedPlayer& state = game->players[player];
    if (!can_play_dev(state, card)) {
        return Status::logic_error;
    }
    const int index = static_cast<int>(card);
    --state.development_cards[index];
    ++state.played_development_cards[index];
    state.has_played_development_card_in_turn = 1;
    return Status::ok;
}

CUDANATRON_HD void advance_turn(PackedGame* game, int direction) {
    const int count = game->num_players;
    game->current_player_index = static_cast<std::uint8_t>(
        (game->current_player_index + direction + count) % count);
    game->current_turn_index = game->current_player_index;
    ++game->num_turns;
}

CUDANATRON_HD void reset_trading_state(PackedGame* game) {
    game->is_resolving_trade = 0;
    game->trade_offering_player = -1;
    game->acceptees = 0;
    for (int i = 0; i < kResourceCount; ++i) {
        game->trade_offering[i] = 0;
        game->trade_asking[i] = 0;
    }
}

CUDANATRON_HD void port_rates(const PackedMap& map, const PackedGame& game, int player, int* rates) {
    for (int i = 0; i < kResourceCount; ++i) {
        rates[i] = 4;
    }
    bool has_31 = false;
    std::uint8_t has_21 = 0;
    for (int t = 0; t < map.num_tiles; ++t) {
        const PackedTile& tile = map.tiles[t];
        if (tile.kind != TileKind::port || tile.port_direction < 0) {
            continue;
        }
        int a = 0;
        int b = 0;
        switch (static_cast<Direction>(tile.port_direction)) {
            case Direction::west:
                a = tile.nodes[5];
                b = tile.nodes[4];
                break;
            case Direction::northwest:
                a = tile.nodes[0];
                b = tile.nodes[5];
                break;
            case Direction::northeast:
                a = tile.nodes[1];
                b = tile.nodes[0];
                break;
            case Direction::east:
                a = tile.nodes[2];
                b = tile.nodes[1];
                break;
            case Direction::southeast:
                a = tile.nodes[3];
                b = tile.nodes[2];
                break;
            case Direction::southwest:
                a = tile.nodes[4];
                b = tile.nodes[3];
                break;
        }
        if (!is_friendly_node(game, a, player) && !is_friendly_node(game, b, player)) {
            continue;
        }
        if (tile.resource < 0) {
            has_31 = true;
        } else {
            has_21 = static_cast<std::uint8_t>(has_21 | (1u << tile.resource));
        }
    }
    if (has_31) {
        for (int i = 0; i < kResourceCount; ++i) {
            rates[i] = 3;
        }
    }
    for (int i = 0; i < kResourceCount; ++i) {
        if ((has_21 >> i) & 1u) {
            rates[i] = 2;
        }
    }
}

CUDANATRON_HD bool buildable_edge(const PackedMap& map, const PackedGame& game, int player, int edge) {
    if (game.edge_owner[edge] != kEmpty) {
        return false;
    }
    return node_in_network(game, player, map.edge_a[edge]) ||
           node_in_network(game, player, map.edge_b[edge]);
}

CUDANATRON_HD bool any_buildable_edge(const PackedMap& map, const PackedGame& game, int player) {
    for (int e = 0; e < map.num_edges; ++e) {
        if (buildable_edge(map, game, player, e)) {
            return true;
        }
    }
    return false;
}

CUDANATRON_HD void erase_component(PackedGame* game, int player, int index) {
    for (int i = index; i < game->num_components[player] - 1; ++i) {
        game->components[player][i] = game->components[player][i + 1];
    }
    --game->num_components[player];
    game->components[player][game->num_components[player]] = 0;
}

CUDANATRON_HD Status build_settlement(
    const PackedMap& map,
    PackedGame* game,
    int player,
    int node,
    bool initial) {
    if (node < 0 || node >= map.num_nodes) {
        return Status::invalid_argument;
    }
    if (!bit_get(game->board_buildable, node)) {
        return Status::invalid_argument;
    }
    if (!initial && !node_in_network(*game, player, node)) {
        return Status::invalid_argument;
    }
    if (game->node_owner[node] != kEmpty) {
        return Status::invalid_argument;
    }

    const int previous_owner = game->road_color;
    game->node_owner[node] = static_cast<std::uint8_t>(player);
    game->node_building[node] = static_cast<std::uint8_t>(Building::settlement);

    if (initial) {
        game->components[player][game->num_components[player]++] = bit_set(0, node);
    } else {
        int adjacent_count[kMaxPlayers]{};
        int adjacent_other[kMaxPlayers][2]{};
        for (int i = 0; i < map.node_neighbor_count[node]; ++i) {
            const int neighbor = map.node_neighbors[node][i];
            const int edge = map.node_edge[node][i];
            const std::uint8_t owner = game->edge_owner[edge];
            if (owner == kEmpty || owner == static_cast<std::uint8_t>(player)) {
                continue;
            }
            const int enemy = owner;
            if (adjacent_count[enemy] < 2) {
                adjacent_other[enemy][adjacent_count[enemy]] = neighbor;
            }
            ++adjacent_count[enemy];
        }
        for (int enemy = 0; enemy < game->num_players; ++enemy) {
            if (adjacent_count[enemy] != 2) {
                continue;
            }
            const int component = component_index(*game, enemy, node);
            if (component < 0) {
                return Status::logic_error;
            }
            erase_component(game, enemy, component);
            game->components[enemy][game->num_components[enemy]++] =
                dfs_walk(map, *game, adjacent_other[enemy][0], enemy);
            game->components[enemy][game->num_components[enemy]++] =
                dfs_walk(map, *game, adjacent_other[enemy][1], enemy);
            recompute_player_road_length(map, game, enemy);
            const int winner = max_road_owner(*game);
            game->road_color = static_cast<std::int8_t>(winner);
            game->road_length =
                winner < 0 ? 0 : game->board_road_lengths[winner];
        }
    }

    game->board_buildable = bit_clear(game->board_buildable, node);
    for (int i = 0; i < map.node_neighbor_count[node]; ++i) {
        game->board_buildable =
            bit_clear(game->board_buildable, map.node_neighbors[node][i]);
    }
    if (!initial) {
        maintain_longest_road(game, previous_owner, game->road_color);
    }
    return Status::ok;
}

CUDANATRON_HD Status build_road(const PackedMap& map, PackedGame* game, int player, int edge) {
    if (edge < 0 || edge >= map.num_edges || !buildable_edge(map, *game, player, edge)) {
        return Status::invalid_argument;
    }
    const int previous_owner = game->road_color;
    game->edge_owner[edge] = static_cast<std::uint8_t>(player);

    const int a = map.edge_a[edge];
    const int b = map.edge_b[edge];
    const int a_index = component_index(*game, player, a);
    const int b_index = component_index(*game, player, b);
    int chosen = 0;
    if (a_index < 0 && !is_enemy_node(*game, a, player)) {
        if (b_index < 0) {
            return Status::logic_error;
        }
        chosen = b_index;
        game->components[player][chosen] = bit_set(game->components[player][chosen], a);
    } else if (b_index < 0 && !is_enemy_node(*game, b, player)) {
        if (a_index < 0) {
            return Status::logic_error;
        }
        chosen = a_index;
        game->components[player][chosen] = bit_set(game->components[player][chosen], b);
    } else if (a_index >= 0 && b_index >= 0 && a_index != b_index) {
        chosen = a_index;
        game->components[player][chosen] |= game->components[player][b_index];
        erase_component(game, player, b_index);
        if (b_index < chosen) {
            --chosen;
        }
    } else {
        const int index = a_index >= 0 ? a_index : b_index;
        if (index < 0) {
            return Status::logic_error;
        }
        chosen = index;
    }

    const int candidate =
        longest_acyclic_path(map, *game, game->components[player][chosen], player);
    if (candidate > game->board_road_lengths[player]) {
        game->board_road_lengths[player] = static_cast<std::int8_t>(candidate);
    }
    if (candidate >= 5 && candidate > game->road_length) {
        game->road_color = static_cast<std::int8_t>(player);
        game->road_length = static_cast<std::int8_t>(candidate);
    }
    if (!game->is_initial_build_phase) {
        maintain_longest_road(game, previous_owner, game->road_color);
    }
    return Status::ok;
}

CUDANATRON_HD Status build_city(PackedGame* game, int player, int node) {
    if (game->node_owner[node] != static_cast<std::uint8_t>(player) ||
        game->node_building[node] != static_cast<std::uint8_t>(Building::settlement)) {
        return Status::invalid_argument;
    }
    game->node_building[node] = static_cast<std::uint8_t>(Building::city);
    return Status::ok;
}

CUDANATRON_HD void yield_resources(const PackedMap& map, PackedGame* game, int number) {
    int payouts[kMaxPlayers][kResourceCount]{};
    int totals[kResourceCount]{};
    for (int i = 0; i < map.num_land_tiles; ++i) {
        const int tile_index = map.land_tile_indices[i];
        const PackedTile& tile = map.tiles[tile_index];
        if (tile.number != number || tile.resource < 0 ||
            tile_index == game->robber_tile) {
            continue;
        }
        const int resource = tile.resource;
        for (int n = 0; n < 6; ++n) {
            const int node = tile.nodes[n];
            const std::uint8_t owner = game->node_owner[node];
            if (owner == kEmpty) {
                continue;
            }
            const int count =
                game->node_building[node] == static_cast<std::uint8_t>(Building::city) ? 2 : 1;
            payouts[owner][resource] += count;
            totals[resource] += count;
        }
    }
    for (int resource = 0; resource < kResourceCount; ++resource) {
        if (totals[resource] > game->resource_bank[resource]) {
            continue;
        }
        for (int player = 0; player < game->num_players; ++player) {
            game->players[player].resources[resource] = static_cast<std::int8_t>(
                game->players[player].resources[resource] + payouts[player][resource]);
            game->resource_bank[resource] = static_cast<std::int8_t>(
                game->resource_bank[resource] - payouts[player][resource]);
        }
    }
}

CUDANATRON_HD bool robber_blocks_low_vp(
    const PackedMap& map,
    const PackedGame& game,
    int player,
    int tile_index) {
    const PackedTile& tile = map.tiles[tile_index];
    for (int n = 0; n < 6; ++n) {
        const std::uint8_t owner = game.node_owner[tile.nodes[n]];
        if (owner != kEmpty && owner != static_cast<std::uint8_t>(player) &&
            game.players[owner].actual_victory_points < 3) {
            return true;
        }
    }
    return false;
}

CUDANATRON_HD void collect_robber_victims(
    const PackedMap& map,
    const PackedGame& game,
    int player,
    int tile_index,
    int* victims,
    int* count) {
    *count = 0;
    const PackedTile& tile = map.tiles[tile_index];
    std::uint8_t seen = 0;
    for (int n = 0; n < 6; ++n) {
        const std::uint8_t owner = game.node_owner[tile.nodes[n]];
        if (owner == kEmpty || owner == static_cast<std::uint8_t>(player) ||
            ((seen >> owner) & 1u) != 0) {
            continue;
        }
        if (resource_total(game.players[owner]) == 0) {
            continue;
        }
        seen = static_cast<std::uint8_t>(seen | (1u << owner));
        victims[(*count)++] = owner;
    }
}

CUDANATRON_HD PackedAction make_basic(ActionType type, int player) {
    PackedAction action{};
    action.type = type;
    action.player = static_cast<std::uint8_t>(player);
    return action;
}

CUDANATRON_HD void push_action(PackedAction* out, int capacity, int* count, PackedAction action) {
    if (*count < capacity) {
        out[*count] = action;
    }
    ++*count;
}

CUDANATRON_HD void maintain_largest_army(PackedGame* game, int player) {
    PackedPlayer& candidate = game->players[player];
    const int candidate_size =
        candidate.played_development_cards[static_cast<int>(DevelopmentCard::knight)];
    if (candidate_size < 3) {
        return;
    }
    int previous = -1;
    for (int i = 0; i < game->num_players; ++i) {
        if (game->players[i].has_army) {
            previous = i;
            break;
        }
    }
    if (previous < 0) {
        candidate.has_army = 1;
        candidate.victory_points = static_cast<std::int8_t>(candidate.victory_points + 2);
        candidate.actual_victory_points =
            static_cast<std::int8_t>(candidate.actual_victory_points + 2);
        return;
    }
    if (previous == player) {
        return;
    }
    const int previous_size =
        game->players[previous]
            .played_development_cards[static_cast<int>(DevelopmentCard::knight)];
    if (candidate_size <= previous_size) {
        return;
    }
    PackedPlayer& loser = game->players[previous];
    loser.has_army = 0;
    loser.victory_points = static_cast<std::int8_t>(loser.victory_points - 2);
    loser.actual_victory_points = static_cast<std::int8_t>(loser.actual_victory_points - 2);
    candidate.has_army = 1;
    candidate.victory_points = static_cast<std::int8_t>(candidate.victory_points + 2);
    candidate.actual_victory_points =
        static_cast<std::int8_t>(candidate.actual_victory_points + 2);
}

}  // namespace

CUDANATRON_HD int turns_since(int completed_turns, int last_completed_turn) {
    return last_completed_turn < 0 ? -1 : completed_turns - last_completed_turn;
}

CUDANATRON_HD int num_resource_cards(const PackedGame& game, int player) {
    return resource_total(game.players[player]);
}

CUDANATRON_HD int winning_player(const PackedGame& game) {
    int winner = -1;
    for (int i = 0; i < game.num_players; ++i) {
        if (game.players[i].actual_victory_points >= game.victory_points_to_win) {
            winner = i;
        }
    }
    return winner;
}

CUDANATRON_HD bool search_equivalent(const PackedGame& lhs, const PackedGame& rhs) {
    const auto* a = reinterpret_cast<const unsigned char*>(&lhs);
    const auto* b = reinterpret_cast<const unsigned char*>(&rhs);
    for (std::size_t i = 0; i < sizeof(PackedGame); ++i) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

CUDANATRON_HD bool packed_actions_equal(const PackedAction& lhs, const PackedAction& rhs) {
    if (lhs.type != rhs.type || lhs.player != rhs.player || lhs.node != rhs.node ||
        lhs.edge != rhs.edge || lhs.resource != rhs.resource ||
        lhs.resource_b != rhs.resource_b || lhs.yop_count != rhs.yop_count ||
        lhs.maritime_rate != rhs.maritime_rate || lhs.maritime_offer != rhs.maritime_offer ||
        lhs.maritime_ask != rhs.maritime_ask || lhs.robber_tile != rhs.robber_tile ||
        lhs.robber_victim != rhs.robber_victim || lhs.trade_partner != rhs.trade_partner) {
        return false;
    }
    for (int i = 0; i < kResourceCount; ++i) {
        if (lhs.trade_offering[i] != rhs.trade_offering[i] ||
            lhs.trade_asking[i] != rhs.trade_asking[i]) {
            return false;
        }
    }
    return true;
}

Status initialize_game(const PackedMap& map, const GameConfig& config, PackedGame* game) {
    if (game == nullptr || config.num_players < 1 || config.num_players > kMaxPlayers) {
        return Status::invalid_argument;
    }
    *game = PackedGame{};
    game->num_players = static_cast<std::uint8_t>(config.num_players);
    game->discard_limit = static_cast<std::uint8_t>(config.discard_limit);
    game->friendly_robber = config.friendly_robber ? 1 : 0;
    game->victory_points_to_win = static_cast<std::uint8_t>(config.victory_points_to_win);
    game->robber_tile = map.desert_tile;
    game->rng.seed(config.game_seed);
    for (int node = 0; node < kMaxNodes; ++node) {
        game->node_owner[node] = kEmpty;
    }
    for (int edge = 0; edge < kMaxEdges; ++edge) {
        game->edge_owner[edge] = kEmpty;
    }
    std::uint64_t land_nodes = 0;
    for (int i = 0; i < map.num_land_tiles; ++i) {
        const PackedTile& tile = map.tiles[map.land_tile_indices[i]];
        for (int n = 0; n < 6; ++n) {
            land_nodes = bit_set(land_nodes, tile.nodes[n]);
        }
    }
    game->board_buildable = land_nodes;

    std::uint8_t deck[kMaxDevDeck];
    int deck_size = 0;
    for (int i = 0; i < 14; ++i) {
        deck[deck_size++] = static_cast<std::uint8_t>(DevelopmentCard::knight);
    }
    for (int i = 0; i < 2; ++i) {
        deck[deck_size++] = static_cast<std::uint8_t>(DevelopmentCard::year_of_plenty);
        deck[deck_size++] = static_cast<std::uint8_t>(DevelopmentCard::monopoly);
        deck[deck_size++] = static_cast<std::uint8_t>(DevelopmentCard::road_building);
    }
    for (int i = 0; i < 5; ++i) {
        deck[deck_size++] = static_cast<std::uint8_t>(DevelopmentCard::victory_point);
    }
    fisher_yates(deck, deck_size, game->rng);
    game->development_deck_size = static_cast<std::uint8_t>(deck_size);
    for (int i = 0; i < deck_size; ++i) {
        game->development_deck[i] = deck[i];
    }
    return Status::ok;
}

CUDANATRON_HD int generate_legal_actions(
    const PackedMap& map,
    const PackedGame& game,
    PackedAction* out,
    int capacity) {
    const int player = game.current_player_index;
    int count = 0;
    if (game.current_prompt == ActionPrompt::build_initial_settlement) {
        for (int node = 0; node < map.num_nodes; ++node) {
            if (!bit_get(game.board_buildable, node)) {
                continue;
            }
            PackedAction action = make_basic(ActionType::build_settlement, player);
            action.node = static_cast<std::int16_t>(node);
            push_action(out, capacity, &count, action);
        }
        return count;
    }
    if (game.current_prompt == ActionPrompt::build_initial_road) {
        const PackedPlayer& state = game.players[player];
        if (state.settlement_count == 0) {
            return 0;
        }
        const int last = state.settlements[state.settlement_count - 1];
        for (int e = 0; e < map.num_edges; ++e) {
            if (!buildable_edge(map, game, player, e)) {
                continue;
            }
            if (map.edge_a[e] != last && map.edge_b[e] != last) {
                continue;
            }
            PackedAction action = make_basic(ActionType::build_road, player);
            action.edge = static_cast<std::int16_t>(e);
            push_action(out, capacity, &count, action);
        }
        return count;
    }
    if (game.current_prompt == ActionPrompt::decide_trade) {
        PackedAction reject = make_basic(ActionType::reject_trade, player);
        push_action(out, capacity, &count, reject);
        bool can_accept = true;
        for (int i = 0; i < kResourceCount; ++i) {
            if (game.players[player].resources[i] < game.trade_asking[i]) {
                can_accept = false;
            }
        }
        if (can_accept) {
            push_action(out, capacity, &count, make_basic(ActionType::accept_trade, player));
        }
        return count;
    }
    if (game.current_prompt == ActionPrompt::decide_acceptees) {
        push_action(out, capacity, &count, make_basic(ActionType::cancel_trade, player));
        for (int i = 0; i < game.num_players; ++i) {
            if (((game.acceptees >> i) & 1u) == 0) {
                continue;
            }
            PackedAction confirm = make_basic(ActionType::confirm_trade, player);
            confirm.trade_partner = static_cast<std::int8_t>(i);
            push_action(out, capacity, &count, confirm);
        }
        return count;
    }
    if (game.current_prompt == ActionPrompt::discard) {
        const PackedPlayer& state = game.players[player];
        for (int i = 0; i < kResourceCount; ++i) {
            if (state.resources[i] <= 0) {
                continue;
            }
            PackedAction action = make_basic(ActionType::discard_resource, player);
            action.resource = static_cast<std::int8_t>(i);
            push_action(out, capacity, &count, action);
        }
        return count;
    }
    if (game.current_prompt == ActionPrompt::move_robber) {
        PackedAction unfiltered[kMaxLegalActions];
        int unfiltered_count = 0;
        for (int i = 0; i < map.num_land_tiles; ++i) {
            const int tile_index = map.land_tile_indices[i];
            if (tile_index == game.robber_tile) {
                continue;
            }
            int victims[kMaxPlayers];
            int victim_count = 0;
            collect_robber_victims(map, game, player, tile_index, victims, &victim_count);
            if (victim_count == 0) {
                PackedAction action = make_basic(ActionType::move_robber, player);
                action.robber_tile = static_cast<std::int16_t>(tile_index);
                action.robber_victim = -1;
                push_action(unfiltered, kMaxLegalActions, &unfiltered_count, action);
            } else {
                for (int v = 0; v < victim_count; ++v) {
                    PackedAction action = make_basic(ActionType::move_robber, player);
                    action.robber_tile = static_cast<std::int16_t>(tile_index);
                    action.robber_victim = static_cast<std::int8_t>(victims[v]);
                    push_action(unfiltered, kMaxLegalActions, &unfiltered_count, action);
                }
            }
        }
        if (!game.friendly_robber) {
            for (int i = 0; i < unfiltered_count && i < capacity; ++i) {
                out[i] = unfiltered[i];
            }
            return unfiltered_count;
        }
        for (int i = 0; i < unfiltered_count; ++i) {
            if (!robber_blocks_low_vp(map, game, player, unfiltered[i].robber_tile)) {
                push_action(out, capacity, &count, unfiltered[i]);
            }
        }
        if (count == 0) {
            for (int i = 0; i < unfiltered_count && i < capacity; ++i) {
                out[i] = unfiltered[i];
            }
            return unfiltered_count;
        }
        return count;
    }

    const PackedPlayer& state = game.players[player];
    if (game.is_road_building) {
        if (state.roads_available > 0) {
            for (int e = 0; e < map.num_edges; ++e) {
                if (!buildable_edge(map, game, player, e)) {
                    continue;
                }
                PackedAction action = make_basic(ActionType::build_road, player);
                action.edge = static_cast<std::int16_t>(e);
                push_action(out, capacity, &count, action);
            }
        }
        return count;
    }

    if (can_play_dev(state, DevelopmentCard::year_of_plenty)) {
        bool single[kResourceCount]{};
        bool pair[kResourceCount][kResourceCount]{};
        for (int first = 0; first < kResourceCount; ++first) {
            for (int second = first; second < kResourceCount; ++second) {
                const bool pair_available =
                    first == second
                        ? game.resource_bank[first] >= 2
                        : game.resource_bank[first] >= 1 && game.resource_bank[second] >= 1;
                if (pair_available) {
                    pair[first][second] = true;
                } else {
                    if (game.resource_bank[first] > 0) {
                        single[first] = true;
                    }
                    if (game.resource_bank[second] > 0) {
                        single[second] = true;
                    }
                }
            }
        }
        for (int i = 0; i < kResourceCount; ++i) {
            if (!single[i]) {
                continue;
            }
            PackedAction action = make_basic(ActionType::play_year_of_plenty, player);
            action.resource = static_cast<std::int8_t>(i);
            action.yop_count = 1;
            push_action(out, capacity, &count, action);
        }
        for (int i = 0; i < kResourceCount; ++i) {
            for (int j = i; j < kResourceCount; ++j) {
                if (!pair[i][j]) {
                    continue;
                }
                PackedAction action = make_basic(ActionType::play_year_of_plenty, player);
                action.resource = static_cast<std::int8_t>(i);
                action.resource_b = static_cast<std::int8_t>(j);
                action.yop_count = 2;
                push_action(out, capacity, &count, action);
            }
        }
    }
    if (can_play_dev(state, DevelopmentCard::monopoly)) {
        for (int i = 0; i < kResourceCount; ++i) {
            PackedAction action = make_basic(ActionType::play_monopoly, player);
            action.resource = static_cast<std::int8_t>(i);
            push_action(out, capacity, &count, action);
        }
    }
    if (can_play_dev(state, DevelopmentCard::knight)) {
        push_action(out, capacity, &count, make_basic(ActionType::play_knight_card, player));
    }
    if (can_play_dev(state, DevelopmentCard::road_building) && state.roads_available > 0 &&
        any_buildable_edge(map, game, player)) {
        push_action(out, capacity, &count, make_basic(ActionType::play_road_building, player));
    }

    if (!state.has_rolled) {
        push_action(out, capacity, &count, make_basic(ActionType::roll, player));
        return count;
    }

    push_action(out, capacity, &count, make_basic(ActionType::end_turn, player));
    int road_cost[kResourceCount];
    fill_road_cost(road_cost);
    if (state.roads_available > 0 && can_afford(state, road_cost)) {
        for (int e = 0; e < map.num_edges; ++e) {
            if (!buildable_edge(map, game, player, e)) {
                continue;
            }
            PackedAction action = make_basic(ActionType::build_road, player);
            action.edge = static_cast<std::int16_t>(e);
            push_action(out, capacity, &count, action);
        }
    }
    int settlement_cost[kResourceCount];
    fill_settlement_cost(settlement_cost);
    if (state.settlements_available > 0 && can_afford(state, settlement_cost)) {
        for (int node = 0; node < map.num_nodes; ++node) {
            if (!bit_get(game.board_buildable, node) || !node_in_network(game, player, node)) {
                continue;
            }
            PackedAction action = make_basic(ActionType::build_settlement, player);
            action.node = static_cast<std::int16_t>(node);
            push_action(out, capacity, &count, action);
        }
    }
    int city_cost[kResourceCount];
    fill_city_cost(city_cost);
    if (state.cities_available > 0 && can_afford(state, city_cost)) {
        for (int i = 0; i < state.settlement_count; ++i) {
            PackedAction action = make_basic(ActionType::build_city, player);
            action.node = state.settlements[i];
            push_action(out, capacity, &count, action);
        }
    }
    int dev_cost[kResourceCount];
    fill_dev_card_cost(dev_cost);
    if (game.development_deck_size > 0 && can_afford(state, dev_cost)) {
        push_action(out, capacity, &count, make_basic(ActionType::buy_development_card, player));
    }

    int rates[kResourceCount];
    port_rates(map, game, player, rates);
    for (int offered = 0; offered < kResourceCount; ++offered) {
        if (state.resources[offered] < rates[offered]) {
            continue;
        }
        for (int received = 0; received < kResourceCount; ++received) {
            if (offered == received || game.resource_bank[received] <= 0) {
                continue;
            }
            PackedAction action = make_basic(ActionType::maritime_trade, player);
            action.maritime_rate = static_cast<std::uint8_t>(rates[offered]);
            action.maritime_offer = static_cast<std::int8_t>(offered);
            action.maritime_ask = static_cast<std::int8_t>(received);
            push_action(out, capacity, &count, action);
        }
    }
    return count;
}

namespace {

CUDANATRON_HD bool action_is_legal(
    const PackedMap& map,
    const PackedGame& game,
    PackedAction action) {
    PackedAction legal[kMaxLegalActions];
    const int count = generate_legal_actions(map, game, legal, kMaxLegalActions);
    for (int i = 0; i < count; ++i) {
        if (packed_actions_equal(legal[i], action)) {
            return true;
        }
    }
    if (action.type == ActionType::offer_trade &&
        action.player == game.current_player_index &&
        game.current_prompt == ActionPrompt::play_turn &&
        game.players[action.player].has_rolled) {
        int offered = 0;
        int asked = 0;
        bool overlap = false;
        for (int i = 0; i < kResourceCount; ++i) {
            offered += action.trade_offering[i];
            asked += action.trade_asking[i];
            if (action.trade_offering[i] > 0 && action.trade_asking[i] > 0) {
                overlap = true;
            }
        }
        return offered > 0 && asked > 0 && !overlap;
    }
    return false;
}

CUDANATRON_HD Status apply_build_settlement(
    const PackedMap& map,
    PackedGame* game,
    PackedAction action) {
    PackedPlayer& state = game->players[action.player];
    const int node = action.node;
    if (game->is_initial_build_phase) {
        const Status status = build_settlement(map, game, action.player, node, true);
        if (status != Status::ok) {
            return status;
        }
        state.settlements[state.settlement_count++] = static_cast<std::int16_t>(node);
        --state.settlements_available;
        ++state.victory_points;
        ++state.actual_victory_points;
        if (state.settlement_count == 2) {
            for (int i = 0; i < map.num_land_tiles; ++i) {
                const PackedTile& tile = map.tiles[map.land_tile_indices[i]];
                if (tile.resource < 0) {
                    continue;
                }
                bool adjacent = false;
                for (int n = 0; n < 6; ++n) {
                    if (tile.nodes[n] == node) {
                        adjacent = true;
                    }
                }
                if (!adjacent) {
                    continue;
                }
                if (game->resource_bank[tile.resource] <= 0) {
                    return Status::logic_error;
                }
                --game->resource_bank[tile.resource];
                ++state.resources[tile.resource];
            }
        }
        game->current_prompt = ActionPrompt::build_initial_road;
        return Status::ok;
    }
    const Status status = build_settlement(map, game, action.player, node, false);
    if (status != Status::ok) {
        return status;
    }
    state.settlements[state.settlement_count++] = static_cast<std::int16_t>(node);
    --state.settlements_available;
    ++state.victory_points;
    ++state.actual_victory_points;
    int settlement_cost[kResourceCount];
    fill_settlement_cost(settlement_cost);
    pay(game, action.player, settlement_cost);
    return Status::ok;
}

CUDANATRON_HD Status apply_build_road(
    const PackedMap& map,
    PackedGame* game,
    PackedAction action) {
    PackedPlayer& state = game->players[action.player];
    const Status status = build_road(map, game, action.player, action.edge);
    if (status != Status::ok) {
        return status;
    }
    state.roads[state.road_count++] = action.edge;
    --state.roads_available;
    if (game->is_initial_build_phase) {
        int buildings = 0;
        for (int i = 0; i < game->num_players; ++i) {
            buildings += game->players[i].settlement_count;
        }
        const int num_players = game->num_players;
        if (buildings < num_players) {
            advance_turn(game, 1);
            game->current_prompt = ActionPrompt::build_initial_settlement;
        } else if (buildings == num_players) {
            game->current_prompt = ActionPrompt::build_initial_settlement;
        } else if (buildings == 2 * num_players) {
            game->is_initial_build_phase = 0;
            game->current_prompt = ActionPrompt::play_turn;
        } else {
            advance_turn(game, -1);
            game->current_prompt = ActionPrompt::build_initial_settlement;
        }
        return Status::ok;
    }
    if (game->is_road_building && game->free_roads_available > 0) {
        --game->free_roads_available;
        if (game->free_roads_available == 0 || state.roads_available == 0 ||
            !any_buildable_edge(map, *game, action.player)) {
            game->is_road_building = 0;
            game->free_roads_available = 0;
        }
    } else {
        int road_cost[kResourceCount];
        fill_road_cost(road_cost);
        pay(game, action.player, road_cost);
    }
    return Status::ok;
}

CUDANATRON_HD Status apply_build_city(PackedGame* game, PackedAction action) {
    PackedPlayer& state = game->players[action.player];
    const Status status = build_city(game, action.player, action.node);
    if (status != Status::ok) {
        return status;
    }
    int found = -1;
    for (int i = 0; i < state.settlement_count; ++i) {
        if (state.settlements[i] == action.node) {
            found = i;
            break;
        }
    }
    if (found < 0) {
        return Status::logic_error;
    }
    for (int i = found; i < state.settlement_count - 1; ++i) {
        state.settlements[i] = state.settlements[i + 1];
    }
    --state.settlement_count;
    state.cities[state.city_count++] = action.node;
    ++state.settlements_available;
    --state.cities_available;
    ++state.victory_points;
    ++state.actual_victory_points;
    int city_cost[kResourceCount];
    fill_city_cost(city_cost);
    pay(game, action.player, city_cost);
    return Status::ok;
}

CUDANATRON_HD Status apply_buy_dev(PackedGame* game, PackedAction action, const Replay* replay) {
    int dev_cost[kResourceCount];
    fill_dev_card_cost(dev_cost);
    if (game->development_deck_size == 0 ||
        !can_afford(game->players[action.player], dev_cost)) {
        return Status::logic_error;
    }
    std::uint8_t card = 0;
    if (replay != nullptr && replay->has_development_card) {
        int found = -1;
        for (int i = 0; i < game->development_deck_size; ++i) {
            if (game->development_deck[i] == static_cast<std::uint8_t>(replay->development_card)) {
                found = i;
                break;
            }
        }
        if (found < 0) {
            return Status::logic_error;
        }
        card = game->development_deck[found];
        for (int i = found; i < game->development_deck_size - 1; ++i) {
            game->development_deck[i] = game->development_deck[i + 1];
        }
        --game->development_deck_size;
    } else {
        card = game->development_deck[game->development_deck_size - 1];
        --game->development_deck_size;
    }
    PackedPlayer& state = game->players[action.player];
    ++state.development_cards[card];
    state.last_dev_bought_completed_turn = game->completed_turns;
    if (card == static_cast<std::uint8_t>(DevelopmentCard::victory_point)) {
        ++state.actual_victory_points;
    }
    pay(game, action.player, dev_cost);
    return Status::ok;
}

CUDANATRON_HD Status apply_roll(const PackedMap& map, PackedGame* game, PackedAction action, const Replay* replay) {
    int die0 = 0;
    int die1 = 0;
    if (replay != nullptr && replay->has_dice) {
        die0 = replay->die0;
        die1 = replay->die1;
    } else {
        die0 = game->rng.uniform_int(1, 6);
        die1 = game->rng.uniform_int(1, 6);
    }
    const int number = die0 + die1;
    game->players[action.player].has_rolled = 1;
    if (number == 7) {
        int first_discarder = -1;
        for (int i = 0; i < game->num_players; ++i) {
            const int cards = resource_total(game->players[i]);
            game->discard_counts[i] =
                cards > game->discard_limit ? static_cast<std::uint8_t>(cards / 2) : 0;
            if (game->discard_counts[i] > 0 && first_discarder < 0) {
                first_discarder = i;
            }
        }
        if (first_discarder >= 0) {
            game->current_player_index = static_cast<std::uint8_t>(first_discarder);
            game->current_prompt = ActionPrompt::discard;
            game->is_discarding = 1;
        } else {
            for (int i = 0; i < game->num_players; ++i) {
                game->discard_counts[i] = 0;
            }
            game->current_prompt = ActionPrompt::move_robber;
            game->is_moving_knight = 1;
        }
        return Status::ok;
    }
    yield_resources(map, game, number);
    game->current_prompt = ActionPrompt::play_turn;
    return Status::ok;
}

CUDANATRON_HD Status apply_discard(PackedGame* game, PackedAction action) {
    const int index = action.player;
    if (game->discard_counts[index] == 0) {
        return Status::logic_error;
    }
    const int resource = action.resource;
    PackedPlayer& state = game->players[index];
    if (state.resources[resource] <= 0) {
        return Status::logic_error;
    }
    --state.resources[resource];
    ++game->resource_bank[resource];
    --game->discard_counts[index];
    if (game->discard_counts[index] > 0) {
        return Status::ok;
    }
    int next_discarder = -1;
    for (int i = index + 1; i < game->num_players; ++i) {
        if (game->discard_counts[i] > 0) {
            next_discarder = i;
            break;
        }
    }
    if (next_discarder >= 0) {
        game->current_player_index = static_cast<std::uint8_t>(next_discarder);
        return Status::ok;
    }
    game->current_player_index = game->current_turn_index;
    game->current_prompt = ActionPrompt::move_robber;
    game->is_discarding = 0;
    game->is_moving_knight = 1;
    for (int i = 0; i < game->num_players; ++i) {
        game->discard_counts[i] = 0;
    }
    return Status::ok;
}

CUDANATRON_HD Status apply_move_robber(
    const PackedMap& map,
    PackedGame* game,
    PackedAction action,
    const Replay* replay) {
    if (action.robber_victim >= 0) {
        PackedPlayer& victim = game->players[action.robber_victim];
        int cards[19 * 5];
        int card_count = 0;
        for (int i = 0; i < kResourceCount; ++i) {
            for (int n = 0; n < victim.resources[i]; ++n) {
                cards[card_count++] = i;
            }
        }
        if (card_count == 0) {
            return Status::logic_error;
        }
        int stolen = 0;
        if (replay != nullptr && replay->has_stolen_resource) {
            stolen = replay->stolen_resource;
            if (victim.resources[stolen] <= 0) {
                return Status::logic_error;
            }
        } else {
            stolen = cards[game->rng.uniform_int(0, card_count - 1)];
        }
        --victim.resources[stolen];
        ++game->players[action.player].resources[stolen];
    }
    if (action.robber_tile < 0 || action.robber_tile >= map.num_tiles ||
        map.tiles[action.robber_tile].kind != TileKind::land ||
        action.robber_tile == game->robber_tile) {
        return Status::invalid_argument;
    }
    game->robber_tile = static_cast<std::uint8_t>(action.robber_tile);
    game->current_prompt = ActionPrompt::play_turn;
    game->is_moving_knight = 0;
    return Status::ok;
}

}  // namespace

CUDANATRON_HD Status execute_action(
    const PackedMap& map,
    PackedGame* game,
    PackedAction action,
    const Replay* replay) {
    if (game == nullptr) {
        return Status::invalid_argument;
    }
    if (!action_is_legal(map, *game, action)) {
        return Status::illegal_action;
    }
    Status status = Status::ok;
    switch (action.type) {
        case ActionType::build_settlement:
            status = apply_build_settlement(map, game, action);
            break;
        case ActionType::build_road:
            status = apply_build_road(map, game, action);
            break;
        case ActionType::build_city:
            status = apply_build_city(game, action);
            break;
        case ActionType::buy_development_card:
            status = apply_buy_dev(game, action, replay);
            break;
        case ActionType::roll:
            status = apply_roll(map, game, action, replay);
            break;
        case ActionType::discard_resource:
            status = apply_discard(game, action);
            break;
        case ActionType::move_robber:
            status = apply_move_robber(map, game, action, replay);
            break;
        case ActionType::play_knight_card: {
            status = consume_dev(game, action.player, DevelopmentCard::knight);
            if (status != Status::ok) {
                break;
            }
            game->players[action.player].last_knight_completed_turn = game->completed_turns;
            maintain_largest_army(game, action.player);
            game->current_prompt = ActionPrompt::move_robber;
            game->is_moving_knight = 1;
            break;
        }
        case ActionType::play_year_of_plenty: {
            int requested[kResourceCount]{};
            requested[action.resource] += 1;
            if (action.yop_count == 2) {
                requested[action.resource_b] += 1;
            }
            for (int i = 0; i < kResourceCount; ++i) {
                if (requested[i] > game->resource_bank[i]) {
                    return Status::logic_error;
                }
            }
            status = consume_dev(game, action.player, DevelopmentCard::year_of_plenty);
            if (status != Status::ok) {
                break;
            }
            for (int i = 0; i < kResourceCount; ++i) {
                game->players[action.player].resources[i] = static_cast<std::int8_t>(
                    game->players[action.player].resources[i] + requested[i]);
                game->resource_bank[i] =
                    static_cast<std::int8_t>(game->resource_bank[i] - requested[i]);
            }
            game->current_prompt = ActionPrompt::play_turn;
            break;
        }
        case ActionType::play_monopoly: {
            status = consume_dev(game, action.player, DevelopmentCard::monopoly);
            if (status != Status::ok) {
                break;
            }
            const int resource = action.resource;
            int stolen = 0;
            for (int i = 0; i < game->num_players; ++i) {
                if (i == action.player) {
                    continue;
                }
                stolen += game->players[i].resources[resource];
                game->players[i].resources[resource] = 0;
            }
            game->players[action.player].resources[resource] = static_cast<std::int8_t>(
                game->players[action.player].resources[resource] + stolen);
            game->current_prompt = ActionPrompt::play_turn;
            break;
        }
        case ActionType::play_road_building:
            status = consume_dev(game, action.player, DevelopmentCard::road_building);
            if (status != Status::ok) {
                break;
            }
            game->is_road_building = 1;
            game->free_roads_available = 2;
            game->current_prompt = ActionPrompt::play_turn;
            break;
        case ActionType::maritime_trade: {
            int offered[kResourceCount]{};
            offered[action.maritime_offer] = action.maritime_rate;
            if (!can_afford(game->players[action.player], offered) ||
                game->resource_bank[action.maritime_ask] <= 0) {
                return Status::logic_error;
            }
            pay(game, action.player, offered);
            --game->resource_bank[action.maritime_ask];
            ++game->players[action.player].resources[action.maritime_ask];
            game->current_prompt = ActionPrompt::play_turn;
            break;
        }
        case ActionType::offer_trade: {
            for (int i = 0; i < kResourceCount; ++i) {
                game->trade_offering[i] = action.trade_offering[i];
                game->trade_asking[i] = action.trade_asking[i];
            }
            game->trade_offering_player = static_cast<std::int8_t>(game->current_turn_index);
            game->is_resolving_trade = 1;
            int next = 0;
            for (; next < game->num_players; ++next) {
                if (next != action.player) {
                    break;
                }
            }
            game->current_player_index = static_cast<std::uint8_t>(next);
            game->current_prompt = ActionPrompt::decide_trade;
            break;
        }
        case ActionType::accept_trade: {
            game->acceptees = static_cast<std::uint8_t>(game->acceptees | (1u << action.player));
            int next = -1;
            for (int i = game->current_player_index + 1; i < game->num_players; ++i) {
                if (i != action.player) {
                    next = i;
                    break;
                }
            }
            if (next >= 0) {
                game->current_player_index = static_cast<std::uint8_t>(next);
            } else {
                game->current_player_index = game->current_turn_index;
                game->current_prompt = ActionPrompt::decide_acceptees;
            }
            break;
        }
        case ActionType::reject_trade: {
            int next = -1;
            for (int i = game->current_player_index + 1; i < game->num_players; ++i) {
                if (i != action.player) {
                    next = i;
                    break;
                }
            }
            if (next >= 0) {
                game->current_player_index = static_cast<std::uint8_t>(next);
                break;
            }
            game->current_player_index = game->current_turn_index;
            if (game->acceptees == 0) {
                reset_trading_state(game);
                game->current_prompt = ActionPrompt::play_turn;
            } else {
                game->current_prompt = ActionPrompt::decide_acceptees;
            }
            break;
        }
        case ActionType::confirm_trade: {
            PackedPlayer& offering = game->players[action.player];
            PackedPlayer& partner = game->players[action.trade_partner];
            for (int i = 0; i < kResourceCount; ++i) {
                offering.resources[i] = static_cast<std::int8_t>(
                    offering.resources[i] - game->trade_offering[i] + game->trade_asking[i]);
                partner.resources[i] = static_cast<std::int8_t>(
                    partner.resources[i] - game->trade_asking[i] + game->trade_offering[i]);
            }
            reset_trading_state(game);
            game->current_player_index = game->current_turn_index;
            game->current_prompt = ActionPrompt::play_turn;
            break;
        }
        case ActionType::cancel_trade:
            reset_trading_state(game);
            game->current_player_index = game->current_turn_index;
            game->current_prompt = ActionPrompt::play_turn;
            break;
        case ActionType::end_turn: {
            PackedPlayer& state = game->players[action.player];
            state.has_played_development_card_in_turn = 0;
            state.has_rolled = 0;
            state.development_card_owned_at_start = 0;
            for (int i = 0; i < kPlayableDevCount; ++i) {
                if (state.development_cards[i] > 0) {
                    state.development_card_owned_at_start =
                        static_cast<std::uint8_t>(state.development_card_owned_at_start | (1u << i));
                }
            }
            ++game->completed_turns;
            advance_turn(game, 1);
            game->current_prompt = ActionPrompt::play_turn;
            break;
        }
        default:
            return Status::logic_error;
    }
    return status;
}

}  // namespace cudanatron
