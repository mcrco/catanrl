#include "cudanatron/action_space.hpp"

#include <algorithm>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

namespace cudanatron {
namespace {

const char* action_type_name(ActionType type) {
    switch (type) {
        case ActionType::roll:
            return "ROLL";
        case ActionType::move_robber:
            return "MOVE_ROBBER";
        case ActionType::discard_resource:
            return "DISCARD_RESOURCE";
        case ActionType::build_road:
            return "BUILD_ROAD";
        case ActionType::build_settlement:
            return "BUILD_SETTLEMENT";
        case ActionType::build_city:
            return "BUILD_CITY";
        case ActionType::buy_development_card:
            return "BUY_DEVELOPMENT_CARD";
        case ActionType::play_knight_card:
            return "PLAY_KNIGHT_CARD";
        case ActionType::play_year_of_plenty:
            return "PLAY_YEAR_OF_PLENTY";
        case ActionType::play_monopoly:
            return "PLAY_MONOPOLY";
        case ActionType::play_road_building:
            return "PLAY_ROAD_BUILDING";
        case ActionType::maritime_trade:
            return "MARITIME_TRADE";
        case ActionType::offer_trade:
            return "OFFER_TRADE";
        case ActionType::accept_trade:
            return "ACCEPT_TRADE";
        case ActionType::reject_trade:
            return "REJECT_TRADE";
        case ActionType::confirm_trade:
            return "CONFIRM_TRADE";
        case ActionType::cancel_trade:
            return "CANCEL_TRADE";
        case ActionType::end_turn:
            return "END_TURN";
    }
    return "";
}

const char* resource_name(int resource) {
    constexpr const char* kNames[kResourceCount] = {
        "WOOD",
        "BRICK",
        "SHEEP",
        "WHEAT",
        "ORE",
    };
    return kNames[resource];
}

std::string resource_repr(int resource) {
    return std::string("'") + resource_name(resource) + "'";
}

std::string value_repr(const FlatAction& action) {
    switch (action.type) {
        case ActionType::roll:
        case ActionType::buy_development_card:
        case ActionType::play_knight_card:
        case ActionType::play_road_building:
        case ActionType::end_turn:
            return "None";
        case ActionType::discard_resource:
        case ActionType::play_monopoly:
            return resource_repr(action.resource);
        case ActionType::build_settlement:
        case ActionType::build_city:
            return std::to_string(action.node);
        case ActionType::build_road: {
            std::ostringstream stream;
            stream << "(" << action.edge_a << ", " << action.edge_b << ")";
            return stream.str();
        }
        case ActionType::play_year_of_plenty: {
            std::ostringstream stream;
            stream << "(" << resource_repr(action.resource);
            if (action.yop_count == 1) {
                stream << ",)";
            } else {
                stream << ", " << resource_repr(action.resource_b) << ")";
            }
            return stream.str();
        }
        case ActionType::move_robber: {
            std::ostringstream stream;
            stream << "((" << static_cast<int>(action.robber_x) << ", "
                   << static_cast<int>(action.robber_y) << ", "
                   << static_cast<int>(action.robber_z) << "), ";
            if (action.robber_slot == 0) {
                stream << "None)";
            } else {
                stream << static_cast<int>(action.robber_slot) << ")";
            }
            return stream.str();
        }
        case ActionType::maritime_trade: {
            std::ostringstream stream;
            stream << "(";
            for (int i = 0; i < 4; ++i) {
                if (i != 0) {
                    stream << ", ";
                }
                if (i < action.maritime_rate) {
                    stream << resource_repr(action.maritime_offer);
                } else {
                    stream << "None";
                }
            }
            stream << ", " << resource_repr(action.maritime_ask) << ")";
            return stream.str();
        }
        default:
            return "None";
    }
}

std::string flat_key(const FlatAction& action) {
    // CatanRL sorts `(ActionType, value)` by str((action_type, value)).
    // ActionType.__repr__ is `AT.{name}`, and tuple str uses repr of each
    // element, so the key is `(AT.BUILD_ROAD, (6, 7))`.
    return std::string("(AT.") + action_type_name(action.type) + ", " +
           value_repr(action) + ")";
}

CUDANATRON_HD FlatAction packed_to_flat(
    const PackedMap& map,
    const PackedGame& game,
    PackedAction action) {
    FlatAction flat{};
    flat.type = action.type;
    switch (action.type) {
        case ActionType::build_settlement:
        case ActionType::build_city:
            flat.node = action.node;
            break;
        case ActionType::build_road:
            flat.edge_a = map.edge_a[action.edge];
            flat.edge_b = map.edge_b[action.edge];
            break;
        case ActionType::discard_resource:
        case ActionType::play_monopoly:
            flat.resource = action.resource;
            break;
        case ActionType::play_year_of_plenty:
            flat.resource = action.resource;
            flat.resource_b = action.resource_b;
            flat.yop_count = action.yop_count;
            break;
        case ActionType::maritime_trade:
            flat.maritime_rate = action.maritime_rate;
            flat.maritime_offer = action.maritime_offer;
            flat.maritime_ask = action.maritime_ask;
            break;
        case ActionType::move_robber: {
            const PackedTile& tile = map.tiles[action.robber_tile];
            flat.robber_x = tile.coordinate.x;
            flat.robber_y = tile.coordinate.y;
            flat.robber_z = tile.coordinate.z;
            if (action.robber_victim < 0) {
                flat.robber_slot = 0;
            } else {
                const int count = game.num_players;
                const int slot =
                    (action.robber_victim - action.player + count) % count;
                flat.robber_slot = static_cast<std::int8_t>(slot);
            }
            break;
        }
        default:
            break;
    }
    return flat;
}

CUDANATRON_HD bool same_flat(const FlatAction& lhs, const FlatAction& rhs) {
    return lhs.type == rhs.type && lhs.node == rhs.node && lhs.edge_a == rhs.edge_a &&
           lhs.edge_b == rhs.edge_b && lhs.resource == rhs.resource &&
           lhs.resource_b == rhs.resource_b && lhs.yop_count == rhs.yop_count &&
           lhs.maritime_rate == rhs.maritime_rate &&
           lhs.maritime_offer == rhs.maritime_offer && lhs.maritime_ask == rhs.maritime_ask &&
           lhs.robber_x == rhs.robber_x && lhs.robber_y == rhs.robber_y &&
           lhs.robber_z == rhs.robber_z && lhs.robber_slot == rhs.robber_slot;
}

}  // namespace

Status build_flat_action_space(
    FlatActionSpace* space,
    const PackedMap& map,
    int num_players) {
    if (space == nullptr || num_players < 1 || num_players > kMaxPlayers) {
        return Status::invalid_argument;
    }
    *space = FlatActionSpace{};
    space->num_players = num_players;
    space->map_type = map.map_type;

    std::vector<FlatAction> actions;
    actions.push_back(FlatAction{ActionType::roll});
    for (int resource = 0; resource < kResourceCount; ++resource) {
        FlatAction action{};
        action.type = ActionType::discard_resource;
        action.resource = static_cast<std::int8_t>(resource);
        actions.push_back(action);
    }
    for (int e = 0; e < map.num_edges; ++e) {
        FlatAction action{};
        action.type = ActionType::build_road;
        action.edge_a = map.edge_a[e];
        action.edge_b = map.edge_b[e];
        actions.push_back(action);
    }
    for (int node = 0; node < map.num_nodes; ++node) {
        FlatAction settlement{};
        settlement.type = ActionType::build_settlement;
        settlement.node = static_cast<std::int16_t>(node);
        actions.push_back(settlement);
        FlatAction city{};
        city.type = ActionType::build_city;
        city.node = static_cast<std::int16_t>(node);
        actions.push_back(city);
    }
    actions.push_back(FlatAction{ActionType::buy_development_card});
    actions.push_back(FlatAction{ActionType::play_knight_card});
    for (int first = 0; first < kResourceCount; ++first) {
        for (int second = first; second < kResourceCount; ++second) {
            FlatAction action{};
            action.type = ActionType::play_year_of_plenty;
            action.resource = static_cast<std::int8_t>(first);
            action.resource_b = static_cast<std::int8_t>(second);
            action.yop_count = 2;
            actions.push_back(action);
        }
        FlatAction single{};
        single.type = ActionType::play_year_of_plenty;
        single.resource = static_cast<std::int8_t>(first);
        single.yop_count = 1;
        actions.push_back(single);
    }
    actions.push_back(FlatAction{ActionType::play_road_building});
    for (int resource = 0; resource < kResourceCount; ++resource) {
        FlatAction action{};
        action.type = ActionType::play_monopoly;
        action.resource = static_cast<std::int8_t>(resource);
        actions.push_back(action);
    }
    for (int i = 0; i < map.num_land_tiles; ++i) {
        const PackedTile& tile = map.tiles[map.land_tile_indices[i]];
        FlatAction none{};
        none.type = ActionType::move_robber;
        none.robber_x = tile.coordinate.x;
        none.robber_y = tile.coordinate.y;
        none.robber_z = tile.coordinate.z;
        none.robber_slot = 0;
        actions.push_back(none);
        for (int slot = 1; slot < num_players; ++slot) {
            FlatAction steal = none;
            steal.robber_slot = static_cast<std::int8_t>(slot);
            actions.push_back(steal);
        }
    }
    for (int offered = 0; offered < kResourceCount; ++offered) {
        for (int received = 0; received < kResourceCount; ++received) {
            if (offered == received) {
                continue;
            }
            for (int rate : {4, 3, 2}) {
                FlatAction action{};
                action.type = ActionType::maritime_trade;
                action.maritime_rate = static_cast<std::uint8_t>(rate);
                action.maritime_offer = static_cast<std::int8_t>(offered);
                action.maritime_ask = static_cast<std::int8_t>(received);
                actions.push_back(action);
            }
        }
    }
    actions.push_back(FlatAction{ActionType::end_turn});

    std::sort(actions.begin(), actions.end(), [](const FlatAction& lhs, const FlatAction& rhs) {
        return flat_key(lhs) < flat_key(rhs);
    });
    if (actions.size() > static_cast<std::size_t>(kMaxActionSpace)) {
        return Status::logic_error;
    }
    space->size = static_cast<int>(actions.size());
    for (int i = 0; i < space->size; ++i) {
        space->actions[i] = actions[static_cast<std::size_t>(i)];
    }
    return Status::ok;
}

CUDANATRON_HD int flat_index(
    const FlatActionSpace& space,
    const PackedMap& map,
    const PackedGame& game,
    PackedAction action) {
    const FlatAction flat = packed_to_flat(map, game, action);
    for (int i = 0; i < space.size; ++i) {
        if (same_flat(space.actions[i], flat)) {
            return i;
        }
    }
    return -1;
}

CUDANATRON_HD PackedAction decode_flat_action(
    const FlatActionSpace& space,
    const PackedMap& map,
    const PackedGame& game,
    int index) {
    PackedAction action{};
    const FlatAction& flat = space.actions[index];
    action.type = flat.type;
    action.player = game.current_player_index;
    switch (flat.type) {
        case ActionType::build_settlement:
        case ActionType::build_city:
            action.node = flat.node;
            break;
        case ActionType::build_road:
            action.edge = static_cast<std::int16_t>(
                find_edge(map, flat.edge_a, flat.edge_b));
            break;
        case ActionType::discard_resource:
        case ActionType::play_monopoly:
            action.resource = flat.resource;
            break;
        case ActionType::play_year_of_plenty:
            action.resource = flat.resource;
            action.resource_b = flat.resource_b;
            action.yop_count = flat.yop_count;
            break;
        case ActionType::maritime_trade:
            action.maritime_rate = flat.maritime_rate;
            action.maritime_offer = flat.maritime_offer;
            action.maritime_ask = flat.maritime_ask;
            break;
        case ActionType::move_robber: {
            action.robber_tile = static_cast<std::int16_t>(find_land_tile_index(
                map,
                Coordinate{flat.robber_x, flat.robber_y, flat.robber_z}));
            if (flat.robber_slot == 0) {
                action.robber_victim = -1;
            } else {
                action.robber_victim = static_cast<std::int8_t>(
                    (action.player + flat.robber_slot) % game.num_players);
            }
            break;
        }
        default:
            break;
    }
    return action;
}

CUDANATRON_HD void write_legal_mask(
    const PackedMap& map,
    const PackedGame& game,
    const FlatActionSpace& space,
    std::uint8_t* mask,
    int mask_size) {
    for (int i = 0; i < mask_size; ++i) {
        mask[i] = 0;
    }
    PackedAction legal[kMaxLegalActions];
    const int count = generate_legal_actions(map, game, legal, kMaxLegalActions);
    for (int i = 0; i < count; ++i) {
        if (legal[i].type == ActionType::offer_trade ||
            legal[i].type == ActionType::accept_trade ||
            legal[i].type == ActionType::reject_trade ||
            legal[i].type == ActionType::confirm_trade ||
            legal[i].type == ActionType::cancel_trade) {
            continue;
        }
        const int index = flat_index(space, map, game, legal[i]);
        if (index >= 0 && index < mask_size) {
            mask[index] = 1;
        }
    }
}

void write_flat_action_key(const FlatAction& action, char* buffer, int buffer_size) {
    if (buffer == nullptr || buffer_size <= 0) {
        return;
    }
    const std::string key = flat_key(action);
    const int n = std::min(buffer_size - 1, static_cast<int>(key.size()));
    std::memcpy(buffer, key.data(), static_cast<std::size_t>(n));
    buffer[n] = '\0';
}

}  // namespace cudanatron
