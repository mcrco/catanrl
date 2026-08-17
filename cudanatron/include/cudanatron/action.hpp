#pragma once

#include <cstdint>

#include "cudanatron/types.hpp"

namespace cudanatron {

// Compact action used by the device engine. Flat-action indices are a separate
// table built once per (player count, map type) and never stored in the game.
struct PackedAction {
    ActionType type{ActionType::end_turn};
    std::uint8_t player{0};
    std::int16_t node{-1};
    std::int16_t edge{-1};
    std::int8_t resource{-1};
    std::int8_t resource_b{-1};
    std::uint8_t yop_count{0};
    std::uint8_t maritime_rate{0};
    std::int8_t maritime_offer{-1};
    std::int8_t maritime_ask{-1};
    std::int16_t robber_tile{-1};
    std::int8_t robber_victim{-1};  // seating index, or -1
    std::int8_t trade_offering[kResourceCount]{};
    std::int8_t trade_asking[kResourceCount]{};
    std::int8_t trade_partner{-1};
};

struct Replay {
    bool has_dice{false};
    std::int8_t die0{0};
    std::int8_t die1{0};
    bool has_development_card{false};
    std::int8_t development_card{-1};
    bool has_stolen_resource{false};
    std::int8_t stolen_resource{-1};
};

struct FlatAction {
    ActionType type{ActionType::end_turn};
    std::int16_t node{-1};
    std::int16_t edge_a{-1};
    std::int16_t edge_b{-1};
    std::int8_t resource{-1};
    std::int8_t resource_b{-1};
    std::uint8_t yop_count{0};
    std::uint8_t maritime_rate{0};
    std::int8_t maritime_offer{-1};
    std::int8_t maritime_ask{-1};
    std::int8_t robber_x{0};
    std::int8_t robber_y{0};
    std::int8_t robber_z{0};
    std::int8_t robber_slot{0};  // 0 = none, else relative opponent slot
};

}  // namespace cudanatron
