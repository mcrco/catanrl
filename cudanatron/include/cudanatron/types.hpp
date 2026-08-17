#pragma once

#include <cstdint>

#include "cudanatron/config.hpp"

namespace cudanatron {

enum class Color : std::uint8_t { red, blue, white, orange };
enum class Resource : std::uint8_t { wood, brick, sheep, wheat, ore };
enum class DevelopmentCard : std::uint8_t {
    knight,
    year_of_plenty,
    monopoly,
    road_building,
    victory_point,
};
enum class Building : std::uint8_t { settlement, city };
enum class MapType : std::uint8_t { base, mini, tournament };
enum class NumberPlacement : std::uint8_t { official_spiral, random };
enum class TileKind : std::uint8_t { land, water, port };

enum class Direction : std::uint8_t {
    east,
    southeast,
    southwest,
    west,
    northwest,
    northeast,
};

enum class NodeRef : std::uint8_t {
    north,
    northeast,
    southeast,
    south,
    southwest,
    northwest,
};

enum class EdgeRef : std::uint8_t {
    east,
    southeast,
    southwest,
    west,
    northwest,
    northeast,
};

enum class ActionPrompt : std::uint8_t {
    build_initial_settlement,
    build_initial_road,
    play_turn,
    discard,
    move_robber,
    decide_trade,
    decide_acceptees,
};

enum class ActionType : std::uint8_t {
    roll,
    move_robber,
    discard_resource,
    build_road,
    build_settlement,
    build_city,
    buy_development_card,
    play_knight_card,
    play_year_of_plenty,
    play_monopoly,
    play_road_building,
    maritime_trade,
    offer_trade,
    accept_trade,
    reject_trade,
    confirm_trade,
    cancel_trade,
    end_turn,
};

enum class Status : std::int32_t {
    ok = 0,
    illegal_action = 1,
    invalid_argument = 2,
    logic_error = 3,
    out_of_range = 4,
    exhausted = 5,
};

struct Coordinate {
    std::int8_t x{};
    std::int8_t y{};
    std::int8_t z{};

    CUDANATRON_HD bool operator==(Coordinate other) const {
        return x == other.x && y == other.y && z == other.z;
    }

    CUDANATRON_HD bool operator!=(Coordinate other) const { return !(*this == other); }
};

CUDANATRON_HD inline Coordinate operator+(Coordinate lhs, Coordinate rhs) {
    return Coordinate{
        static_cast<std::int8_t>(lhs.x + rhs.x),
        static_cast<std::int8_t>(lhs.y + rhs.y),
        static_cast<std::int8_t>(lhs.z + rhs.z),
    };
}

CUDANATRON_HD inline Coordinate direction_vector(Direction direction) {
    switch (direction) {
        case Direction::east:
            return {1, -1, 0};
        case Direction::southeast:
            return {0, -1, 1};
        case Direction::southwest:
            return {-1, 0, 1};
        case Direction::west:
            return {-1, 1, 0};
        case Direction::northwest:
            return {0, 1, -1};
        case Direction::northeast:
            return {1, 0, -1};
    }
    return {0, 0, 0};
}

CUDANATRON_HD inline int resource_index(Resource resource) {
    return static_cast<int>(resource);
}

CUDANATRON_HD inline Color default_color(int seating_index) {
    constexpr Color kOrder[kMaxPlayers] = {
        Color::red,
        Color::blue,
        Color::white,
        Color::orange,
    };
    return kOrder[seating_index];
}

}  // namespace cudanatron
