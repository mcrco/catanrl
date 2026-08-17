#pragma once

#include <cstdint>

#include "cudanatron/action.hpp"
#include "cudanatron/map.hpp"
#include "cudanatron/rng.hpp"
#include "cudanatron/types.hpp"

namespace cudanatron {

struct PackedPlayer {
    std::int8_t victory_points{0};
    std::int8_t actual_victory_points{0};
    std::int8_t roads_available{kInitialRoads};
    std::int8_t settlements_available{kInitialSettlements};
    std::int8_t cities_available{kInitialCities};
    std::uint8_t has_road{0};
    std::uint8_t has_army{0};
    std::uint8_t has_rolled{0};
    std::uint8_t has_played_development_card_in_turn{0};
    std::int8_t longest_road_length{0};
    std::int8_t resources[kResourceCount]{};
    std::int8_t development_cards[kDevCardCount]{};
    std::int8_t played_development_cards[kDevCardCount]{};
    std::uint8_t development_card_owned_at_start{0};
    std::int16_t last_knight_completed_turn{-1};
    std::int16_t last_dev_bought_completed_turn{-1};
    std::int16_t settlements[kMaxSettlements]{};
    std::int16_t cities[kMaxCities]{};
    std::int16_t roads[kMaxRoads]{};
    std::uint8_t settlement_count{0};
    std::uint8_t city_count{0};
    std::uint8_t road_count{0};
};

struct PackedGame {
    PackedPlayer players[kMaxPlayers]{};
    std::uint8_t num_players{0};
    std::uint8_t current_player_index{0};
    std::uint8_t current_turn_index{0};
    ActionPrompt current_prompt{ActionPrompt::build_initial_settlement};
    std::uint8_t is_initial_build_phase{1};
    std::uint8_t is_discarding{0};
    std::uint8_t is_moving_knight{0};
    std::uint8_t is_road_building{0};
    std::uint8_t is_resolving_trade{0};
    std::uint8_t friendly_robber{0};
    std::uint8_t free_roads_available{0};
    std::uint8_t discard_limit{7};
    std::uint8_t victory_points_to_win{10};
    std::int16_t num_turns{0};
    std::int16_t completed_turns{0};
    std::int8_t resource_bank[kResourceCount]{
        kBankStartingAmount,
        kBankStartingAmount,
        kBankStartingAmount,
        kBankStartingAmount,
        kBankStartingAmount,
    };
    std::uint8_t development_deck[kMaxDevDeck]{};
    std::uint8_t development_deck_size{0};
    std::uint8_t robber_tile{0};
    std::uint8_t node_owner[kMaxNodes]{};
    std::uint8_t node_building[kMaxNodes]{};
    std::uint8_t edge_owner[kMaxEdges]{};
    std::uint8_t discard_counts[kMaxPlayers]{};
    std::int8_t trade_offering[kResourceCount]{};
    std::int8_t trade_asking[kResourceCount]{};
    std::int8_t trade_offering_player{-1};
    std::uint8_t acceptees{0};
    std::int8_t road_color{-1};
    std::int8_t road_length{0};
    std::int8_t board_road_lengths[kMaxPlayers]{};
    std::uint64_t board_buildable{};
    std::uint64_t components[kMaxPlayers][kMaxComponents]{};
    std::uint8_t num_components[kMaxPlayers]{};
    Rng rng{};
};

struct GameConfig {
    int num_players{2};
    MapType map_type{MapType::base};
    NumberPlacement number_placement{NumberPlacement::official_spiral};
    std::uint64_t map_seed{0};
    std::uint64_t game_seed{0};
    int discard_limit{7};
    bool friendly_robber{false};
    int victory_points_to_win{10};
};

}  // namespace cudanatron
