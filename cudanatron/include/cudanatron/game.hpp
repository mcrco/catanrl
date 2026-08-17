#pragma once

#include "cudanatron/state.hpp"

namespace cudanatron {

Status initialize_game(const PackedMap& map, const GameConfig& config, PackedGame* game);

CUDANATRON_HD int generate_legal_actions(
    const PackedMap& map,
    const PackedGame& game,
    PackedAction* out,
    int capacity);

CUDANATRON_HD Status execute_action(
    const PackedMap& map,
    PackedGame* game,
    PackedAction action,
    const Replay* replay);

CUDANATRON_HD int winning_player(const PackedGame& game);

CUDANATRON_HD int num_resource_cards(const PackedGame& game, int player);

CUDANATRON_HD bool search_equivalent(const PackedGame& lhs, const PackedGame& rhs);

CUDANATRON_HD int turns_since(int completed_turns, int last_completed_turn);

}  // namespace cudanatron
