#pragma once

#include "cudanatron/game.hpp"

namespace cudanatron {

struct ChanceOutcome {
    Replay replay{};
    float probability{1.0F};
};

struct ChanceTable {
    int count{0};
    ChanceOutcome outcomes[kMaxChanceOutcomes]{};
};

CUDANATRON_HD int enumerate_chance_outcomes(
    const PackedMap& map,
    const PackedGame& game,
    PackedAction action,
    ChanceOutcome* outcomes,
    int capacity);

}  // namespace cudanatron
