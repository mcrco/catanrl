#pragma once

#include "cudanatron/game.hpp"

namespace cudanatron {

struct FlatActionSpace {
    int num_players{0};
    MapType map_type{MapType::base};
    int size{0};
    FlatAction actions[kMaxActionSpace]{};
};

Status build_flat_action_space(
    FlatActionSpace* space,
    const PackedMap& map,
    int num_players);

CUDANATRON_HD int flat_index(
    const FlatActionSpace& space,
    const PackedMap& map,
    const PackedGame& game,
    PackedAction action);

CUDANATRON_HD PackedAction decode_flat_action(
    const FlatActionSpace& space,
    const PackedMap& map,
    const PackedGame& game,
    int index);

CUDANATRON_HD void write_legal_mask(
    const PackedMap& map,
    const PackedGame& game,
    const FlatActionSpace& space,
    std::uint8_t* mask,
    int mask_size);

// Sort key matching `str((ActionType, value))` in Catanatron / catanrl.
void write_flat_action_key(const FlatAction& action, char* buffer, int buffer_size);

}  // namespace cudanatron
