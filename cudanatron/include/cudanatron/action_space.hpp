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

int flat_index(
    const FlatActionSpace& space,
    const PackedMap& map,
    const PackedGame& game,
    PackedAction action);

PackedAction decode_flat_action(
    const FlatActionSpace& space,
    const PackedMap& map,
    const PackedGame& game,
    int index);

void write_legal_mask(
    const PackedMap& map,
    const PackedGame& game,
    const FlatActionSpace& space,
    std::uint8_t* mask,
    int mask_size);

}  // namespace cudanatron
