#include "cudanatron/chance.hpp"

namespace cudanatron {
namespace {

CUDANATRON_HD void add_outcome(
    ChanceOutcome* outcomes,
    int capacity,
    int* count,
    Replay replay,
    float probability) {
    if (*count < capacity) {
        outcomes[*count].replay = replay;
        outcomes[*count].probability = probability;
    }
    ++*count;
}

}  // namespace

CUDANATRON_HD int enumerate_chance_outcomes(
    const PackedMap& map,
    const PackedGame& game,
    PackedAction action,
    ChanceOutcome* outcomes,
    int capacity) {
    (void)map;
    int count = 0;
    if (action.type == ActionType::roll) {
        constexpr int kMultiplicities[11] = {1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
        for (int total = 2; total <= 12; ++total) {
            Replay replay{};
            replay.has_dice = true;
            if (total <= 7) {
                replay.die0 = 1;
                replay.die1 = static_cast<std::int8_t>(total - 1);
            } else {
                replay.die0 = 6;
                replay.die1 = static_cast<std::int8_t>(total - 6);
            }
            add_outcome(
                outcomes,
                capacity,
                &count,
                replay,
                static_cast<float>(kMultiplicities[total - 2]) / 36.0F);
        }
        return count > capacity ? capacity : count;
    }

    if (action.type == ActionType::buy_development_card) {
        int card_counts[kDevCardCount]{};
        int total = 0;
        for (int i = 0; i < game.development_deck_size; ++i) {
            const int card = game.development_deck[i];
            if (card < 0 || card >= kDevCardCount) {
                continue;
            }
            ++card_counts[card];
            ++total;
        }
        if (total <= 0) {
            add_outcome(outcomes, capacity, &count, Replay{}, 1.0F);
            return count > capacity ? capacity : count;
        }
        for (int card = 0; card < kDevCardCount; ++card) {
            if (card_counts[card] <= 0) {
                continue;
            }
            Replay replay{};
            replay.has_development_card = true;
            replay.development_card = static_cast<std::int8_t>(card);
            add_outcome(
                outcomes,
                capacity,
                &count,
                replay,
                static_cast<float>(card_counts[card]) / static_cast<float>(total));
        }
        return count > capacity ? capacity : count;
    }

    if (action.type == ActionType::move_robber && action.robber_victim >= 0 &&
        action.robber_victim < game.num_players) {
        const PackedPlayer& victim = game.players[action.robber_victim];
        int total = 0;
        for (int i = 0; i < kResourceCount; ++i) {
            total += victim.resources[i];
        }
        if (total > 0) {
            for (int resource = 0; resource < kResourceCount; ++resource) {
                if (victim.resources[resource] <= 0) {
                    continue;
                }
                Replay replay{};
                replay.has_stolen_resource = true;
                replay.stolen_resource = static_cast<std::int8_t>(resource);
                add_outcome(
                    outcomes,
                    capacity,
                    &count,
                    replay,
                    static_cast<float>(victim.resources[resource]) /
                        static_cast<float>(total));
            }
            return count > capacity ? capacity : count;
        }
    }

    add_outcome(outcomes, capacity, &count, Replay{}, 1.0F);
    return count > capacity ? capacity : count;
}

}  // namespace cudanatron
