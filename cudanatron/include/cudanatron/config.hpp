#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define CUDANATRON_HD __host__ __device__
#define CUDANATRON_HOST __host__
#define CUDANATRON_DEVICE __device__
#else
#define CUDANATRON_HD
#define CUDANATRON_HOST
#define CUDANATRON_DEVICE
#endif

namespace cudanatron {

inline constexpr int kMaxPlayers = 4;
inline constexpr int kResourceCount = 5;
inline constexpr int kDevCardCount = 5;
inline constexpr int kPlayableDevCount = 4;
inline constexpr int kMaxTiles = 37;
inline constexpr int kMaxLandTiles = 19;
inline constexpr int kMaxPorts = 9;
inline constexpr int kMaxNodes = 54;
inline constexpr int kMaxEdges = 72;
inline constexpr int kMaxNodeDegree = 3;
inline constexpr int kMaxComponents = 16;
inline constexpr int kMaxDevDeck = 25;
inline constexpr int kMaxSettlements = 5;
inline constexpr int kMaxCities = 4;
inline constexpr int kMaxRoads = 15;
inline constexpr int kMaxLegalActions = 512;
inline constexpr int kMaxActionSpace = 512;
inline constexpr int kBoardWidth = 21;
inline constexpr int kBoardHeight = 11;
inline constexpr int kBoardChannelsWithoutPlayers = 12;
inline constexpr int kPlayerFullFeatureCount = 32;
inline constexpr int kPlayerPublicFeatureCount = 16;
inline constexpr int kPlayerPrivateFeatureCount = 16;
inline constexpr int kSharedNumericFeatureCount = 10;
inline constexpr int kMaxChanceOutcomes = 36;
inline constexpr int kEmpty = 255;
inline constexpr int kNoPlayer = -1;
inline constexpr int kBankStartingAmount = 19;

inline constexpr int kInitialRoads = 15;
inline constexpr int kInitialSettlements = 5;
inline constexpr int kInitialCities = 4;

CUDANATRON_HD inline void fill_road_cost(int* cost) {
    cost[0] = 1;
    cost[1] = 1;
    cost[2] = 0;
    cost[3] = 0;
    cost[4] = 0;
}

CUDANATRON_HD inline void fill_settlement_cost(int* cost) {
    cost[0] = 1;
    cost[1] = 1;
    cost[2] = 1;
    cost[3] = 1;
    cost[4] = 0;
}

CUDANATRON_HD inline void fill_city_cost(int* cost) {
    cost[0] = 0;
    cost[1] = 0;
    cost[2] = 0;
    cost[3] = 2;
    cost[4] = 3;
}

CUDANATRON_HD inline void fill_dev_card_cost(int* cost) {
    cost[0] = 0;
    cost[1] = 0;
    cost[2] = 1;
    cost[3] = 1;
    cost[4] = 1;
}

}  // namespace cudanatron
