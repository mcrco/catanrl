#pragma once

#include <cstdint>

#include "cudanatron/config.hpp"

namespace cudanatron {

// PCG-XSH-RR 32-bit generator. Each game and each search keeps an independent
// stream. Replay of dice, development cards, and robber steals bypasses this
// generator, so engine parity does not depend on matching Python's MT19937.
struct Rng {
    std::uint64_t state{0x853c49e6748fea9bULL};
    std::uint64_t inc{0xda3e39cb94b95bdbULL};

    CUDANATRON_HD void seed(std::uint64_t value) {
        state = 0;
        inc = (value << 1u) | 1u;
        next_u32();
        state += value;
        next_u32();
    }

    CUDANATRON_HD std::uint32_t next_u32() {
        const std::uint64_t old = state;
        state = old * 6364136223846793005ULL + (inc | 1u);
        const std::uint32_t xorshifted =
            static_cast<std::uint32_t>(((old >> 18u) ^ old) >> 27u);
        const std::uint32_t rot = static_cast<std::uint32_t>(old >> 59u);
        return (xorshifted >> rot) | (xorshifted << ((-rot) & 31u));
    }

    CUDANATRON_HD int uniform_int(int lo, int hi_inclusive) {
        const int span = hi_inclusive - lo + 1;
        return lo + static_cast<int>(next_u32() % static_cast<std::uint32_t>(span));
    }
};

CUDANATRON_HD inline void fisher_yates(std::uint8_t* values, int count, Rng& rng) {
    for (int i = count - 1; i > 0; --i) {
        const int j = rng.uniform_int(0, i);
        const std::uint8_t tmp = values[i];
        values[i] = values[j];
        values[j] = tmp;
    }
}

}  // namespace cudanatron
