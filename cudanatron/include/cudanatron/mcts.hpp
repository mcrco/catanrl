#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "cudanatron/action_space.hpp"
#include "cudanatron/game.hpp"
#include "cudanatron/observation.hpp"

namespace cudanatron {

struct WDL {
    double win{0.0};
    double draw{1.0};
    double loss{0.0};

    CUDANATRON_HD double q() const { return win - loss; }
    CUDANATRON_HD WDL flipped() const { return WDL{loss, draw, win}; }
    static WDL from_value(double value);
};

struct MCTSSearchMetrics {
    std::uint64_t simulations{0};
    std::uint32_t principal_variation_depth{0};
    std::uint32_t maximum_depth{0};
    double mean_depth{0.0};
    double root_value{0.0};
    std::uint32_t retained_root_visits{0};
    std::uint64_t pruned_actions{0};
    std::uint64_t coalesced_outcomes{0};
    bool tree_reused{false};
    double q_min{0.0};
    double q_max{0.0};
};

class MCTSSearch {
public:
    MCTSSearch(
        PackedMap map,
        PackedGame root_game,
        FlatActionSpace action_space,
        double c_puct = 1.5,
        std::uint64_t seed = 0,
        bool canonical_pruning = false);
    ~MCTSSearch();

    MCTSSearch(const MCTSSearch&) = delete;
    MCTSSearch& operator=(const MCTSSearch&) = delete;
    MCTSSearch(MCTSSearch&&) noexcept;
    MCTSSearch& operator=(MCTSSearch&&) noexcept;

    void initialize_root(const float* policy_logits, int policy_size);
    void set_root_network_value(double value);
    void set_root_network_wdl(WDL wdl);
    void enable_completed_q_selection(double c_visit = 50.0, double c_scale = 1.0);

    // Returns true when a neural leaf is pending. Terminals are backed up
    // internally and return false.
    bool select_leaf();
    void evaluate_leaf(const float* policy_logits, int policy_size, double value);
    void evaluate_leaf_wdl(const float* policy_logits, int policy_size, WDL wdl);

    bool has_pending_leaf() const;
    bool root_expanded() const;
    int pending_player_index() const;
    const PackedGame& pending_game() const;
    const PackedGame& root_game() const;
    const PackedMap& map() const;
    const FlatActionSpace& action_space() const;

    void write_pending_observation(
        const ObservationLayout& layout,
        float* output,
        int output_size) const;
    void write_root_observation(
        const ObservationLayout& layout,
        int base_player,
        float* output,
        int output_size) const;

    void root_visits(std::uint32_t* visits, int visit_count) const;
    void root_action_values(double* values, int value_count) const;
    WDL root_wdl() const;
    MCTSSearchMetrics metrics() const;
    void reset_metrics();

    bool advance(int action_index);
    bool advance_to(int action_index, const PackedGame& observed_game);
    void add_root_dirichlet_noise(double alpha, double fraction);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class SearchPool {
public:
    SearchPool(
        ObservationLayout layout,
        std::vector<std::unique_ptr<MCTSSearch>> searches);

    int size() const;
    int observation_size() const;
    MCTSSearch& search(int index);
    const MCTSSearch& search(int index) const;

    void initialize_roots(const float* policy_logits, int policy_stride, int policy_size);
    void set_root_network_wdls(const double* wdls, int wdl_stride);
    void enable_completed_q_selection(double c_visit = 50.0, double c_scale = 1.0);
    void add_root_dirichlet_noise(double alpha, double fraction);

    void add_simulations(int index, int count);
    void add_simulations_all(int count);
    int remaining_simulations() const;

    // Eat terminal backups internally. Write at most `capacity` pending neural
    // leaves into contiguous observation rows. Returns how many leaves need
    // evaluation. Extra pending leaves stay queued for the next call.
    int select_leaves(
        int capacity,
        float* observations,
        int observation_stride,
        int* players,
        int* tokens);

    void evaluate_leaves(
        const int* tokens,
        int count,
        const float* policy_logits,
        int policy_stride,
        int policy_size,
        const double* wdls,
        int wdl_stride);

    bool advance(int index, int action_index);
    bool advance_to(int index, int action_index, const PackedGame& observed_game);

private:
    ObservationLayout layout_{};
    std::vector<std::unique_ptr<MCTSSearch>> searches_{};
    std::vector<int> remaining_{};
};

}  // namespace cudanatron
