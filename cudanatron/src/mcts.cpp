#include "cudanatron/mcts.hpp"

#include "cudanatron/chance.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace cudanatron {
namespace {

constexpr double kProbabilityTolerance = 1e-12;

void normalize_probabilities(ChanceOutcome* outcomes, int count) {
    double total = 0.0;
    for (int i = 0; i < count; ++i) {
        total += outcomes[i].probability;
    }
    if (count <= 0 || total <= kProbabilityTolerance) {
        throw std::logic_error("action produced no positive-probability outcomes");
    }
    for (int i = 0; i < count; ++i) {
        outcomes[i].probability = static_cast<float>(
            static_cast<double>(outcomes[i].probability) / total);
    }
}

WDL normalized_wdl(WDL wdl) {
    if (!std::isfinite(wdl.win) || !std::isfinite(wdl.draw) || !std::isfinite(wdl.loss) ||
        wdl.win < 0.0 || wdl.draw < 0.0 || wdl.loss < 0.0) {
        throw std::invalid_argument("WDL probabilities must be finite and non-negative");
    }
    const double total = wdl.win + wdl.draw + wdl.loss;
    if (total <= kProbabilityTolerance) {
        throw std::invalid_argument("WDL probabilities must have positive mass");
    }
    wdl.win /= total;
    wdl.draw /= total;
    wdl.loss /= total;
    return wdl;
}

void add_scaled(WDL& target, const WDL& source, double weight) {
    target.win += weight * source.win;
    target.draw += weight * source.draw;
    target.loss += weight * source.loss;
}

WDL scaled(WDL source, double weight) {
    source.win *= weight;
    source.draw *= weight;
    source.loss *= weight;
    return source;
}

bool is_stochastic(const PackedGame& game, const PackedAction& action) {
    if (action.type == ActionType::roll || action.type == ActionType::buy_development_card) {
        return true;
    }
    if (action.type != ActionType::move_robber || action.robber_victim < 0) {
        return false;
    }
    return num_resource_cards(game, action.robber_victim) > 0;
}

}  // namespace

WDL WDL::from_value(double value) {
    const double bounded = std::clamp(value, -1.0, 1.0);
    return {(1.0 + bounded) * 0.5, 0.0, (1.0 - bounded) * 0.5};
}

struct MCTSSearch::Impl {
    struct Node;

    struct OutcomeEdge {
        std::unique_ptr<Node> child;
        double probability{0.0};
    };

    struct ActionEdge {
        PackedAction action{};
        int action_index{0};
        double logit{0.0};
        double network_prior{0.0};
        double prior{0.0};
        bool stochastic{false};
        std::vector<OutcomeEdge> outcomes;

        std::uint32_t visits() const {
            std::uint32_t total = 0;
            for (const auto& outcome : outcomes) {
                total += outcome.child->visits;
            }
            return total;
        }
    };

    struct Node {
        explicit Node(PackedGame state, int minimum_discard = -1)
            : game(state),
              to_play(state.current_player_index),
              min_discard_resource(minimum_discard) {}

        PackedGame game{};
        int to_play{0};
        std::uint32_t visits{0};
        double value_sum{0.0};
        WDL wdl_sum{0.0, 0.0, 0.0};
        double network_value{0.0};
        WDL network_wdl{};
        double completed_value{0.0};
        WDL completed_wdl{};
        bool has_network_value{false};
        bool expanded{false};
        int min_discard_resource{-1};
        std::vector<ActionEdge> actions;

        double value() const { return visits == 0 ? 0.0 : value_sum / static_cast<double>(visits); }

        WDL wdl() const {
            return visits == 0 ? WDL{} : scaled(wdl_sum, 1.0 / static_cast<double>(visits));
        }
    };

    Impl(
        PackedMap packed_map,
        PackedGame root_game,
        FlatActionSpace space,
        double exploration,
        std::uint64_t seed,
        bool prune)
        : map(packed_map),
          action_space(space),
          root(root_game),
          c_puct(exploration),
          random(seed),
          canonical_pruning(prune) {
        if (!std::isfinite(c_puct) || c_puct < 0.0) {
            throw std::invalid_argument("c_puct must be finite and non-negative");
        }
    }

    PackedMap map{};
    FlatActionSpace action_space{};
    Node root;
    double c_puct{1.5};
    std::mt19937_64 random;
    bool canonical_pruning{false};
    Node* pending_leaf{nullptr};
    std::vector<Node*> pending_path;
    std::uint64_t completed_simulations{0};
    std::uint64_t depth_sum{0};
    std::uint32_t maximum_depth{0};
    std::uint32_t retained_root_visits{0};
    std::uint64_t pruned_actions{0};
    std::uint64_t coalesced_outcomes{0};
    bool tree_reused{false};
    bool completed_q_selection{false};
    double c_visit{50.0};
    double c_scale{1.0};
    double q_min{0.0};
    double q_max{0.0};
    std::vector<double> selection_scratch;

    static int child_minimum_discard(
        const Node& parent,
        const PackedAction& action,
        const PackedGame& child) {
        if (action.type != ActionType::discard_resource ||
            child.current_prompt != ActionPrompt::discard ||
            child.current_player_index != parent.to_play) {
            return -1;
        }
        return std::max(parent.min_discard_resource, static_cast<int>(action.resource));
    }

    static bool equivalent_edges(const ActionEdge& lhs, const ActionEdge& rhs) {
        if (lhs.stochastic != rhs.stochastic || lhs.outcomes.size() != rhs.outcomes.size()) {
            return false;
        }
        for (std::size_t i = 0; i < lhs.outcomes.size(); ++i) {
            if (std::abs(lhs.outcomes[i].probability - rhs.outcomes[i].probability) >
                    kProbabilityTolerance ||
                !search_equivalent(lhs.outcomes[i].child->game, rhs.outcomes[i].child->game)) {
                return false;
            }
        }
        return true;
    }

    void materialize_outcomes(Node& node, ActionEdge& edge, bool record_coalescing) {
        if (!edge.outcomes.empty()) {
            return;
        }
        ChanceOutcome chance[kMaxChanceOutcomes];
        int outcome_count =
            enumerate_chance_outcomes(map, node.game, edge.action, chance, kMaxChanceOutcomes);
        if (outcome_count > kMaxChanceOutcomes) {
            outcome_count = kMaxChanceOutcomes;
        }
        normalize_probabilities(chance, outcome_count);
        edge.outcomes.reserve(static_cast<std::size_t>(outcome_count));
        for (int i = 0; i < outcome_count; ++i) {
            PackedGame next = node.game;
            const Status status = execute_action(map, &next, edge.action, &chance[i].replay);
            if (status != Status::ok) {
                throw std::logic_error("chance outcome could not be executed");
            }
            auto duplicate = std::find_if(
                edge.outcomes.begin(),
                edge.outcomes.end(),
                [&](const OutcomeEdge& existing) {
                    return search_equivalent(existing.child->game, next);
                });
            if (canonical_pruning && duplicate != edge.outcomes.end()) {
                duplicate->probability += chance[i].probability;
                if (record_coalescing) {
                    ++coalesced_outcomes;
                }
                continue;
            }
            const int minimum_discard =
                canonical_pruning ? child_minimum_discard(node, edge.action, next) : -1;
            edge.outcomes.push_back(
                {std::make_unique<Node>(next, minimum_discard), chance[i].probability});
        }
    }

    void clear_metrics() {
        completed_simulations = 0;
        depth_sum = 0;
        maximum_depth = 0;
        retained_root_visits = root.visits;
        pruned_actions = 0;
        coalesced_outcomes = 0;
        q_min = 0.0;
        q_max = 0.0;
    }

    static bool canonical_discard_action_allowed(const Node& node, const PackedAction& action) {
        const int resource = action.resource;
        if (resource < node.min_discard_resource) {
            return false;
        }
        const auto& hand = node.game.players[action.player].resources;
        int suffix_cards = 0;
        for (int i = resource; i < kResourceCount; ++i) {
            suffix_cards += hand[i];
        }
        return suffix_cards >= remaining_discards(node.game, action.player);
    }

    void validate_logits(const float* logits, int policy_size) const {
        if (logits == nullptr || policy_size != action_space.size) {
            throw std::invalid_argument("policy logits have incorrect size");
        }
    }

    static bool is_trade_action(ActionType type) {
        return type == ActionType::offer_trade || type == ActionType::accept_trade ||
               type == ActionType::reject_trade || type == ActionType::confirm_trade ||
               type == ActionType::cancel_trade;
    }

    void expand(Node& node, const float* logits, int policy_size) {
        validate_logits(logits, policy_size);
        if (node.expanded) {
            throw std::logic_error("cannot expand a node twice");
        }
        if (winning_player(node.game) >= 0) {
            throw std::logic_error("cannot expand a terminal node");
        }

        PackedAction legal[kMaxLegalActions];
        const int legal_count = generate_legal_actions(map, node.game, legal, kMaxLegalActions);
        if (legal_count <= 0) {
            node.expanded = true;
            return;
        }

        std::vector<int> playable;
        playable.reserve(static_cast<std::size_t>(legal_count));
        for (int i = 0; i < legal_count; ++i) {
            if (is_trade_action(legal[i].type)) {
                continue;
            }
            if (canonical_pruning && node.game.current_prompt == ActionPrompt::discard &&
                legal[i].type == ActionType::discard_resource &&
                !canonical_discard_action_allowed(node, legal[i])) {
                ++pruned_actions;
                continue;
            }
            playable.push_back(i);
        }
        if (playable.empty()) {
            throw std::logic_error("canonical pruning removed every legal action");
        }

        std::vector<int> indices;
        indices.reserve(playable.size());
        float max_logit = -std::numeric_limits<float>::infinity();
        for (int legal_index : playable) {
            const int index = flat_index(action_space, map, node.game, legal[legal_index]);
            if (index < 0) {
                throw std::logic_error("legal action is absent from the flat action space");
            }
            indices.push_back(index);
            max_logit = std::max(max_logit, logits[index]);
        }

        std::vector<double> weights;
        weights.reserve(indices.size());
        double weight_sum = 0.0;
        for (int index : indices) {
            const double weight = std::isfinite(logits[index])
                                      ? std::exp(
                                            static_cast<double>(logits[index]) -
                                            static_cast<double>(max_logit))
                                      : 0.0;
            weights.push_back(weight);
            weight_sum += weight;
        }
        if (!std::isfinite(weight_sum) || weight_sum <= kProbabilityTolerance) {
            std::fill(weights.begin(), weights.end(), 1.0);
            weight_sum = static_cast<double>(weights.size());
        }

        node.actions.reserve(playable.size());
        for (std::size_t i = 0; i < playable.size(); ++i) {
            ActionEdge edge;
            edge.action = legal[playable[i]];
            edge.action_index = indices[i];
            edge.logit = static_cast<double>(logits[indices[i]]);
            edge.network_prior = weights[i] / weight_sum;
            edge.prior = weights[i] / weight_sum;
            edge.stochastic = is_stochastic(node.game, edge.action);
            if (canonical_pruning) {
                materialize_outcomes(node, edge, true);
                auto duplicate = std::find_if(
                    node.actions.begin(),
                    node.actions.end(),
                    [&](const ActionEdge& existing) { return equivalent_edges(existing, edge); });
                if (duplicate != node.actions.end()) {
                    duplicate->prior += edge.prior;
                    ++pruned_actions;
                    continue;
                }
            }
            node.actions.push_back(std::move(edge));
        }
        for (auto& action : node.actions) {
            action.outcomes.clear();
        }
        for (auto& action : node.actions) {
            action.network_prior = action.prior;
            action.logit = std::log(std::max(action.network_prior, kProbabilityTolerance));
        }
        node.expanded = true;
    }

    double node_value(const Node& node) const {
        return completed_q_selection && node.has_network_value ? node.completed_value
                                                               : node.value();
    }

    WDL node_wdl(const Node& node) const {
        return completed_q_selection && node.has_network_value ? node.completed_wdl : node.wdl();
    }

    WDL action_wdl(const Node& node, const ActionEdge& action) const {
        WDL result{0.0, 0.0, 0.0};
        if (completed_q_selection && action.stochastic && action.visits() > 0) {
            for (const auto& outcome : action.outcomes) {
                const Node& child = *outcome.child;
                const WDL oriented =
                    child.to_play == node.to_play ? node_wdl(child) : node_wdl(child).flipped();
                add_scaled(result, oriented, static_cast<double>(child.visits));
            }
            return scaled(result, 1.0 / static_cast<double>(action.visits()));
        }
        for (const auto& outcome : action.outcomes) {
            const Node& child = *outcome.child;
            const WDL oriented =
                child.to_play == node.to_play ? node_wdl(child) : node_wdl(child).flipped();
            add_scaled(result, oriented, outcome.probability);
        }
        return result;
    }

    double action_value(const Node& node, const ActionEdge& action) const {
        return action_wdl(node, action).q();
    }

    void recompute_completed_value(Node& node) const {
        if (!node.has_network_value) {
            return;
        }
        std::uint32_t total_visits = 0;
        WDL backed_up_wdl{0.0, 0.0, 0.0};
        for (const auto& action : node.actions) {
            total_visits += action.visits();
            add_scaled(backed_up_wdl, action_wdl(node, action), static_cast<double>(action.visits()));
        }
        WDL completed = node.network_wdl;
        add_scaled(completed, backed_up_wdl, 1.0);
        node.completed_wdl = scaled(completed, 1.0 / (1.0 + static_cast<double>(total_visits)));
        node.completed_value = node.completed_wdl.q();
    }

    double value_mix(const Node& node) const {
        std::uint32_t total_visits = 0;
        for (const auto& action : node.actions) {
            total_visits += action.visits();
        }
        const double network_value = node.has_network_value ? node.network_value : 0.0;
        if (total_visits == 0) {
            return network_value;
        }
        double prior_weighted_q = 0.0;
        double visited_prior_sum = 0.0;
        for (const auto& action : node.actions) {
            if (action.visits() == 0) {
                continue;
            }
            prior_weighted_q += action.network_prior * action_value(node, action);
            visited_prior_sum += action.network_prior;
        }
        const double weighted_q = visited_prior_sum > kProbabilityTolerance
                                      ? prior_weighted_q / visited_prior_sum
                                      : 0.0;
        return (network_value + static_cast<double>(total_visits) * weighted_q) /
               (1.0 + static_cast<double>(total_visits));
    }

    double completed_q(const Node& node, const ActionEdge& action, double mixed_value) const {
        return action.visits() == 0 ? mixed_value : action_value(node, action);
    }

    double normalize_q(const Node& node, double q_value) const {
        const bool root_perspective = node.to_play == root.to_play;
        const double lower = root_perspective ? q_min : -q_max;
        const double upper = root_perspective ? q_max : -q_min;
        const double range = upper - lower;
        if (range <= std::numeric_limits<double>::epsilon()) {
            return 0.5;
        }
        return std::clamp((q_value - lower) / range, 0.0, 1.0);
    }

    ActionEdge& select_puct_action(Node& node) {
        std::uint32_t total_visits = 0;
        for (const auto& action : node.actions) {
            total_visits += action.visits();
        }
        const double mixed_value = completed_q_selection ? value_mix(node) : 0.0;
        ActionEdge* best_action = nullptr;
        double best_score = -std::numeric_limits<double>::infinity();
        for (auto& action : node.actions) {
            const std::uint32_t action_visits = action.visits();
            const double q_value = completed_q_selection
                                       ? completed_q(node, action, mixed_value)
                                       : (action_visits == 0 ? 0.0 : action_value(node, action));
            const double u_value =
                c_puct * action.prior *
                std::sqrt(
                    static_cast<double>(total_visits) + (completed_q_selection ? 0.0 : 1.0)) /
                (1.0 + static_cast<double>(action_visits));
            const double score = q_value + u_value;
            if (score > best_score) {
                best_score = score;
                best_action = &action;
            }
        }
        if (best_action == nullptr) {
            throw std::logic_error("expanded search node has no outcomes");
        }
        return *best_action;
    }

    ActionEdge& select_interior_action(Node& node) {
        std::uint32_t total_visits = 0;
        std::uint32_t max_visits = 0;
        for (const auto& action : node.actions) {
            const std::uint32_t visits = action.visits();
            total_visits += visits;
            max_visits = std::max(max_visits, visits);
        }
        const double mixed_value = value_mix(node);
        selection_scratch.clear();
        selection_scratch.reserve(node.actions.size());
        double max_logit = -std::numeric_limits<double>::infinity();
        for (const auto& action : node.actions) {
            const double q_value = completed_q(node, action, mixed_value);
            const double score =
                action.logit +
                (c_visit + static_cast<double>(max_visits)) * c_scale * normalize_q(node, q_value);
            selection_scratch.push_back(score);
            max_logit = std::max(max_logit, score);
        }
        double probability_sum = 0.0;
        for (double& value : selection_scratch) {
            value = std::exp(value - max_logit);
            probability_sum += value;
        }
        if (!std::isfinite(probability_sum) || probability_sum <= kProbabilityTolerance) {
            std::fill(selection_scratch.begin(), selection_scratch.end(), 1.0);
            probability_sum = static_cast<double>(selection_scratch.size());
        }

        ActionEdge* best_action = nullptr;
        double best_score = -std::numeric_limits<double>::infinity();
        const double denominator = 1.0 + static_cast<double>(total_visits);
        for (std::size_t i = 0; i < node.actions.size(); ++i) {
            auto& action = node.actions[i];
            const double score = selection_scratch[i] / probability_sum -
                                 static_cast<double>(action.visits()) / denominator;
            if (score > best_score) {
                best_score = score;
                best_action = &action;
            }
        }
        if (best_action == nullptr) {
            throw std::logic_error("expanded search node has no outcomes");
        }
        return *best_action;
    }

    Node& select_child(Node& node) {
        ActionEdge& best_action = completed_q_selection && &node != &root
                                      ? select_interior_action(node)
                                      : select_puct_action(node);
        if (best_action.outcomes.empty()) {
            materialize_outcomes(node, best_action, false);
        }
        std::vector<double> probabilities;
        probabilities.reserve(best_action.outcomes.size());
        for (const auto& outcome : best_action.outcomes) {
            probabilities.push_back(outcome.probability);
        }
        std::discrete_distribution<std::size_t> distribution(
            probabilities.begin(), probabilities.end());
        return *best_action.outcomes[distribution(random)].child;
    }

    void backup(const std::vector<Node*>& path, WDL leaf_wdl) {
        WDL wdl = normalized_wdl(leaf_wdl);
        for (std::size_t index = path.size(); index-- > 0;) {
            Node& node = *path[index];
            ++node.visits;
            add_scaled(node.wdl_sum, wdl, 1.0);
            node.value_sum += wdl.q();
            if (index > 0 && path[index - 1]->to_play != node.to_play) {
                wdl = wdl.flipped();
            }
        }
        if (completed_q_selection) {
            for (std::size_t index = path.size(); index-- > 0;) {
                recompute_completed_value(*path[index]);
            }
        }
        for (const Node* node : path) {
            const double root_value =
                node->to_play == root.to_play ? node_value(*node) : -node_value(*node);
            q_min = std::min(q_min, root_value);
            q_max = std::max(q_max, root_value);
        }
    }

    void record_simulation(const std::vector<Node*>& path) {
        const auto depth =
            path.empty() ? std::uint32_t{0} : static_cast<std::uint32_t>(path.size() - 1);
        ++completed_simulations;
        depth_sum += depth;
        maximum_depth = std::max(maximum_depth, depth);
    }

    std::uint32_t principal_variation_depth() const {
        const Node* node = &root;
        std::uint32_t depth = 0;
        while (node->expanded && !node->actions.empty()) {
            auto best_action = std::max_element(
                node->actions.begin(),
                node->actions.end(),
                [](const ActionEdge& lhs, const ActionEdge& rhs) {
                    return lhs.visits() < rhs.visits();
                });
            if (best_action == node->actions.end() || best_action->visits() == 0) {
                break;
            }
            auto best_outcome = std::max_element(
                best_action->outcomes.begin(),
                best_action->outcomes.end(),
                [](const OutcomeEdge& lhs, const OutcomeEdge& rhs) {
                    return lhs.child->visits < rhs.child->visits;
                });
            if (best_outcome == best_action->outcomes.end() || best_outcome->child->visits == 0) {
                break;
            }
            node = best_outcome->child.get();
            ++depth;
        }
        return depth;
    }
};

MCTSSearch::MCTSSearch(
    PackedMap map,
    PackedGame root_game,
    FlatActionSpace action_space,
    double c_puct,
    std::uint64_t seed,
    bool canonical_pruning)
    : impl_(std::make_unique<Impl>(
          map,
          root_game,
          action_space,
          c_puct,
          seed,
          canonical_pruning)) {}

MCTSSearch::~MCTSSearch() = default;
MCTSSearch::MCTSSearch(MCTSSearch&&) noexcept = default;
MCTSSearch& MCTSSearch::operator=(MCTSSearch&&) noexcept = default;

void MCTSSearch::initialize_root(const float* policy_logits, int policy_size) {
    if (impl_->pending_leaf != nullptr) {
        throw std::logic_error("cannot initialize root with a pending leaf");
    }
    impl_->expand(impl_->root, policy_logits, policy_size);
}

void MCTSSearch::set_root_network_value(double value) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument("root network value must be finite");
    }
    set_root_network_wdl(WDL::from_value(value));
}

void MCTSSearch::set_root_network_wdl(WDL wdl) {
    const WDL normalized = normalized_wdl(wdl);
    impl_->root.network_wdl = normalized;
    impl_->root.network_value = normalized.q();
    impl_->root.completed_wdl = normalized;
    impl_->root.completed_value = normalized.q();
    impl_->root.has_network_value = true;
    impl_->q_min = std::min(impl_->q_min, impl_->root.network_value);
    impl_->q_max = std::max(impl_->q_max, impl_->root.network_value);
}

void MCTSSearch::enable_completed_q_selection(double c_visit, double c_scale) {
    if (!std::isfinite(c_visit) || c_visit < 0.0 || !std::isfinite(c_scale) || c_scale < 0.0) {
        throw std::invalid_argument(
            "completed-Q selection parameters must be finite and non-negative");
    }
    impl_->completed_q_selection = true;
    impl_->c_visit = c_visit;
    impl_->c_scale = c_scale;
}

bool MCTSSearch::select_leaf() {
    if (impl_->pending_leaf != nullptr) {
        throw std::logic_error("previous leaf has not been evaluated");
    }
    if (!impl_->root.expanded) {
        throw std::logic_error("search root has not been initialized");
    }

    Impl::Node* node = &impl_->root;
    std::vector<Impl::Node*> path{node};
    while (node->expanded && winning_player(node->game) < 0 && !node->actions.empty()) {
        node = &impl_->select_child(*node);
        path.push_back(node);
    }

    const int winner = winning_player(node->game);
    if (winner >= 0) {
        impl_->record_simulation(path);
        impl_->backup(
            path,
            winner == node->to_play ? WDL{1.0, 0.0, 0.0} : WDL{0.0, 0.0, 1.0});
        return false;
    }
    if (node->expanded && node->actions.empty()) {
        impl_->record_simulation(path);
        impl_->backup(path, WDL{});
        return false;
    }

    impl_->pending_leaf = node;
    impl_->pending_path = std::move(path);
    return true;
}

void MCTSSearch::evaluate_leaf(const float* policy_logits, int policy_size, double value) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument("leaf value must be finite");
    }
    evaluate_leaf_wdl(policy_logits, policy_size, WDL::from_value(value));
}

void MCTSSearch::evaluate_leaf_wdl(const float* policy_logits, int policy_size, WDL wdl) {
    if (impl_->pending_leaf == nullptr) {
        throw std::logic_error("no pending search leaf");
    }
    const WDL normalized = normalized_wdl(wdl);
    impl_->pending_leaf->network_wdl = normalized;
    impl_->pending_leaf->network_value = normalized.q();
    impl_->pending_leaf->completed_wdl = normalized;
    impl_->pending_leaf->completed_value = normalized.q();
    impl_->pending_leaf->has_network_value = true;
    const double root_value = impl_->pending_leaf->to_play == impl_->root.to_play
                                  ? impl_->pending_leaf->network_value
                                  : -impl_->pending_leaf->network_value;
    impl_->q_min = std::min(impl_->q_min, root_value);
    impl_->q_max = std::max(impl_->q_max, root_value);
    impl_->expand(*impl_->pending_leaf, policy_logits, policy_size);
    impl_->record_simulation(impl_->pending_path);
    impl_->backup(impl_->pending_path, normalized);
    impl_->pending_leaf = nullptr;
    impl_->pending_path.clear();
}

bool MCTSSearch::has_pending_leaf() const { return impl_->pending_leaf != nullptr; }

bool MCTSSearch::root_expanded() const { return impl_->root.expanded; }

int MCTSSearch::pending_player_index() const {
    if (impl_->pending_leaf == nullptr) {
        throw std::logic_error("no pending search leaf");
    }
    return impl_->pending_leaf->to_play;
}

const PackedGame& MCTSSearch::pending_game() const {
    if (impl_->pending_leaf == nullptr) {
        throw std::logic_error("no pending search leaf");
    }
    return impl_->pending_leaf->game;
}

const PackedGame& MCTSSearch::root_game() const { return impl_->root.game; }

const PackedMap& MCTSSearch::map() const { return impl_->map; }

const FlatActionSpace& MCTSSearch::action_space() const { return impl_->action_space; }

void MCTSSearch::write_pending_observation(
    const ObservationLayout& layout,
    float* output,
    int output_size) const {
    const Status status = write_full_observation(
        impl_->map,
        pending_game(),
        pending_player_index(),
        layout,
        output,
        output_size);
    if (status != Status::ok) {
        throw std::runtime_error("failed to write pending observation");
    }
}

void MCTSSearch::write_root_observation(
    const ObservationLayout& layout,
    int base_player,
    float* output,
    int output_size) const {
    const Status status = write_full_observation(
        impl_->map,
        impl_->root.game,
        base_player,
        layout,
        output,
        output_size);
    if (status != Status::ok) {
        throw std::runtime_error("failed to write root observation");
    }
}

void MCTSSearch::root_visits(std::uint32_t* visits, int visit_count) const {
    if (visits == nullptr || visit_count != impl_->action_space.size) {
        throw std::invalid_argument("root visit buffer has incorrect size");
    }
    std::fill(visits, visits + visit_count, 0U);
    for (const auto& action : impl_->root.actions) {
        visits[action.action_index] = action.visits();
    }
}

void MCTSSearch::root_action_values(double* values, int value_count) const {
    if (values == nullptr || value_count != impl_->action_space.size) {
        throw std::invalid_argument("root value buffer has incorrect size");
    }
    std::fill(
        values,
        values + value_count,
        std::numeric_limits<double>::quiet_NaN());
    for (const auto& action : impl_->root.actions) {
        if (action.visits() == 0) {
            continue;
        }
        values[action.action_index] = impl_->action_value(impl_->root, action);
    }
}

WDL MCTSSearch::root_wdl() const { return impl_->node_wdl(impl_->root); }

MCTSSearchMetrics MCTSSearch::metrics() const {
    return {
        impl_->completed_simulations,
        impl_->principal_variation_depth(),
        impl_->maximum_depth,
        impl_->completed_simulations == 0
            ? 0.0
            : static_cast<double>(impl_->depth_sum) /
                  static_cast<double>(impl_->completed_simulations),
        impl_->node_value(impl_->root),
        impl_->retained_root_visits,
        impl_->pruned_actions,
        impl_->coalesced_outcomes,
        impl_->tree_reused,
        impl_->q_min,
        impl_->q_max,
    };
}

void MCTSSearch::reset_metrics() { impl_->clear_metrics(); }

bool MCTSSearch::advance(int action_index) {
    if (impl_->pending_leaf != nullptr) {
        throw std::logic_error("cannot advance with a pending leaf");
    }
    if (!impl_->root.expanded) {
        return false;
    }
    auto edge = std::find_if(
        impl_->root.actions.begin(),
        impl_->root.actions.end(),
        [action_index](const Impl::ActionEdge& action) {
            return action.action_index == action_index;
        });
    if (edge == impl_->root.actions.end() || edge->stochastic) {
        return false;
    }
    impl_->materialize_outcomes(impl_->root, *edge, false);
    if (edge->outcomes.size() != 1) {
        return false;
    }
    std::unique_ptr<Impl::Node> next = std::move(edge->outcomes.front().child);
    impl_->root = std::move(*next);
    impl_->tree_reused = true;
    impl_->clear_metrics();
    return true;
}

bool MCTSSearch::advance_to(int action_index, const PackedGame& observed_game) {
    if (impl_->pending_leaf != nullptr) {
        throw std::logic_error("cannot advance with a pending leaf");
    }
    if (!impl_->root.expanded) {
        return false;
    }
    auto edge = std::find_if(
        impl_->root.actions.begin(),
        impl_->root.actions.end(),
        [action_index](const Impl::ActionEdge& action) {
            return action.action_index == action_index;
        });
    if (edge == impl_->root.actions.end()) {
        return false;
    }
    impl_->materialize_outcomes(impl_->root, *edge, false);
    auto outcome = std::find_if(
        edge->outcomes.begin(),
        edge->outcomes.end(),
        [&observed_game](const Impl::OutcomeEdge& candidate) {
            return search_equivalent(candidate.child->game, observed_game);
        });
    if (outcome == edge->outcomes.end()) {
        return false;
    }
    std::unique_ptr<Impl::Node> next = std::move(outcome->child);
    impl_->root = std::move(*next);
    impl_->tree_reused = true;
    impl_->clear_metrics();
    return true;
}

void MCTSSearch::add_root_dirichlet_noise(double alpha, double fraction) {
    if (!impl_->root.expanded) {
        throw std::logic_error("cannot add noise before root initialization");
    }
    if (!std::isfinite(alpha) || alpha <= 0.0) {
        throw std::invalid_argument("Dirichlet alpha must be positive and finite");
    }
    if (!std::isfinite(fraction) || fraction < 0.0 || fraction > 1.0) {
        throw std::invalid_argument("Dirichlet fraction must be in [0, 1]");
    }
    if (impl_->root.actions.empty() || fraction == 0.0) {
        return;
    }

    std::gamma_distribution<double> gamma(alpha, 1.0);
    std::vector<double> noise;
    noise.reserve(impl_->root.actions.size());
    for (std::size_t i = 0; i < impl_->root.actions.size(); ++i) {
        noise.push_back(gamma(impl_->random));
    }
    double total = std::accumulate(noise.begin(), noise.end(), 0.0);
    if (total <= kProbabilityTolerance) {
        std::fill(noise.begin(), noise.end(), 1.0);
        total = static_cast<double>(noise.size());
    }
    for (std::size_t i = 0; i < impl_->root.actions.size(); ++i) {
        auto& action = impl_->root.actions[i];
        action.prior = (1.0 - fraction) * action.prior + fraction * noise[i] / total;
    }
}

SearchPool::SearchPool(
    ObservationLayout layout,
    std::vector<std::unique_ptr<MCTSSearch>> searches)
    : layout_(layout), searches_(std::move(searches)), remaining_(searches_.size(), 0) {
    if (searches_.empty()) {
        throw std::invalid_argument("search pool requires at least one search");
    }
}

int SearchPool::size() const { return static_cast<int>(searches_.size()); }

int SearchPool::observation_size() const {
    const int num_players = searches_.front()->root_game().num_players;
    return full_observation_size(num_players, layout_.width, layout_.height);
}

MCTSSearch& SearchPool::search(int index) { return *searches_.at(static_cast<std::size_t>(index)); }

const MCTSSearch& SearchPool::search(int index) const {
    return *searches_.at(static_cast<std::size_t>(index));
}

void SearchPool::initialize_roots(
    const float* policy_logits,
    int policy_stride,
    int policy_size) {
    if (policy_logits == nullptr) {
        throw std::invalid_argument("root policy logits are null");
    }
    const int n = size();
    std::vector<std::string> errors(static_cast<std::size_t>(n));
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < n; ++i) {
        try {
            searches_[static_cast<std::size_t>(i)]->initialize_root(
                policy_logits + static_cast<std::ptrdiff_t>(i) * policy_stride,
                policy_size);
        } catch (const std::exception& error) {
            errors[static_cast<std::size_t>(i)] = error.what();
        }
    }
    for (const auto& error : errors) {
        if (!error.empty()) {
            throw std::runtime_error(error);
        }
    }
}

void SearchPool::set_root_network_wdls(const double* wdls, int wdl_stride) {
    if (wdls == nullptr || wdl_stride < 3) {
        throw std::invalid_argument("root WDL buffer is invalid");
    }
    for (int i = 0; i < size(); ++i) {
        const double* row = wdls + static_cast<std::ptrdiff_t>(i) * wdl_stride;
        searches_[static_cast<std::size_t>(i)]->set_root_network_wdl(
            WDL{row[0], row[1], row[2]});
    }
}

void SearchPool::enable_completed_q_selection(double c_visit, double c_scale) {
    for (auto& search : searches_) {
        search->enable_completed_q_selection(c_visit, c_scale);
    }
}

void SearchPool::add_root_dirichlet_noise(double alpha, double fraction) {
    for (auto& search : searches_) {
        search->add_root_dirichlet_noise(alpha, fraction);
    }
}

void SearchPool::add_simulations(int index, int count) {
    if (index < 0 || index >= size() || count < 0) {
        throw std::out_of_range("invalid search pool simulation request");
    }
    remaining_[static_cast<std::size_t>(index)] += count;
}

void SearchPool::add_simulations_all(int count) {
    if (count < 0) {
        throw std::invalid_argument("simulation count must be non-negative");
    }
    for (int& remaining : remaining_) {
        remaining += count;
    }
}

int SearchPool::remaining_simulations() const {
    return std::accumulate(remaining_.begin(), remaining_.end(), 0);
}

int SearchPool::select_leaves(
    int capacity,
    float* observations,
    int observation_stride,
    int* players,
    int* tokens) {
    if (capacity < 0 || observations == nullptr || players == nullptr || tokens == nullptr) {
        throw std::invalid_argument("select_leaves received invalid buffers");
    }
    const int obs_size = observation_size();
    if (observation_stride < obs_size) {
        throw std::invalid_argument("observation stride is smaller than one observation");
    }

    const int n = size();
    std::vector<std::string> errors(static_cast<std::size_t>(n));
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic)
#endif
    for (int i = 0; i < n; ++i) {
        try {
            auto& search = *searches_[static_cast<std::size_t>(i)];
            while (remaining_[static_cast<std::size_t>(i)] > 0 && !search.has_pending_leaf()) {
                if (!search.select_leaf()) {
                    --remaining_[static_cast<std::size_t>(i)];
                }
            }
        } catch (const std::exception& error) {
            errors[static_cast<std::size_t>(i)] = error.what();
        }
    }
    for (const auto& error : errors) {
        if (!error.empty()) {
            throw std::runtime_error(error);
        }
    }

    int filled = 0;
    for (int i = 0; i < n && filled < capacity; ++i) {
        auto& search = *searches_[static_cast<std::size_t>(i)];
        if (!search.has_pending_leaf()) {
            continue;
        }
        search.write_pending_observation(
            layout_,
            observations + static_cast<std::ptrdiff_t>(filled) * observation_stride,
            obs_size);
        players[filled] = search.pending_player_index();
        tokens[filled] = i;
        ++filled;
    }
    return filled;
}

void SearchPool::evaluate_leaves(
    const int* tokens,
    int count,
    const float* policy_logits,
    int policy_stride,
    int policy_size,
    const double* wdls,
    int wdl_stride) {
    if (tokens == nullptr || policy_logits == nullptr || wdls == nullptr || count < 0) {
        throw std::invalid_argument("evaluate_leaves received invalid buffers");
    }
    if (wdl_stride < 3) {
        throw std::invalid_argument("WDL stride must be at least 3");
    }
    std::vector<std::string> errors(static_cast<std::size_t>(count));
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < count; ++i) {
        try {
            const int token = tokens[i];
            if (token < 0 || token >= size()) {
                throw std::out_of_range("search token is out of range");
            }
            const double* wdl = wdls + static_cast<std::ptrdiff_t>(i) * wdl_stride;
            searches_[static_cast<std::size_t>(token)]->evaluate_leaf_wdl(
                policy_logits + static_cast<std::ptrdiff_t>(i) * policy_stride,
                policy_size,
                WDL{wdl[0], wdl[1], wdl[2]});
            if (remaining_[static_cast<std::size_t>(token)] > 0) {
                --remaining_[static_cast<std::size_t>(token)];
            }
        } catch (const std::exception& error) {
            errors[static_cast<std::size_t>(i)] = error.what();
        }
    }
    for (const auto& error : errors) {
        if (!error.empty()) {
            throw std::runtime_error(error);
        }
    }
}

bool SearchPool::advance(int index, int action_index) {
    return search(index).advance(action_index);
}

bool SearchPool::advance_to(int index, int action_index, const PackedGame& observed_game) {
    return search(index).advance_to(action_index, observed_game);
}

}  // namespace cudanatron
