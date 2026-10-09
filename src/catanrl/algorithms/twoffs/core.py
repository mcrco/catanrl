"""Two-fidelity fast-slow (2FFS) best-action identification on stochastic minimax trees.

Implements 2FFS from Peter Chen and Xi Chen, "Two-Fidelity Best-Action
Identification for Stochastic Minimax Tree" (NeurIPS 2026, arXiv:2606.01708).
The control flow (root leader/challenger obligations, side certificates, dyadic
scales, capped recursive scales, budgeted race between expansion and local slow
sampling) follows the authors' reference implementation,
github.com/PeterLauLukChen/2FFS ``method/twoffs/twoffs.py``, so that the
synthetic-tree results match it exactly (see ``tests/test_twoffs_core.py``).

Each exposed node carries an interval ``[L, U]`` intersecting a deterministic
fast-oracle interval ``V_F +- B(h)``, a time-uniform slow-oracle confidence
interval, and the minimax backup of its children.

Game-search extensions beyond the paper, none of which change behaviour on the
paper's max/min trees:

* ``CHANCE`` nodes back up the probability-weighted sum of child intervals. A
  side of a chance node is certified at scale ``k`` when the probability-weighted
  width of its uncertified children is at most ``rho_k / 2`` of their mass
  (which bounds ``sum_c p_c (V*(c) - L_c)`` by ``rho_k / 2``). Its recursive
  route refines the uncertified child with the largest ``p_c * width_c``.
* Trees are generated lazily by an oracle, and children of chance nodes are
  exposed together with the chance node.
* An optional global evaluation budget makes the search anytime. When it runs
  out before the stopping rule fires, the leader (largest lower bound) is
  recommended.
* Local slow steps may draw several samples at once (``slow_batch``).
* Certificates can be latched along the changed root path only
  (``latch_all=False``) instead of over every explored node.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

Side = Literal["L", "U"]
Status = Literal["cert", "progress", "blocked"]

MAX = 0
MIN = 1
CHANCE = 2
LEAF = 3  # Not expandable: a terminal state or a depth-limit frontier.

_INF = math.inf


@dataclass
class ChildSpec:
    """An unexposed child produced by an oracle.

    ``probability`` is the transition probability when the parent is a chance
    node. ``exact_value`` short-circuits the fast oracle (terminal states).
    ``outcomes`` lists the children of a ``CHANCE`` spec.
    """

    kind: int
    remaining_depth: int
    payload: Any
    probability: float = 1.0
    exact_value: float | None = None
    outcomes: list[ChildSpec] | None = None
    key: Any = None


class TwoFidelityOracle(Protocol):
    """Tree generator plus the two evaluation fidelities."""

    fast_query_cost: float
    slow_sigma: float

    def root(self) -> tuple[Any, int]:
        """Root payload and remaining depth. The root is a ``MAX`` node."""

    def expand(self, payload: Any, kind: int, remaining_depth: int) -> list[ChildSpec]:
        """Children of an expandable ``MAX``/``MIN`` node, without fast values."""

    def fast_evaluate(self, specs: Sequence[ChildSpec]) -> list[tuple[float, float]]:
        """``(fast value, bias bound B(h))`` for each spec, in one batch."""

    def fast_bound(self, kind: int, remaining_depth: int) -> float:
        """Bias envelope ``B(h)`` used for local certification costs."""

    def slow_sample(self, payload: Any, kind: int, count: int) -> tuple[list[float], float]:
        """``count`` unbiased slow samples of the node value and their total cost."""

    def slow_cost_estimate(self) -> float:
        """Expected cost of one slow sample."""


@dataclass(frozen=True)
class TwoFFSConfig:
    delta: float = 0.05
    epsilon: float = 0.0
    max_outer_rounds: int = 200_000
    max_scale: int = 80
    alpha_power: float = 2.0
    alpha_offset: float = 1.0
    # Per-node confidence is delta / confidence_nodes. The reference divides by
    # the size of the full tree; lazily generated trees supply a cap instead.
    confidence_nodes: int | None = None
    # Additive slow-oracle bias bound xi (Theorem 3.7 inflates radii by xi).
    slow_bias: float = 0.0
    # Search values are known to lie in this range, so intervals may be clipped.
    value_range: tuple[float, float] | None = None
    slow_batch: int = 1
    budget: float = _INF
    max_nodes: int | None = None
    latch_all: bool = True
    # When the stopping rule has not fired: "leader" (largest L), "midpoint",
    # or "prior" (the oracle's prior favourite among actions not yet
    # eliminated, i.e. with U >= max L - epsilon).
    fallback: Literal["leader", "midpoint", "prior"] = "leader"


class _Node:
    __slots__ = (
        "child_l",
        "child_u",
        "children",
        "depth",
        "expanded",
        "fast_bound",
        "fast_l",
        "fast_u",
        "h",
        "index",
        "key",
        "kind",
        "l",
        "local_l",
        "local_u",
        "n_slow",
        "parent",
        "payload",
        "pending",
        "probability",
        "slow_sum",
        "u",
    )

    def __init__(
        self,
        index: int,
        parent: _Node | None,
        depth: int,
        h: int,
        kind: int,
        payload: Any,
        probability: float,
        key: Any,
    ) -> None:
        self.index = index
        self.parent = parent
        self.depth = depth
        self.h = h
        self.kind = kind
        self.payload = payload
        self.probability = probability
        self.key = key
        self.children: list[_Node] = []
        self.expanded = False
        self.pending: list[ChildSpec] | None = None
        self.fast_bound = 0.0
        self.fast_l = -_INF
        self.fast_u = _INF
        self.n_slow = 0
        self.slow_sum = 0.0
        self.local_l = -_INF
        self.local_u = _INF
        self.child_l = -_INF
        self.child_u = _INF
        self.l = -_INF
        self.u = _INF


@dataclass
class _ScaleState:
    m_loc: float = 0.0
    m_exp: float = 0.0
    done_l: bool = False
    done_u: bool = False


@dataclass
class TwoFFSResult:
    recommendation_index: int | None
    recommendation_key: Any
    stopped: bool
    stop_reason: str
    outer_rounds: int
    total_cost: float
    fast_cost: float
    slow_cost: float
    num_fast_queries: int
    num_slow_queries: int
    num_nodes: int
    num_expanded_nodes: int
    max_depth: int
    root_intervals: list[tuple[float, float]] = field(default_factory=list)
    root_keys: list[Any] = field(default_factory=list)
    root_slow_queries: list[int] = field(default_factory=list)
    plausible_root_actions: int = 0


class TwoFFS:
    """One 2FFS search from an oracle's root."""

    def __init__(self, oracle: TwoFidelityOracle, config: TwoFFSConfig | None = None) -> None:
        self.oracle = oracle
        self.config = config or TwoFFSConfig()
        self._validate_config()
        self.nodes: list[_Node] = []
        self.scale_state: dict[tuple[int, int], _ScaleState] = {}
        self.max_depth = 0
        payload, root_h = oracle.root()
        self.root = self._new_node(None, 0, root_h, MAX, payload, 1.0, None)
        confidence_nodes = max(1, int(self.config.confidence_nodes or 1))
        self.node_delta = self.config.delta / confidence_nodes
        self.rho0 = 1.0
        self.total_cost = 0.0
        self.fast_cost = 0.0
        self.slow_cost = 0.0
        self.num_fast_queries = 0
        self.num_slow_queries = 0
        self.num_expanded = 0
        self._affected: list[_Node] = []
        self._m_required_cache: dict[float, int] = {}

    def _validate_config(self) -> None:
        config = self.config
        if not (0.0 < config.delta < 1.0):
            raise ValueError("delta must lie in (0, 1)")
        if config.epsilon < 0:
            raise ValueError("epsilon must be nonnegative")
        if config.alpha_offset <= 0 or config.alpha_power < 0:
            raise ValueError("alpha_offset must be positive and alpha_power nonnegative")
        if config.slow_batch < 1:
            raise ValueError("slow_batch must be positive")
        if config.slow_bias < 0:
            raise ValueError("slow_bias must be nonnegative")

    # ------------------------------------------------------------------
    # Tree construction
    # ------------------------------------------------------------------

    def _new_node(
        self,
        parent: _Node | None,
        depth: int,
        h: int,
        kind: int,
        payload: Any,
        probability: float,
        key: Any,
    ) -> _Node:
        node = _Node(len(self.nodes), parent, depth, h, kind, payload, probability, key)
        self.nodes.append(node)
        self.max_depth = max(self.max_depth, depth)
        return node

    @staticmethod
    def _fast_specs(specs: Sequence[ChildSpec]) -> list[ChildSpec]:
        """Specs that need a fast query, descending through chance outcomes."""
        needed: list[ChildSpec] = []
        stack = list(reversed(specs))
        while stack:
            spec = stack.pop()
            if spec.kind == CHANCE:
                stack.extend(reversed(spec.outcomes or ()))
            elif spec.exact_value is None:
                needed.append(spec)
        return needed

    @staticmethod
    def _count_specs(specs: Sequence[ChildSpec]) -> int:
        return sum(
            1 + (TwoFFS._count_specs(spec.outcomes or ()) if spec.kind == CHANCE else 0)
            for spec in specs
        )

    def _children_specs(self, node: _Node) -> list[ChildSpec]:
        if node.pending is None:
            node.pending = self.oracle.expand(node.payload, node.kind, node.h)
        return node.pending

    def _expansion_cost(self, node: _Node) -> float:
        return self.oracle.fast_query_cost * len(self._fast_specs(self._children_specs(node)))

    def _expansion_nodes(self, node: _Node) -> int:
        return self._count_specs(self._children_specs(node))

    def _attach_children(self, node: _Node, specs: list[ChildSpec]) -> list[_Node]:
        """Expose ``specs`` under ``node`` with one batched fast-oracle call."""
        needed = self._fast_specs(specs)
        fast_values = self.oracle.fast_evaluate(needed) if needed else []
        fast_by_spec = {id(spec): value for spec, value in zip(needed, fast_values)}
        cost = self.oracle.fast_query_cost * len(needed)
        self.total_cost += cost
        self.fast_cost += cost
        self.num_fast_queries += len(needed)

        created: list[_Node] = []

        def expose(parent: _Node, spec: ChildSpec) -> _Node:
            child = self._new_node(
                parent,
                parent.depth + 1,
                spec.remaining_depth,
                spec.kind,
                spec.payload,
                spec.probability,
                spec.key,
            )
            parent.children.append(child)
            created.append(child)
            if spec.kind == CHANCE:
                for outcome in spec.outcomes or ():
                    expose(child, outcome)
                child.expanded = True
                child.fast_bound = self.oracle.fast_bound(CHANCE, spec.remaining_depth)
                return child
            if spec.exact_value is not None:
                value, bound = float(spec.exact_value), 0.0
            else:
                value, bound = fast_by_spec[id(spec)]
            child.fast_bound = float(bound)
            child.fast_l, child.fast_u = self._clip(value - bound, value + bound)
            return child

        for spec in specs:
            expose(node, spec)
        node.expanded = True
        node.pending = None
        return created

    def _clip(self, low: float, high: float) -> tuple[float, float]:
        value_range = self.config.value_range
        if value_range is None:
            return low, high
        return max(low, value_range[0]), min(high, value_range[1])

    def _is_leaf(self, node: _Node) -> bool:
        return node.kind == LEAF or (node.kind != CHANCE and node.h <= 0)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> TwoFFSResult:
        self.initialize()
        outer_rounds = 0
        idle_rounds = 0
        stop_reason = "separated"
        budget = self.config.budget
        while not self.is_stopped():
            if outer_rounds >= self.config.max_outer_rounds:
                stop_reason = "max_outer_rounds"
                break
            if self.total_cost >= budget:
                stop_reason = "budget"
                break
            x, side, scale = self.root_obligation()
            if _is_inf(scale):
                stop_reason = "no_finite_root_obligation"
                break
            cost, status = self.resolve(x, side, int(scale), budget - self.total_cost)
            outer_rounds += 1
            if status == "blocked":
                stop_reason = "budget"
                break
            idle_rounds = idle_rounds + 1 if cost <= 0 else 0
            if idle_rounds > 100_000:
                stop_reason = "no_progress"
                break
        return self._result(outer_rounds, stop_reason)

    def initialize(self) -> None:
        specs = self.oracle.expand(self.root.payload, MAX, self.root.h)
        if len(specs) < 2:
            raise ValueError("2FFS needs at least two root actions")
        self._attach_children(self.root, specs)
        self._refresh_all()
        widths = [self.width(child) for child in self.root.children]
        self.rho0 = max(max(widths), 1e-12)

    def _result(self, outer_rounds: int, stop_reason: str) -> TwoFFSResult:
        stopped = self.is_stopped()
        recommendation = self.recommendation()
        if recommendation is None:
            recommendation = self.fallback_recommendation()
        children = self.root.children
        return TwoFFSResult(
            recommendation_index=None if recommendation is None else children.index(recommendation),
            recommendation_key=None if recommendation is None else recommendation.key,
            stopped=stopped,
            stop_reason="separated" if stopped else stop_reason,
            outer_rounds=outer_rounds,
            total_cost=self.total_cost,
            fast_cost=self.fast_cost,
            slow_cost=self.slow_cost,
            num_fast_queries=self.num_fast_queries,
            num_slow_queries=self.num_slow_queries,
            num_nodes=len(self.nodes),
            num_expanded_nodes=self.num_expanded,
            max_depth=self.max_depth,
            root_intervals=[(child.l, child.u) for child in children],
            root_keys=[child.key for child in children],
            root_slow_queries=[child.n_slow for child in children],
            plausible_root_actions=len(self.plausible_root_children()) if children else 0,
        )

    # ------------------------------------------------------------------
    # Interval propagation
    # ------------------------------------------------------------------

    def _refresh_all(self) -> None:
        for node in sorted(self.nodes, key=lambda item: item.depth, reverse=True):
            self._refresh(node)

    def _refresh_nodes(self, nodes: Sequence[_Node]) -> None:
        for node in sorted(nodes, key=lambda item: item.depth, reverse=True):
            self._refresh(node)

    def _refresh(self, node: _Node) -> None:
        if node is self.root:
            node.child_l, node.child_u = self._backup(node)
            node.l, node.u = node.child_l, node.child_u
            return
        low, high = node.fast_l, node.fast_u
        if node.n_slow > 0:
            low = max(low, node.local_l)
            high = min(high, node.local_u)
        if node.expanded:
            node.child_l, node.child_u = self._backup(node)
            low = max(low, node.child_l)
            high = min(high, node.child_u)
        node.l, node.u = low, high

    @staticmethod
    def _backup(node: _Node) -> tuple[float, float]:
        children = node.children
        if not children:
            return node.l, node.u
        if node.kind == MAX:
            return max(child.l for child in children), max(child.u for child in children)
        if node.kind == MIN:
            return min(child.l for child in children), min(child.u for child in children)
        if node.kind == CHANCE:
            return (
                sum(child.probability * child.l for child in children),
                sum(child.probability * child.u for child in children),
            )
        raise ValueError("cannot back up the children of a leaf")

    @staticmethod
    def _path_to_root(node: _Node) -> list[_Node]:
        path = []
        current: _Node | None = node
        while current is not None:
            path.append(current)
            current = current.parent
        return path

    def _refresh_and_latch(self, scale: int, affected: list[_Node]) -> None:
        self._refresh_nodes(affected)
        self._affected = affected
        self.latch_scale(scale)

    def latch_scale(self, scale: int) -> None:
        nodes = self.nodes if self.config.latch_all else self._affected
        for node in nodes:
            if node is self.root:
                continue
            self.ensure_scale(node, scale)
            for side in ("L", "U"):
                if self.cert(node, side, scale):
                    self.set_done(node, side, scale)

    # ------------------------------------------------------------------
    # Confidence, costs, and dyadic scales
    # ------------------------------------------------------------------

    def radius(self, n: int) -> float:
        if n <= 0:
            return _INF
        log_term = math.log((math.pi * math.pi * n * n) / (6.0 * self.node_delta))
        return self.oracle.slow_sigma * math.sqrt(2.0 * log_term / n) + self.config.slow_bias

    def m_required(self, rho: float) -> int | float:
        target = rho / 4.0
        if target <= self.config.slow_bias:
            return _INF
        cached = self._m_required_cache.get(target)
        if cached is not None:
            return cached
        n = 1
        while self.radius(n) > target:
            n *= 2
            if n > 10_000_000_000:
                return _INF
        lo, hi = n // 2 + 1, n
        while lo < hi:
            mid = (lo + hi) // 2
            if self.radius(mid) <= target:
                hi = mid
            else:
                lo = mid + 1
        self._m_required_cache[target] = lo
        return lo

    def gamma(self, node: _Node, rho: float) -> float:
        if node.fast_bound <= rho / 4.0:
            return 0.0
        return self.oracle.slow_cost_estimate() * self.m_required(rho)

    def alpha(self, node: _Node) -> float:
        return (node.h + self.config.alpha_offset) ** self.config.alpha_power

    def race_budget(self, node: _Node, scale: int) -> float:
        gamma = self.gamma(node, self.rho(scale))
        return 0.0 if gamma == 0.0 else self.alpha(node) * gamma

    def rho(self, scale: int) -> float:
        return self.rho0 / (2.0**scale)

    @staticmethod
    def width(node: _Node) -> float:
        return max(0.0, node.u - node.l)

    def width_scale(self, node: _Node) -> int | float:
        w = self.width(node)
        if w <= 1e-14:
            return _INF
        if w > self.rho0:
            return 0
        # rho_k / 2 < w <= rho_k  <=>  k = floor(log2(rho0 / w)).
        scale = int(math.floor(math.log2(self.rho0 / w)))
        scale = max(0, scale)
        while scale > 0 and not (w <= self.rho(scale)):
            scale -= 1
        while scale < self.config.max_scale and not (self.rho(scale) / 2.0 < w):
            scale += 1
        return min(scale, self.config.max_scale)

    def active_scale(self, node: _Node, side: Side) -> int | float:
        underline = self.width_scale(node)
        if _is_inf(underline):
            return _INF
        for scale in range(int(underline), self.config.max_scale + 1):
            if not self.comp(node, side, scale):
                return scale
        return _INF

    def capped_scale(self, node: _Node, side: Side, parent_scale: int) -> int | float:
        for scale in range(parent_scale + 1):
            if not self.comp(node, side, scale):
                return scale
        return _INF

    # ------------------------------------------------------------------
    # Certificates and active witnesses
    # ------------------------------------------------------------------

    def ensure_scale(self, node: _Node, scale: int) -> _ScaleState:
        key = (node.index, scale)
        state = self.scale_state.get(key)
        if state is None:
            state = _ScaleState()
            self.scale_state[key] = state
        return state

    def set_done(self, node: _Node, side: Side, scale: int) -> None:
        state = self.ensure_scale(node, scale)
        if side == "L":
            state.done_l = True
        else:
            state.done_u = True

    def is_done(self, node: _Node, side: Side, scale: int) -> bool:
        state = self.scale_state.get((node.index, scale))
        if state is None:
            return False
        return state.done_l if side == "L" else state.done_u

    def comp(self, node: _Node, side: Side, scale: int) -> bool:
        return self.is_done(node, side, scale) or self.cert(node, side, scale)

    def cert(self, node: _Node, side: Side, scale: int) -> bool:
        if node is self.root:
            return False
        tolerance = self.rho(scale) / 2.0
        if self.width(node) <= tolerance:
            return True
        if node.n_slow > 0 and (node.local_u - node.local_l) <= tolerance:
            return True
        if not node.expanded or not node.children:
            return False
        children = node.children
        kind = node.kind
        if kind == MAX and side == "L":
            lambda_l = max(child.l for child in children)
            return all(
                child.u <= lambda_l + tolerance or self.comp(child, "L", scale)
                for child in children
            )
        if kind == MIN and side == "U":
            lambda_u = min(child.u for child in children)
            return all(
                child.l >= lambda_u - tolerance or self.comp(child, "U", scale)
                for child in children
            )
        if kind == CHANCE:
            open_mass = 0.0
            open_width = 0.0
            for child in children:
                if not self.comp(child, side, scale):
                    open_mass += child.probability
                    open_width += child.probability * self.width(child)
            return open_width <= open_mass * tolerance
        active = self.active_children(node, side, scale)
        return bool(active) and all(self.comp(child, side, scale) for child in active)

    @staticmethod
    def is_comparison_case(node: _Node, side: Side) -> bool:
        return (node.kind == MAX and side == "L") or (node.kind == MIN and side == "U")

    def active_children(self, node: _Node, side: Side, scale: int) -> list[_Node]:
        children = node.children
        if not children:
            return []
        tolerance = self.rho(scale) / 2.0
        kind = node.kind
        if kind == CHANCE:
            return [child for child in children if not self.comp(child, side, scale)]
        if kind == MIN and side == "L":
            best = min(child.l for child in children)
            return [child for child in children if abs(child.l - best) <= 1e-14]
        if kind == MAX and side == "U":
            best = max(child.u for child in children)
            return [child for child in children if abs(child.u - best) <= 1e-14]
        if kind == MAX and side == "L":
            lambda_l = max(child.l for child in children)
            return [
                child
                for child in children
                if child.u > lambda_l + tolerance and not self.comp(child, "L", scale)
            ]
        if kind == MIN and side == "U":
            lambda_u = min(child.u for child in children)
            return [
                child
                for child in children
                if child.l < lambda_u - tolerance and not self.comp(child, "U", scale)
            ]
        raise ValueError(f"invalid node/side combination: kind={kind}, side={side}")

    # ------------------------------------------------------------------
    # Root and recursive obligation selection
    # ------------------------------------------------------------------

    def is_stopped(self) -> bool:
        return self.recommendation() is not None

    def recommendation(self) -> _Node | None:
        children = self.root.children
        epsilon = self.config.epsilon
        if len(children) < 2:
            return children[0] if children else None
        # max U over "everyone but c" from the top two upper bounds.
        first_u, second_u, first_index = -_INF, -_INF, -1
        for index, child in enumerate(children):
            if child.u > first_u:
                second_u, first_u, first_index = first_u, child.u, index
            elif child.u > second_u:
                second_u = child.u
        for index, child in enumerate(children):
            other_upper = second_u if index == first_index else first_u
            if child.l >= other_upper - epsilon:
                return child
        return None

    def plausible_root_children(self) -> list[_Node]:
        """Root actions not eliminated: ``U_a >= max_b L_b - epsilon``."""
        children = self.root.children
        best_lower = max(child.l for child in children)
        return [child for child in children if child.u >= best_lower - self.config.epsilon]

    def fallback_recommendation(self) -> _Node | None:
        children = self.root.children
        if not children:
            return None
        if self.config.fallback == "midpoint":
            return max(children, key=lambda child: (_midpoint(child), child.l))
        if self.config.fallback == "prior":
            plausible = self.plausible_root_children()
            scores = self.oracle.prior_scores([child.key for child in plausible])  # type: ignore[attr-defined]
            return plausible[max(range(len(plausible)), key=lambda index: scores[index])]
        return max(children, key=lambda child: (child.l, child.u))

    def contender_scale(self, node: _Node) -> tuple[Side, int | float]:
        k_l = self.active_scale(node, "L")
        k_u = self.active_scale(node, "U")
        if not _is_inf(k_l) and (_is_inf(k_u) or self.rho(int(k_l)) >= self.rho(int(k_u))):
            return "L", k_l
        return "U", k_u

    def root_obligation(self) -> tuple[_Node, Side, int | float]:
        children = self.root.children
        leader = max(children, key=lambda child: child.l)
        challenger = max(
            (child for child in children if child is not leader), key=lambda child: child.u
        )
        k_leader = self.active_scale(leader, "L")
        side_chal, k_chal = self.contender_scale(challenger)
        if not _is_inf(k_leader) and (
            _is_inf(k_chal) or self.rho(int(k_leader)) >= self.rho(int(k_chal))
        ):
            return leader, "L", k_leader
        return challenger, side_chal, k_chal

    def child_obligation(
        self, node: _Node, side: Side, scale: int
    ) -> tuple[_Node, Side, int | float] | None:
        if self.is_comparison_case(node, side):
            active = self.active_children(node, side, scale)
            if not active:
                return None
            if node.kind == MAX:
                blocker = max(active, key=lambda child: child.u)
            else:
                blocker = min(active, key=lambda child: child.l)
            opposite: Side = "U" if side == "L" else "L"
            k_opp = self.capped_scale(blocker, opposite, scale)
            if not _is_inf(k_opp):
                return blocker, opposite, k_opp
            return blocker, side, self.capped_scale(blocker, side, scale)

        if node.kind == CHANCE:
            candidates = self.active_children(node, side, scale)
            if not candidates:
                return None
            child = max(candidates, key=lambda item: item.probability * self.width(item))
            return child, side, self.capped_scale(child, side, scale)

        candidates = [
            child
            for child in self.active_children(node, side, scale)
            if not self.comp(child, side, scale)
        ]
        if not candidates:
            return None
        child = max(candidates, key=self.width)
        return child, side, self.capped_scale(child, side, scale)

    # ------------------------------------------------------------------
    # Atomic actions and the budgeted recursive resolver
    # ------------------------------------------------------------------

    def local_step(self, node: _Node, scale: int, cap: float) -> tuple[float, Status]:
        count = self.config.slow_batch
        if cap < count * self.oracle.slow_cost_estimate():
            return 0.0, "blocked"
        samples, cost = self.oracle.slow_sample(node.payload, node.kind, count)
        node.n_slow += len(samples)
        node.slow_sum += float(sum(samples))
        empirical = node.slow_sum / node.n_slow
        rad = self.radius(node.n_slow)
        node.local_l, node.local_u = self._clip(empirical - rad, empirical + rad)
        self.ensure_scale(node, scale).m_loc += cost
        self.total_cost += cost
        self.slow_cost += cost
        self.num_slow_queries += len(samples)
        self._refresh_and_latch(scale, self._path_to_root(node))
        return cost, "progress"

    def expand_node(self, node: _Node, scale: int) -> tuple[float, Status]:
        if node.expanded:
            return 0.0, "cert"
        before = self.total_cost
        created = self._attach_children(node, self._children_specs(node))
        self.num_expanded += 1
        cost = self.total_cost - before
        for child in created:
            self.ensure_scale(child, scale)
        self.ensure_scale(node, scale).m_exp += cost
        affected = created + self._path_to_root(node)
        self._refresh_and_latch(scale, affected)
        return cost, "progress"

    def resolve(self, node: _Node, side: Side, scale: int, cap: float) -> tuple[float, Status]:
        self.ensure_scale(node, scale)
        if self.comp(node, side, scale):
            self.set_done(node, side, scale)
            return 0.0, "cert"

        if self._is_leaf(node):
            return self.local_step(node, scale, cap)

        scale_state = self.ensure_scale(node, scale)
        remaining_race_budget = self.race_budget(node, scale) - scale_state.m_exp
        if remaining_race_budget <= 0:
            return self.local_step(node, scale, cap)

        recursive_cap = min(cap, remaining_race_budget)

        if not node.expanded:
            child_cost = self._expansion_cost(node)
            too_many_nodes = (
                self.config.max_nodes is not None
                and len(self.nodes) + self._expansion_nodes(node) > self.config.max_nodes
            )
            if child_cost > remaining_race_budget or too_many_nodes:
                return self.local_step(node, scale, cap)
            if child_cost > cap:
                return 0.0, "blocked"
            return self.expand_node(node, scale)

        obligation = self.child_obligation(node, side, scale)
        if obligation is None:
            self.latch_scale(scale)
            if self.comp(node, side, scale):
                self.set_done(node, side, scale)
                return 0.0, "cert"
            return self.local_step(node, scale, cap)

        child, child_side, child_scale = obligation
        if _is_inf(child_scale):
            self.latch_scale(scale)
            return self.local_step(node, scale, cap)

        cost, status = self.resolve(child, child_side, int(child_scale), recursive_cap)
        if status == "blocked":
            if cap < remaining_race_budget:
                return 0.0, "blocked"
            return self.local_step(node, scale, cap)

        scale_state.m_exp += cost
        self.latch_scale(scale)
        return cost, status


def _is_inf(scale: float) -> bool:
    return isinstance(scale, float) and math.isinf(scale)


def _midpoint(node: _Node) -> float:
    if math.isinf(node.l) or math.isinf(node.u):
        return node.l if math.isinf(node.u) else node.u
    return 0.5 * (node.l + node.u)


__all__ = [
    "CHANCE",
    "LEAF",
    "MAX",
    "MIN",
    "ChildSpec",
    "TwoFFS",
    "TwoFFSConfig",
    "TwoFFSResult",
    "TwoFidelityOracle",
]
