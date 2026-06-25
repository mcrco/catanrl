"""Unified experiment store: self-describing training runs.

A single training run lives under one directory (by default
``experiments/<name>/``) and is fully self-describing:

    experiments/<name>/
        metadata.json        # how to rebuild every network in this run
        checkpoints.json      # which checkpoint is "best" / "latest" / step N
        checkpoints/          # the actual *.pt state_dicts
        eval/                 # (optional) eval result dumps

The point of ``metadata.json`` is that you never again have to remember which
``--backbone-type``/``--policy-hidden-dims``/``--map-type`` flags a checkpoint
was trained with. The metadata records the *exact* backbone config and head
type, so a network can be rebuilt and its weights loaded with a single call:

    exp = load_experiment("ppo-sarl-f-winreward-xdim-flat")
    policy = exp.build_policy(which="best", device="cuda")
    critic = exp.build_critic(which="best", device="cuda")

Each run is self-describing on disk: inspect ``experiments/<name>/metadata.json``
and ``experiments/<name>/checkpoints.json`` directly.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

import torch

from .models.backbones import (
    BackboneConfig,
    CrossDimensionalBackboneConfig,
    MLPBackboneConfig,
)
from .models.models import (
    build_flat_policy_network,
    build_flat_policy_value_network,
    build_hierarchical_policy_network,
    build_hierarchical_policy_value_network,
    build_value_network,
)
from .models.wrappers import policy_value_to_policy_only
from .utils.catanatron_action_space import get_action_space_size

SCHEMA_VERSION = 1

METADATA_FILENAME = "metadata.json"
CHECKPOINTS_FILENAME = "checkpoints.json"
CHECKPOINTS_DIRNAME = "checkpoints"
# Sidecar holding the full mutable training state (optimizer/scheduler/step
# counters/model weights/W&B run id) needed to *continue* a run in place.
# Deliberately NOT a ``*.pt`` under ``checkpoints/`` so checkpoint discovery
# never mistakes it for a network weight file.
TRAINING_STATE_FILENAME = "training_state.pt"

# Network "kind": what heads the saved state_dict actually contains.
KIND_POLICY = "policy"
KIND_POLICY_VALUE = "policy_value"
KIND_VALUE = "value"


# --------------------------------------------------------------------------- #
# Location helpers
# --------------------------------------------------------------------------- #
def experiments_root() -> str:
    """Root directory that holds all experiment folders.

    Overridable via the ``CATANRL_EXPERIMENTS_DIR`` environment variable so the
    same code works on machines with different storage layouts.
    """
    env = os.environ.get("CATANRL_EXPERIMENTS_DIR")
    if env:
        return os.path.abspath(env)
    return os.path.abspath("experiments")


def experiment_dir(name_or_path: str) -> str:
    """Resolve an experiment name or path to its directory.

    Accepts either a bare experiment name (looked up under
    :func:`experiments_root`) or a direct path to an experiment directory.
    """
    candidate = os.path.abspath(name_or_path)
    if os.path.isfile(os.path.join(candidate, METADATA_FILENAME)):
        return candidate
    return os.path.join(experiments_root(), name_or_path)


# --------------------------------------------------------------------------- #
# Backbone (de)serialization
# --------------------------------------------------------------------------- #
def backbone_config_to_dict(config: BackboneConfig) -> Dict[str, Any]:
    args = asdict(config.args)
    # JSON has no tuples; normalize the kernel size to a list on write.
    if "cnn_kernel_size" in args and args["cnn_kernel_size"] is not None:
        args["cnn_kernel_size"] = list(args["cnn_kernel_size"])
    return {"architecture": config.architecture, "args": args}


def backbone_config_from_dict(data: Dict[str, Any]) -> BackboneConfig:
    architecture = data["architecture"]
    args = dict(data["args"])
    if architecture == "mlp":
        return BackboneConfig(architecture=architecture, args=MLPBackboneConfig(**args))
    if architecture in ("cross_dimensional", "residual_cross_dimensional"):
        if "cnn_kernel_size" in args and args["cnn_kernel_size"] is not None:
            args["cnn_kernel_size"] = tuple(args["cnn_kernel_size"])
        return BackboneConfig(
            architecture=architecture, args=CrossDimensionalBackboneConfig(**args)
        )
    raise ValueError(f"Unknown backbone architecture '{architecture}'")


# --------------------------------------------------------------------------- #
# Schema dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class GameConfig:
    """Environment/game settings that affect feature and action dimensions."""

    num_players: int
    map_type: str
    vps_to_win: Optional[int] = None
    discard_limit: Optional[int] = None


@dataclass
class NetworkSpec:
    """Everything needed to rebuild one network and load its weights."""

    kind: str  # KIND_POLICY | KIND_POLICY_VALUE | KIND_VALUE
    backbone: BackboneConfig
    model_type: Optional[str] = None  # "flat" | "hierarchical" (policy nets only)
    observation_level: Optional[str] = None  # feature level used at train time
    num_actions: Optional[int] = None  # flat policy action-space size

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "kind": self.kind,
            "backbone": backbone_config_to_dict(self.backbone),
        }
        if self.model_type is not None:
            out["model_type"] = self.model_type
        if self.observation_level is not None:
            out["observation_level"] = self.observation_level
        if self.num_actions is not None:
            out["num_actions"] = self.num_actions
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NetworkSpec":
        return cls(
            kind=data["kind"],
            backbone=backbone_config_from_dict(data["backbone"]),
            model_type=data.get("model_type"),
            observation_level=data.get("observation_level"),
            num_actions=data.get("num_actions"),
        )


@dataclass
class ExperimentMetadata:
    name: str
    algorithm: str
    game: GameConfig
    networks: Dict[str, NetworkSpec]
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    git_sha: Optional[str] = None
    train_config: Dict[str, Any] = field(default_factory=dict)
    wandb: Dict[str, Any] = field(default_factory=dict)
    notes: Optional[str] = None
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "algorithm": self.algorithm,
            "created_at": self.created_at,
            "git_sha": self.git_sha,
            "game": asdict(self.game),
            "networks": {name: spec.to_dict() for name, spec in self.networks.items()},
            "train_config": self.train_config,
            "wandb": self.wandb,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentMetadata":
        return cls(
            name=data["name"],
            algorithm=data.get("algorithm", "unknown"),
            game=GameConfig(**data["game"]),
            networks={
                name: NetworkSpec.from_dict(spec)
                for name, spec in data.get("networks", {}).items()
            },
            created_at=data.get("created_at", ""),
            git_sha=data.get("git_sha"),
            train_config=data.get("train_config", {}),
            wandb=data.get("wandb", {}),
            notes=data.get("notes"),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )


# --------------------------------------------------------------------------- #
# Checkpoint registry
# --------------------------------------------------------------------------- #
@dataclass
class CheckpointRegistry:
    """Maps selectors ("best"/"latest"/step) to checkpoint files per role.

    ``selectors`` looks like::

        {"best":   {"policy": "checkpoints/policy_best.pt", "critic": ...},
         "latest": {"policy": "checkpoints/policy_update_900.pt", ...}}
    """

    selectors: Dict[str, Dict[str, str]] = field(default_factory=dict)
    checkpoints: List[Dict[str, Any]] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "selectors": self.selectors,
            "checkpoints": self.checkpoints,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointRegistry":
        return cls(
            selectors=data.get("selectors", {}),
            checkpoints=data.get("checkpoints", []),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )

    def resolve(self, which: Union[str, int], role: str) -> Optional[str]:
        """Return the checkpoint path for ``role`` under selector ``which``.

        ``which`` may be a named selector ("best"/"latest"), an integer step,
        or an explicit filename.
        """
        key = str(which)
        if key in self.selectors and role in self.selectors[key]:
            return self.selectors[key][role]
        # Allow selecting an explicit training step.
        for entry in self.checkpoints:
            if str(entry.get("step")) != key:
                continue
            files = entry.get("files")
            if isinstance(files, dict) and role in files:
                return files[role]
            if entry.get("role") == role and entry.get("file"):
                return entry["file"]
        # Fall back to treating ``which`` as a literal relative path.
        if any(ch in key for ch in ("/", ".")):
            return key
        return None


# --------------------------------------------------------------------------- #
# Building networks from specs
# --------------------------------------------------------------------------- #
def build_network(spec: NetworkSpec, game: GameConfig) -> torch.nn.Module:
    """Reconstruct a network module (without weights) from its spec."""
    num_actions = spec.num_actions
    if num_actions is None and spec.kind in (KIND_POLICY, KIND_POLICY_VALUE):
        num_actions = get_action_space_size(game.num_players, game.map_type)

    if spec.kind == KIND_VALUE:
        return build_value_network(spec.backbone)

    if spec.model_type == "hierarchical":
        if spec.kind == KIND_POLICY_VALUE:
            return build_hierarchical_policy_value_network(
                spec.backbone, game.num_players, game.map_type
            )
        return build_hierarchical_policy_network(
            spec.backbone, game.num_players, game.map_type
        )

    # Default to flat heads.
    if spec.kind == KIND_POLICY_VALUE:
        return build_flat_policy_value_network(spec.backbone, num_actions)
    return build_flat_policy_network(spec.backbone, num_actions)


# --------------------------------------------------------------------------- #
# Experiment
# --------------------------------------------------------------------------- #
class Experiment:
    """A loaded experiment: metadata + checkpoint registry + builders."""

    def __init__(self, path: str, metadata: ExperimentMetadata, registry: CheckpointRegistry):
        self.path = path
        self.metadata = metadata
        self.registry = registry

    # -- loading -----------------------------------------------------------
    def resolve_checkpoint(self, which: Union[str, int], role: str) -> str:
        rel = self.registry.resolve(which, role)
        if rel is None:
            available = sorted(self.registry.selectors.keys())
            raise FileNotFoundError(
                f"No checkpoint for role '{role}' / selector '{which}' in {self.path}. "
                f"Available selectors: {available}"
            )
        abs_path = rel if os.path.isabs(rel) else os.path.join(self.path, rel)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Checkpoint file does not exist: {abs_path}")
        return abs_path

    def load_network(
        self,
        role: str,
        which: Union[str, int] = "best",
        device: Optional[Union[str, torch.device]] = None,
        as_policy_only: bool = True,
        eval_mode: bool = True,
    ) -> torch.nn.Module:
        if role not in self.metadata.networks:
            raise KeyError(
                f"Experiment '{self.metadata.name}' has no network '{role}'. "
                f"Available: {sorted(self.metadata.networks)}"
            )
        device = torch.device(device) if device is not None else torch.device("cpu")
        spec = self.metadata.networks[role]
        model = build_network(spec, self.metadata.game)
        ckpt_path = self.resolve_checkpoint(which, role)
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)

        if as_policy_only and spec.kind == KIND_POLICY_VALUE:
            model = policy_value_to_policy_only(model).to(device)

        if eval_mode:
            model.eval()
        return model

    def build_policy(
        self,
        which: Union[str, int] = "best",
        device: Optional[Union[str, torch.device]] = None,
        as_policy_only: bool = True,
        eval_mode: bool = True,
    ) -> torch.nn.Module:
        return self.load_network(
            "policy", which=which, device=device,
            as_policy_only=as_policy_only, eval_mode=eval_mode,
        )

    def build_critic(
        self,
        which: Union[str, int] = "best",
        device: Optional[Union[str, torch.device]] = None,
        eval_mode: bool = True,
    ) -> torch.nn.Module:
        role = "critic" if "critic" in self.metadata.networks else "policy"
        return self.load_network(
            role, which=which, device=device,
            as_policy_only=False, eval_mode=eval_mode,
        )

    # -- convenience accessors --------------------------------------------
    @property
    def policy_spec(self) -> NetworkSpec:
        return self.metadata.networks["policy"]

    @property
    def model_type(self) -> Optional[str]:
        return self.policy_spec.model_type

    @property
    def map_type(self) -> str:
        return self.metadata.game.map_type

    @property
    def num_players(self) -> int:
        return self.metadata.game.num_players

    def __repr__(self) -> str:
        return (
            f"Experiment(name={self.metadata.name!r}, algorithm={self.metadata.algorithm!r}, "
            f"networks={sorted(self.metadata.networks)}, "
            f"selectors={sorted(self.registry.selectors)})"
        )


_BACKBONE_DISPLAY = {
    "mlp": "mlp",
    "cross_dimensional": "xdim",
    "residual_cross_dimensional": "xdim_res",
}


def backbone_display_type(backbone: BackboneConfig) -> str:
    """Map a backbone architecture to the CLI ``--backbone-type`` spelling."""
    return _BACKBONE_DISPLAY.get(backbone.architecture, backbone.architecture)


def backbone_hidden_dims(backbone: BackboneConfig) -> List[int]:
    """Best-effort hidden-layer sizes for display/logging."""
    args = backbone.args
    return list(getattr(args, "hidden_dims", None) or getattr(args, "numeric_hidden_dims", []))


def load_experiment(name_or_path: str) -> Experiment:
    """Load an experiment by name (under ``experiments/``) or by directory path."""
    path = experiment_dir(name_or_path)
    meta_path = os.path.join(path, METADATA_FILENAME)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"No {METADATA_FILENAME} found at {path}. "
            f"Is '{name_or_path}' a valid experiment? "
            f"(Experiments are written by the training entrypoints under "
            f"src/catanrl/experiments/.)"
        )
    with open(meta_path, "r") as f:
        metadata = ExperimentMetadata.from_dict(json.load(f))

    registry_path = os.path.join(path, CHECKPOINTS_FILENAME)
    if os.path.exists(registry_path):
        with open(registry_path, "r") as f:
            registry = CheckpointRegistry.from_dict(json.load(f))
    else:
        registry = CheckpointRegistry()
    return Experiment(path, metadata, registry)


# --------------------------------------------------------------------------- #
# Training warm-start helpers
# --------------------------------------------------------------------------- #
@dataclass
class TrainingCheckpoints:
    """Resolved checkpoint paths for warm-starting a training run."""

    experiment_name: str
    which: Union[str, int]
    policy: str
    critic: Optional[str] = None


@dataclass
class TrainingWarmStart:
    """A source experiment plus resolved checkpoint paths for warm-start."""

    experiment: Experiment
    checkpoints: TrainingCheckpoints


def _set_arg(args: argparse.Namespace, name: str, value: Any) -> None:
    if value is not None and hasattr(args, name):
        setattr(args, name, value)


def _hidden_dims_csv(backbone: BackboneConfig) -> str:
    return ",".join(str(dim) for dim in backbone_hidden_dims(backbone))


def _apply_xdim_backbone_args(args: argparse.Namespace, backbone: BackboneConfig) -> None:
    if backbone.architecture not in ("cross_dimensional", "residual_cross_dimensional"):
        return
    bb_args = backbone.args
    if not isinstance(bb_args, CrossDimensionalBackboneConfig):
        return
    channels = ",".join(str(ch) for ch in bb_args.cnn_channels)
    kernel = f"{bb_args.cnn_kernel_size[0]},{bb_args.cnn_kernel_size[1]}"
    _set_arg(args, "xdim_cnn_channels", channels)
    _set_arg(args, "xdim_cnn_kernel_size", kernel)
    _set_arg(args, "xdim_policy_fusion_hidden_dim", bb_args.fusion_hidden_dim)


def apply_experiment_architecture_to_args(args: argparse.Namespace, exp: Experiment) -> None:
    """Override architecture-related CLI args from experiment metadata.

    Mirrors the eval scripts' ``--experiment`` path: when warm-starting training,
    backbone/head settings are taken from ``metadata.json`` (with ``train_config``
    as a fallback for a few run-specific knobs).
    """
    meta = exp.metadata
    tc = meta.train_config
    policy = exp.policy_spec
    critic = meta.networks.get("critic")

    _set_arg(args, "model_type", policy.model_type or tc.get("model_type"))
    _set_arg(args, "map_type", meta.game.map_type)
    _set_arg(args, "num_players", meta.game.num_players)
    _set_arg(args, "vps_to_win", meta.game.vps_to_win)
    _set_arg(args, "discard_limit", meta.game.discard_limit)

    _set_arg(args, "backbone_type", backbone_display_type(policy.backbone))
    policy_dims = _hidden_dims_csv(policy.backbone)
    _set_arg(args, "policy_hidden_dims", policy_dims)
    _apply_xdim_backbone_args(args, policy.backbone)

    if policy.observation_level is not None:
        _set_arg(args, "policy_mode", policy.observation_level)
        _set_arg(args, "actor_observation_level", policy.observation_level)
    elif "actor_observation_level" in tc:
        _set_arg(args, "actor_observation_level", tc["actor_observation_level"])
    if "policy_mode" in tc:
        _set_arg(args, "policy_mode", tc["policy_mode"])

    if critic is not None:
        critic_dims = _hidden_dims_csv(critic.backbone)
        _set_arg(args, "critic_hidden_dims", critic_dims)
        if critic.observation_level is not None:
            _set_arg(args, "critic_mode", critic.observation_level)
            _set_arg(args, "critic_observation_level", critic.observation_level)
        if isinstance(critic.backbone.args, CrossDimensionalBackboneConfig):
            _set_arg(
                args,
                "xdim_critic_fusion_hidden_dim",
                critic.backbone.args.fusion_hidden_dim,
            )
    elif isinstance(tc.get("critic_hidden_dims"), list):
        _set_arg(
            args,
            "critic_hidden_dims",
            ",".join(str(dim) for dim in tc["critic_hidden_dims"]),
        )

    if isinstance(tc.get("policy_hidden_dims"), list):
        _set_arg(
            args,
            "policy_hidden_dims",
            ",".join(str(dim) for dim in tc["policy_hidden_dims"]),
        )
    elif isinstance(tc.get("hidden_dims"), list):
        _set_arg(
            args,
            "policy_hidden_dims",
            ",".join(str(dim) for dim in tc["hidden_dims"]),
        )

    if "critic_observation_level" in tc:
        _set_arg(args, "critic_observation_level", tc["critic_observation_level"])
    if "critic_mode" in tc and tc["critic_mode"] in ("private", "public", "full"):
        _set_arg(args, "critic_mode", tc["critic_mode"])
    if "network_mode" in tc:
        _set_arg(args, "network_mode", tc["network_mode"])
    elif "critic_mode" in tc and tc["critic_mode"] in ("shared", "privileged"):
        _set_arg(
            args,
            "network_mode",
            "shared" if tc["critic_mode"] == "shared" else "separate",
        )
    elif policy.kind == KIND_POLICY_VALUE:
        _set_arg(args, "network_mode", "shared")
    elif critic is None and policy.kind != KIND_POLICY_VALUE:
        _set_arg(args, "network_mode", "separate")

    if policy.kind == KIND_POLICY_VALUE and critic is None:
        actor_level = policy.observation_level or tc.get("actor_observation_level")
        if actor_level is not None:
            _set_arg(args, "critic_mode", actor_level)
            _set_arg(args, "critic_observation_level", actor_level)

    if "xdim_cnn_channels" in tc and hasattr(args, "xdim_cnn_channels"):
        channels = tc["xdim_cnn_channels"]
        if isinstance(channels, list):
            _set_arg(args, "xdim_cnn_channels", ",".join(str(ch) for ch in channels))
    if "xdim_cnn_kernel_size" in tc and hasattr(args, "xdim_cnn_kernel_size"):
        kernel = tc["xdim_cnn_kernel_size"]
        if isinstance(kernel, list):
            _set_arg(args, "xdim_cnn_kernel_size", ",".join(str(k) for k in kernel))
    for key in (
        "xdim_policy_fusion_hidden_dim",
        "xdim_critic_fusion_hidden_dim",
    ):
        if key in tc:
            _set_arg(args, key, tc[key])

    print(
        f"Architecture inherited from experiment '{meta.name}': "
        f"backbone={getattr(args, 'backbone_type', backbone_display_type(policy.backbone))}, "
        f"model={getattr(args, 'model_type', policy.model_type)}, "
        f"map={meta.game.map_type}, players={meta.game.num_players}"
    )


def resolve_training_checkpoints_from_experiment(
    exp: Experiment,
    which: Union[str, int] = "best",
    *,
    require_critic: bool = False,
) -> TrainingCheckpoints:
    """Resolve on-disk checkpoint paths from a loaded experiment."""
    policy_spec = exp.policy_spec
    uses_joint_network = policy_spec.kind == KIND_POLICY_VALUE
    if uses_joint_network:
        require_critic = False

    policy = exp.resolve_checkpoint(which, "policy")
    critic: Optional[str] = None
    if "critic" in exp.metadata.networks:
        try:
            critic = exp.resolve_checkpoint(which, "critic")
        except FileNotFoundError:
            if require_critic:
                raise
    if require_critic and critic is None:
        raise FileNotFoundError(
            f"Experiment '{exp.metadata.name}' has no critic checkpoint for "
            f"selector '{which}', but one is required."
        )
    return TrainingCheckpoints(
        experiment_name=exp.metadata.name,
        which=which,
        policy=policy,
        critic=critic,
    )


def resolve_training_checkpoints(
    name_or_path: str,
    which: Union[str, int] = "best",
    *,
    require_critic: bool = False,
) -> TrainingCheckpoints:
    """Resolve on-disk checkpoint paths from a saved experiment."""
    exp = load_experiment(name_or_path)
    return resolve_training_checkpoints_from_experiment(
        exp, which, require_critic=require_critic
    )


def add_load_from_experiment_arguments(parser: argparse.ArgumentParser) -> None:
    """Register ``--load-from-experiment`` and ``--load-from-which`` on a parser."""
    parser.add_argument(
        "--load-from-experiment",
        type=str,
        default=None,
        dest="load_from_experiment",
        help=(
            "Warm-start weights from a saved experiment (name under experiments/ or path). "
            "When --config is omitted, architecture is read from the experiment metadata. "
            "When both are set, the preset must match the source experiment."
        ),
    )
    parser.add_argument(
        "--load-from-which",
        type=str,
        default="best",
        dest="load_from_which",
        help=(
            "Checkpoint selector for --load-from-experiment: "
            "'best', 'latest', or a training step."
        ),
    )


def add_resume_argument(parser: argparse.ArgumentParser) -> None:
    """Register ``--resume`` which turns ``--load-from-experiment`` into a true continuation.

    Without ``--resume`` the existing behaviour is unchanged: ``--load-from-experiment``
    warm-starts a brand-new run (fresh optimizer/scheduler/step, new W&B run).
    With ``--resume`` the run continues *in place*: same experiment directory, same
    W&B run id, restored optimizer/scheduler/global-step, and the training budget
    (e.g. ``--total-timesteps`` / ``--iterations``) is treated as *additional* work.
    """
    parser.add_argument(
        "--resume",
        action="store_true",
        dest="resume",
        help=(
            "Continue the --load-from-experiment run in place (same experiment dir, "
            "same W&B run, restored optimizer/scheduler/step). The training budget is "
            "interpreted as ADDITIONAL work on top of what was already done."
        ),
    )


LINEAGE_TAG_PREFIX = "lineage:"
WARMSTART_TAG_PREFIX = "warmstart:"


def _parent_lineage_root(exp: "Experiment") -> str:
    """Return the ``lineage:<root>`` tag a child should inherit from a parent.

    Reuses the parent's own ``lineage:`` tag (so a whole chain shares one root);
    falls back to the parent's name as the root when it has none.
    """
    for tag in exp.metadata.wandb.get("tags", []) or []:
        if isinstance(tag, str) and tag.startswith(LINEAGE_TAG_PREFIX):
            return tag
    return f"{LINEAGE_TAG_PREFIX}{exp.metadata.name}"


def wandb_grouping_kwargs(
    args: argparse.Namespace,
    *,
    group_default: str,
    warm_start: Optional["TrainingWarmStart"] = None,
    resume: Optional["ResumeContext"] = None,
) -> Dict[str, Any]:
    """Build ``group``/``tags`` kwargs for ``wandb.init``.

    ``group`` defaults to the algorithm family so all runs of one method
    aggregate together.

    Lineage tags are added automatically so a whole chain is filterable at once:
      * ``warmstart:<immediate-parent>`` — the direct link to the source run.
      * ``lineage:<root>`` — the chain root, propagated from the parent so
        dagger -> ppo -> ppo-extended all share one ``lineage:`` tag.
    In-place continuations (``--resume``) add a ``resume`` tag instead of a
    self-referential warm-start link.
    """
    group = getattr(args, "wandb_group", None) or group_default
    tags = list(getattr(args, "wandb_tags", None) or [])

    def _add(tag: str) -> None:
        if tag not in tags:
            tags.append(tag)

    is_resume = resume is not None and getattr(resume, "active", False)
    if is_resume:
        _add("resume")
    if warm_start is not None and not is_resume:
        parent = warm_start.experiment
        _add(f"{WARMSTART_TAG_PREFIX}{parent.metadata.name}")
        _add(_parent_lineage_root(parent))

    out: Dict[str, Any] = {"group": group}
    if tags:
        out["tags"] = tags
    return out


def training_state_file(name_or_path: str) -> str:
    """Absolute path of the training-state sidecar for an experiment."""
    return os.path.join(experiment_dir(name_or_path), TRAINING_STATE_FILENAME)


def save_training_state(
    name_or_path: str,
    state: Dict[str, Any],
    *,
    exp_dir: Optional[str] = None,
) -> str:
    """Persist mutable training state next to an experiment's metadata.

    ``exp_dir`` may be passed directly (e.g. when the experiment folder does not
    yet contain ``metadata.json``); otherwise it is resolved from ``name_or_path``.
    """
    path = exp_dir or experiment_dir(name_or_path)
    os.makedirs(path, exist_ok=True)
    out = os.path.join(path, TRAINING_STATE_FILENAME)
    payload: Dict[str, Any] = dict(state)
    payload.setdefault("schema_version", SCHEMA_VERSION)
    payload.setdefault("saved_at", datetime.now(timezone.utc).isoformat())
    torch.save(payload, out)
    return out


def load_training_state(
    name_or_path: str,
    *,
    map_location: Union[str, torch.device] = "cpu",
) -> Optional[Dict[str, Any]]:
    """Load the training-state sidecar for an experiment, or ``None`` if absent."""
    path = os.path.join(experiment_dir(name_or_path), TRAINING_STATE_FILENAME)
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location=map_location, weights_only=False)


@dataclass
class ResumeContext:
    """Resolved information for continuing a run in place (``--resume``)."""

    active: bool
    experiment_name: Optional[str] = None
    exp_dir: Optional[str] = None
    training_state_path: Optional[str] = None
    state: Optional[Dict[str, Any]] = None
    wandb_run_id: Optional[str] = None
    wandb_run_name: Optional[str] = None


def prepare_resume(
    args: argparse.Namespace,
    warm_start: Optional[TrainingWarmStart],
) -> ResumeContext:
    """Resolve resume context from ``--resume`` + ``--load-from-experiment``.

    Returns an inactive context when ``--resume`` was not passed. Raises when
    ``--resume`` is set without a source experiment to continue.
    """
    if not getattr(args, "resume", False):
        return ResumeContext(active=False)
    if warm_start is None:
        raise ValueError(
            "--resume requires --load-from-experiment (the run to continue in place)."
        )
    exp = warm_start.experiment
    exp_dir = exp.path
    state = load_training_state(exp_dir)
    wandb_run_id: Optional[str] = None
    if state is not None:
        wandb_run_id = state.get("wandb_run_id")
    if wandb_run_id is None:
        wandb_run_id = exp.metadata.wandb.get("id")
    wandb_run_name = exp.metadata.wandb.get("name") or exp.metadata.name
    if state is None:
        print(
            f"--resume: no {TRAINING_STATE_FILENAME} found for '{exp.metadata.name}'. "
            "Will continue from saved weights but with a fresh optimizer/step counter."
        )
    else:
        print(
            f"--resume: continuing '{exp.metadata.name}' in place "
            f"(global_step={state.get('global_step')}, wandb_run_id={wandb_run_id})."
        )
    return ResumeContext(
        active=True,
        experiment_name=exp.metadata.name,
        exp_dir=exp_dir,
        training_state_path=os.path.join(exp_dir, TRAINING_STATE_FILENAME),
        state=state,
        wandb_run_id=wandb_run_id,
        wandb_run_name=wandb_run_name,
    )


def _assert_config_matches_experiment_architecture(
    config_arch: "ArchitecturePreset", exp: Experiment
) -> None:
    """Raise when ``--config`` and ``--load-from-experiment`` describe different architectures."""
    from catanrl.experiments.architecture_config import architecture_train_config_fields

    exp_arch = architecture_preset_from_experiment(exp)
    if config_arch == exp_arch:
        return

    config_fields = architecture_train_config_fields(config_arch)
    exp_fields = architecture_train_config_fields(exp_arch)
    diffs: list[str] = []
    for key in sorted(set(config_fields) | set(exp_fields)):
        config_value = config_fields.get(key)
        exp_value = exp_fields.get(key)
        if config_value != exp_value:
            diffs.append(f"  {key}: preset={config_value!r}, experiment={exp_value!r}")
    raise ValueError(
        f"--config architecture does not match --load-from-experiment "
        f"'{exp.metadata.name}':\n" + "\n".join(diffs)
    )


def prepare_training_warm_start(
    args: argparse.Namespace,
    *,
    require_critic: bool = False,
    architecture: "ArchitecturePreset | None" = None,
) -> Optional[TrainingWarmStart]:
    """Load checkpoint paths for warm-start. Architecture comes from --config, not the source experiment."""
    name_or_path = getattr(args, "load_from_experiment", None)
    if not name_or_path:
        return None
    which = getattr(args, "load_from_which", "best")
    exp = load_experiment(name_or_path)
    if architecture is not None:
        _assert_config_matches_experiment_architecture(architecture, exp)
    checkpoints = resolve_training_checkpoints_from_experiment(
        exp, which, require_critic=require_critic
    )
    print(
        f"Warm-start checkpoints from '{checkpoints.experiment_name}' "
        f"(which={checkpoints.which})"
    )
    print(f"  policy: {checkpoints.policy}")
    if checkpoints.critic:
        print(f"  critic: {checkpoints.critic}")
    return TrainingWarmStart(experiment=exp, checkpoints=checkpoints)


def architecture_preset_from_experiment(exp: Experiment) -> "ArchitecturePreset":
    """Build an :class:`ArchitecturePreset` from saved experiment metadata."""
    from catanrl.experiments.architecture_config import ArchitecturePreset

    meta = exp.metadata
    tc = meta.train_config
    policy = exp.policy_spec
    critic = meta.networks.get("critic")
    game = meta.game

    policy_dims = tuple(backbone_hidden_dims(policy.backbone))
    if not policy_dims:
        raise ValueError(
            f"Experiment '{meta.name}' has no policy hidden dims in metadata."
        )

    if critic is not None:
        critic_dims = tuple(backbone_hidden_dims(critic.backbone))
    elif isinstance(tc.get("critic_hidden_dims"), list):
        critic_dims = tuple(int(dim) for dim in tc["critic_hidden_dims"])
    else:
        critic_dims = policy_dims

    policy_mode = (
        policy.observation_level
        or tc.get("policy_mode")
        or tc.get("actor_observation_level")
    )
    if policy_mode not in ("private", "public", "full"):
        raise ValueError(
            f"Experiment '{meta.name}' has no policy observation level in metadata."
        )

    if critic is not None and critic.observation_level in ("private", "public", "full"):
        critic_mode = critic.observation_level
    elif tc.get("critic_mode") in ("private", "public", "full"):
        critic_mode = tc["critic_mode"]
    elif tc.get("critic_observation_level") in ("private", "public", "full"):
        critic_mode = tc["critic_observation_level"]
    elif policy.kind == KIND_POLICY_VALUE:
        critic_mode = policy_mode
    else:
        raise ValueError(
            f"Experiment '{meta.name}' has no critic observation level in metadata."
        )

    network_mode = tc.get("network_mode")
    if network_mode is None:
        if policy.kind == KIND_POLICY_VALUE:
            network_mode = "shared"
        elif tc.get("critic_mode") in ("shared", "privileged"):
            network_mode = "shared" if tc["critic_mode"] == "shared" else "separate"
        elif critic is not None:
            network_mode = "separate"
        else:
            network_mode = "shared"

    model_type = policy.model_type or tc.get("model_type")
    if model_type not in ("flat", "hierarchical"):
        raise ValueError(f"Experiment '{meta.name}' has no model_type in metadata.")

    xdim_cnn_channels = (64, 128, 128)
    xdim_cnn_kernel_size = (3, 5)
    xdim_policy_fusion_hidden_dim: int | None = None
    xdim_critic_fusion_hidden_dim: int | None = None

    if isinstance(policy.backbone.args, CrossDimensionalBackboneConfig):
        bb = policy.backbone.args
        xdim_cnn_channels = tuple(bb.cnn_channels)
        xdim_cnn_kernel_size = (bb.cnn_kernel_size[0], bb.cnn_kernel_size[1])
        xdim_policy_fusion_hidden_dim = bb.fusion_hidden_dim

    if critic is not None and isinstance(critic.backbone.args, CrossDimensionalBackboneConfig):
        xdim_critic_fusion_hidden_dim = critic.backbone.args.fusion_hidden_dim

    if isinstance(tc.get("xdim_cnn_channels"), list):
        xdim_cnn_channels = tuple(int(ch) for ch in tc["xdim_cnn_channels"])
    kernel = tc.get("xdim_cnn_kernel_size")
    if isinstance(kernel, list) and len(kernel) == 2:
        xdim_cnn_kernel_size = (int(kernel[0]), int(kernel[1]))
    if tc.get("xdim_policy_fusion_hidden_dim") is not None:
        xdim_policy_fusion_hidden_dim = int(tc["xdim_policy_fusion_hidden_dim"])
    if tc.get("xdim_critic_fusion_hidden_dim") is not None:
        xdim_critic_fusion_hidden_dim = int(tc["xdim_critic_fusion_hidden_dim"])

    return ArchitecturePreset(
        model_type=model_type,
        backbone_type=backbone_display_type(policy.backbone),
        policy_hidden_dims=policy_dims,
        critic_hidden_dims=critic_dims,
        policy_mode=policy_mode,
        critic_mode=critic_mode,
        network_mode=network_mode,
        xdim_cnn_channels=xdim_cnn_channels,
        xdim_cnn_kernel_size=xdim_cnn_kernel_size,
        xdim_policy_fusion_hidden_dim=xdim_policy_fusion_hidden_dim,
        xdim_critic_fusion_hidden_dim=xdim_critic_fusion_hidden_dim,
        map_type=game.map_type,
        vps_to_win=game.vps_to_win or tc.get("vps_to_win") or 15,
        discard_limit=game.discard_limit or tc.get("discard_limit") or 9,
        num_players=game.num_players,
    )


@dataclass
class TrainingArchitectureSetup:
    """Resolved architecture plus optional warm-start checkpoints."""

    arch: "ArchitecturePreset"
    warm_start: Optional[TrainingWarmStart]
    architecture_source: str


def resolve_training_architecture_and_warm_start(
    args: argparse.Namespace,
) -> TrainingArchitectureSetup:
    """Require ``--config`` and/or ``--load-from-experiment``; resolve architecture and warm-start."""
    from catanrl.experiments.architecture_config import (
        load_architecture_preset,
        validate_architecture_source,
    )

    config_path = getattr(args, "config", None)
    load_from = getattr(args, "load_from_experiment", None)
    validate_architecture_source(config_path, load_from)

    exp: Optional[Experiment] = None
    which = getattr(args, "load_from_which", "best")
    if load_from:
        exp = load_experiment(load_from)

    if config_path:
        arch = load_architecture_preset(config_path)
        architecture_source = config_path
        if exp is not None:
            _assert_config_matches_experiment_architecture(arch, exp)
    else:
        assert exp is not None
        arch = architecture_preset_from_experiment(exp)
        architecture_source = f"experiment:{exp.metadata.name}"

    warm_start: Optional[TrainingWarmStart] = None
    if exp is not None:
        require_critic = arch.network_mode == "separate"
        checkpoints = resolve_training_checkpoints_from_experiment(
            exp, which, require_critic=require_critic
        )
        print(
            f"Warm-start checkpoints from '{checkpoints.experiment_name}' "
            f"(which={checkpoints.which})"
        )
        print(f"  policy: {checkpoints.policy}")
        if checkpoints.critic:
            print(f"  critic: {checkpoints.critic}")
        warm_start = TrainingWarmStart(experiment=exp, checkpoints=checkpoints)

    return TrainingArchitectureSetup(
        arch=arch,
        warm_start=warm_start,
        architecture_source=architecture_source,
    )


# --------------------------------------------------------------------------- #
# Writing (for trainers / migration)
# --------------------------------------------------------------------------- #
def current_git_sha() -> Optional[str]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return sha.decode().strip()
    except Exception:
        return None


def network_spec_from_model(
    model: torch.nn.Module,
    *,
    kind: str,
    model_type: Optional[str] = None,
    observation_level: Optional[str] = None,
) -> NetworkSpec:
    """Build a :class:`NetworkSpec` from a freshly constructed model.

    Relies on the ``backbone_config`` / ``action_space_size`` attributes that
    the model builders attach in ``catanrl.models.models``.
    """
    backbone = getattr(model, "backbone_config", None)
    if not isinstance(backbone, BackboneConfig):
        raise ValueError(
            "Model has no `backbone_config`; build it via catanrl.models.models builders."
        )
    num_actions = getattr(model, "action_space_size", None)
    return NetworkSpec(
        kind=kind,
        backbone=backbone,
        model_type=model_type,
        observation_level=observation_level,
        num_actions=num_actions if kind != KIND_VALUE else None,
    )


def save_metadata(path: str, metadata: ExperimentMetadata) -> str:
    os.makedirs(path, exist_ok=True)
    meta_path = os.path.join(path, METADATA_FILENAME)
    with open(meta_path, "w") as f:
        json.dump(metadata.to_dict(), f, indent=2, sort_keys=False)
    return meta_path


def save_checkpoint_registry(path: str, registry: CheckpointRegistry) -> str:
    os.makedirs(path, exist_ok=True)
    registry_path = os.path.join(path, CHECKPOINTS_FILENAME)
    with open(registry_path, "w") as f:
        json.dump(registry.to_dict(), f, indent=2, sort_keys=False)
    return registry_path


# --------------------------------------------------------------------------- #
# Checkpoint discovery (arch-free: groups *.pt files by role + best/latest)
# --------------------------------------------------------------------------- #
_STEP_RE = re.compile(r"(\d+)")


def _role_of(filename: str) -> Optional[str]:
    name = filename.lower()
    if not name.endswith(".pt"):
        return None
    if "critic" in name:
        return "critic"
    # Everything else (policy_*, best.pt, policy_value.pt, ...) is a
    # policy-bearing checkpoint (possibly a joint policy-value network).
    return "policy"


def _step_of(filename: str) -> Optional[int]:
    nums = _STEP_RE.findall(filename)
    return int(nums[-1]) if nums else None


def discover_checkpoints(ckpt_dir: str) -> Dict[str, Dict]:
    """Group *.pt files in ``ckpt_dir`` by role with best/latest selection."""
    files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
    roles: Dict[str, Dict] = {}
    for role in ("policy", "critic"):
        # Critic-tagged files must not also be counted as policy.
        role_files = [f for f in files if _role_of(f) == role]
        if not role_files:
            continue
        best = next(
            (
                f
                for f in role_files
                if f
                in (
                    f"{role}_best.pt",
                    "best.pt",
                    "policy_value_best.pt",
                    "policy_value.pt",
                )
            ),
            None,
        )
        if best is None:
            best = next((f for f in role_files if "best" in f.lower()), None)
        stepped = [(f, _step_of(f)) for f in role_files if _step_of(f) is not None]
        latest = max(stepped, key=lambda x: x[1])[0] if stepped else best
        roles[role] = {
            "files": sorted(role_files),
            "best": best,
            "latest": latest,
            "stepped": stepped,
        }
    return roles


def build_checkpoint_registry(
    ckpt_dir: str,
    roles_present: Optional[List[str]] = None,
    *,
    relative_prefix: str = CHECKPOINTS_DIRNAME,
) -> CheckpointRegistry:
    """Build a :class:`CheckpointRegistry` by scanning ``ckpt_dir`` for *.pt files.

    Only roles in ``roles_present`` (if given) are registered; this lets a
    caller restrict the registry to the networks it actually saved.
    """
    discovered = discover_checkpoints(ckpt_dir)
    selectors: Dict[str, Dict[str, str]] = {"best": {}, "latest": {}}
    checkpoints: List[dict] = []
    for role, info in discovered.items():
        if roles_present is not None and role not in roles_present:
            continue
        if info["best"]:
            selectors["best"][role] = f"{relative_prefix}/{info['best']}"
        if info["latest"]:
            selectors["latest"][role] = f"{relative_prefix}/{info['latest']}"
        for f, step in info["stepped"]:
            checkpoints.append({"step": step, "role": role, "file": f"{relative_prefix}/{f}"})
    return CheckpointRegistry(selectors=selectors, checkpoints=checkpoints)


# --------------------------------------------------------------------------- #
# Experiment naming / location
# --------------------------------------------------------------------------- #
def default_checkpoints_dir(name: str) -> str:
    """Standard checkpoints directory for a new experiment: experiments/<name>/checkpoints."""
    return os.path.join(experiments_root(), name, CHECKPOINTS_DIRNAME)


def make_experiment_name(
    algorithm: str,
    wandb_run_name: Optional[str] = None,
    explicit: Optional[str] = None,
) -> str:
    """Resolve the experiment name for a training run.

    Precedence: an explicit name, then the W&B run name (so they always match),
    then a generated ``<algorithm>-<timestamp>`` fallback.
    """
    if explicit:
        return explicit
    if wandb_run_name:
        return wandb_run_name
    ts = _datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{algorithm}-{ts}"


# --------------------------------------------------------------------------- #
# High-level writer: build an experiment from explicit specs + saved files
# --------------------------------------------------------------------------- #
def save_experiment(
    name: str,
    ckpt_dir: str,
    *,
    algorithm: str,
    game: GameConfig,
    networks: Dict[str, NetworkSpec],
    train_config: Optional[dict] = None,
    wandb_info: Optional[dict] = None,
    notes: Optional[str] = None,
    relative_prefix: str = CHECKPOINTS_DIRNAME,
) -> str:
    """Write ``experiments/<name>/`` from explicit network specs.

    Unlike the old shape-inference path, the architecture is taken directly from
    ``networks`` (built by the trainer that just trained the models), so the
    metadata is guaranteed to match the saved weights without any re-derivation.

    ``ckpt_dir`` is expected to be the experiment's own
    ``experiments/<name>/checkpoints`` directory (see :func:`default_checkpoints_dir`).
    Returns the experiment directory path.
    """
    exp_path = experiment_dir(name)
    os.makedirs(exp_path, exist_ok=True)

    registry = build_checkpoint_registry(
        ckpt_dir, roles_present=list(networks.keys()), relative_prefix=relative_prefix
    )

    metadata = ExperimentMetadata(
        name=name,
        algorithm=algorithm,
        game=game,
        networks=networks,
        git_sha=current_git_sha(),
        train_config=train_config or {},
        wandb=wandb_info or {},
        notes=notes,
    )

    save_metadata(exp_path, metadata)
    save_checkpoint_registry(exp_path, registry)
    return exp_path


__all__ = [
    "SCHEMA_VERSION",
    "KIND_POLICY",
    "KIND_POLICY_VALUE",
    "KIND_VALUE",
    "GameConfig",
    "NetworkSpec",
    "ExperimentMetadata",
    "CheckpointRegistry",
    "Experiment",
    "TrainingCheckpoints",
    "TrainingWarmStart",
    "TrainingArchitectureSetup",
    "load_experiment",
    "architecture_preset_from_experiment",
    "resolve_training_architecture_and_warm_start",
    "apply_experiment_architecture_to_args",
    "resolve_training_checkpoints",
    "resolve_training_checkpoints_from_experiment",
    "add_load_from_experiment_arguments",
    "add_resume_argument",
    "wandb_grouping_kwargs",
    "training_state_file",
    "save_training_state",
    "load_training_state",
    "ResumeContext",
    "prepare_resume",
    "prepare_training_warm_start",
    "build_network",
    "experiments_root",
    "experiment_dir",
    "backbone_config_to_dict",
    "backbone_config_from_dict",
    "backbone_display_type",
    "backbone_hidden_dims",
    "network_spec_from_model",
    "current_git_sha",
    "save_metadata",
    "save_checkpoint_registry",
    "discover_checkpoints",
    "build_checkpoint_registry",
    "default_checkpoints_dir",
    "make_experiment_name",
    "save_experiment",
]
