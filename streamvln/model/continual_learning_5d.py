"""
Continual-learning manager for 5D Tucker-LoRA.

Replaces EWC with Progressive Shared-Subspace Expansion + Zero-Padding.

Per task t = (S_s, E_e, L_p):
    1. Decide if expansion is needed:
           1_expand(t) = 1  iff  s is new  OR  e is new  OR  p is new
       If YES:
           * grow r1, r2 by (delta_r1, delta_r2)          (shared subspace)
           * grow r3 by delta_r3  iff s is new
           * grow r4 by delta_r4  iff e is new
           * grow r5 by delta_r5  iff p is new
           * append ONE new row to U3/U4/U5 per newly-seen category
       If NO:
           * re-use existing coords; only the active (s,e,p) rows on the
             current task's own new-coord block can learn. (This block was
             created by the most recent prior expansion; if no expansion
             ever happened for this triple's novelty, the task is strictly
             interpolative and only the active rows of U3/U4/U5 in the
             existing (possibly already frozen) subspace are trained.)

    2. set_active_route(s, e, p) on every Tucker5D layer.
    3. Register a backward hook (or post-backward call) that invokes
       zero_inactive_gradients_all() before optimizer.step().
    4. After optimizer.step() -> enforce_zero_pad_all().
    5. After the WHOLE task's training finishes -> commit_frozen_all().

State persists across tasks via a small JSON file so that a new training
run can resume the expansion state from where the previous one left off.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import torch

from .tucker_5d_lora_layers import (
    Tucker5DLoRALinear,
    Tucker5DLoRALayer,
    iter_tucker5d_layers,
    set_active_route_all,
    zero_inactive_gradients_all,
    enforce_zero_pad_all,
    commit_frozen_all,
    expand_all,
    append_category_row_all,
)


# ---------------------------------------------------------------------------
# Global persistent state
# ---------------------------------------------------------------------------

@dataclass
class CategoryState:
    """Track which categories have been seen and their row indices in U3/U4/U5."""
    seen_scenes: List[int] = field(default_factory=list)     # original scene ids
    seen_envs: List[int] = field(default_factory=list)
    seen_instrs: List[int] = field(default_factory=list)

    scene_row: Dict[int, int] = field(default_factory=dict)  # original -> row
    env_row: Dict[int, int] = field(default_factory=dict)
    instr_row: Dict[int, int] = field(default_factory=dict)

    # Current ranks (mirrored from layer state for convenience)
    r1: int = 0
    r2: int = 0
    r3: int = 0
    r4: int = 0
    r5: int = 0

    completed_tasks: List[Tuple[int, int, int]] = field(default_factory=list)

    def as_dict(self):
        return {
            "seen_scenes": self.seen_scenes,
            "seen_envs": self.seen_envs,
            "seen_instrs": self.seen_instrs,
            # JSON cannot have int keys -> stringify on save, int on load
            "scene_row": {str(k): v for k, v in self.scene_row.items()},
            "env_row": {str(k): v for k, v in self.env_row.items()},
            "instr_row": {str(k): v for k, v in self.instr_row.items()},
            "r1": self.r1,
            "r2": self.r2,
            "r3": self.r3,
            "r4": self.r4,
            "r5": self.r5,
            "completed_tasks": self.completed_tasks,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CategoryState":
        return cls(
            seen_scenes=list(d.get("seen_scenes", [])),
            seen_envs=list(d.get("seen_envs", [])),
            seen_instrs=list(d.get("seen_instrs", [])),
            scene_row={int(k): v for k, v in d.get("scene_row", {}).items()},
            env_row={int(k): v for k, v in d.get("env_row", {}).items()},
            instr_row={int(k): v for k, v in d.get("instr_row", {}).items()},
            r1=d.get("r1", 0),
            r2=d.get("r2", 0),
            r3=d.get("r3", 0),
            r4=d.get("r4", 0),
            r5=d.get("r5", 0),
            completed_tasks=[tuple(t) for t in d.get("completed_tasks", [])],
        )


# ---------------------------------------------------------------------------
# Task Expansion Manager
# ---------------------------------------------------------------------------

class TaskExpansionManager:
    """
    Orchestrates per-task expand -> route -> train -> commit_freeze.

    Usage per task:
        mgr = TaskExpansionManager(model, state_path=..., delta_r=...)
        s_row, e_row, p_row = mgr.begin_task(scene_idx, env_idx, instr_idx)
        ... training loop ...
        mgr.end_task()
    """

    def __init__(
        self,
        model: torch.nn.Module,
        state_path: Optional[str] = None,
        delta_r1: int = 4,
        delta_r2: int = 4,
        delta_r3: int = 2,
        delta_r4: int = 2,
        delta_r5: int = 1,
        expand_on_any_new: bool = True,
        verbose: bool = True,
    ):
        self.model = model
        self.state_path = state_path
        self.delta_r1 = delta_r1
        self.delta_r2 = delta_r2
        self.delta_r3 = delta_r3
        self.delta_r4 = delta_r4
        self.delta_r5 = delta_r5
        self.expand_on_any_new = expand_on_any_new
        self.verbose = verbose

        self.state = CategoryState()
        if state_path and os.path.exists(state_path):
            self.load()
        else:
            self._sync_state_from_model()

        self._current_triple: Optional[Tuple[int, int, int]] = None

    # -- persistence -------------------------------------------------------

    def save(self, path: Optional[str] = None):
        path = path or self.state_path
        if path is None:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.state.as_dict(), f, indent=2)
        if self.verbose:
            print(f"[TaskExpansionManager] saved state -> {path}")

    def load(self, path: Optional[str] = None):
        path = path or self.state_path
        if path is None or not os.path.exists(path):
            return
        with open(path, "r") as f:
            d = json.load(f)
        self.state = CategoryState.from_dict(d)
        if self.verbose:
            print(f"[TaskExpansionManager] loaded state <- {path} "
                  f"(scenes={len(self.state.seen_scenes)}, envs={len(self.state.seen_envs)}, "
                  f"instrs={len(self.state.seen_instrs)}, ranks=({self.state.r1},{self.state.r2},"
                  f"{self.state.r3},{self.state.r4},{self.state.r5}))")

    def _sync_state_from_model(self):
        first = next(iter_tucker5d_layers(self.model), None)
        if first is None:
            return
        self.state.r1 = first.r1
        self.state.r2 = first.r2
        self.state.r3 = first.r3
        self.state.r4 = first.r4
        self.state.r5 = first.r5

    # -- task lifecycle ---------------------------------------------------

    def begin_task(self, scene_idx: int, env_idx: int, instr_idx: int) -> Tuple[int, int, int]:
        """
        Return (scene_row, env_row, instr_row) to pass to set_active_route.
        Expands + appends rows as necessary. Call BEFORE the training loop.
        """
        scene_new = scene_idx not in self.state.scene_row
        env_new = env_idx not in self.state.env_row
        instr_new = instr_idx not in self.state.instr_row

        need_expand = scene_new or env_new or instr_new

        if need_expand and self.expand_on_any_new:
            expand_all(
                self.model,
                delta_r1=self.delta_r1 if (scene_new or env_new or instr_new) else 0,
                delta_r2=self.delta_r2 if (scene_new or env_new or instr_new) else 0,
                delta_r3=self.delta_r3 if scene_new else 0,
                delta_r4=self.delta_r4 if env_new else 0,
                delta_r5=self.delta_r5 if instr_new else 0,
            )

        # Append rows for newly-seen categories
        if scene_new:
            row = append_category_row_all(self.model, "U3")
            self.state.scene_row[scene_idx] = row
            self.state.seen_scenes.append(scene_idx)
        if env_new:
            row = append_category_row_all(self.model, "U4")
            self.state.env_row[env_idx] = row
            self.state.seen_envs.append(env_idx)
        if instr_new:
            row = append_category_row_all(self.model, "U5")
            self.state.instr_row[instr_idx] = row
            self.state.seen_instrs.append(instr_idx)

        # Sync rank state
        self._sync_state_from_model()

        s_row = self.state.scene_row[scene_idx]
        e_row = self.state.env_row[env_idx]
        p_row = self.state.instr_row[instr_idx]

        set_active_route_all(self.model, s_row, e_row, p_row)
        self._current_triple = (scene_idx, env_idx, instr_idx)

        if self.verbose:
            print(f"[TaskExpansionManager] begin_task  scene={scene_idx}(row={s_row})  "
                  f"env={env_idx}(row={e_row})  instr={instr_idx}(row={p_row})  "
                  f"expanded={need_expand}  ranks=({self.state.r1},{self.state.r2},"
                  f"{self.state.r3},{self.state.r4},{self.state.r5})")
        return s_row, e_row, p_row

    def pre_step(self):
        """Call between loss.backward() and optimizer.step()."""
        zero_inactive_gradients_all(self.model)

    def post_step(self):
        """Call after optimizer.step()."""
        enforce_zero_pad_all(self.model)

    def end_task(self):
        """Commit the current shape as frozen. Call when a task finishes."""
        commit_frozen_all(self.model)
        if self._current_triple is not None:
            self.state.completed_tasks.append(self._current_triple)
        self._sync_state_from_model()
        self.save()
        if self.verbose:
            print(f"[TaskExpansionManager] end_task  total completed={len(self.state.completed_tasks)}  "
                  f"frozen ranks=({self.state.r1},{self.state.r2},{self.state.r3},"
                  f"{self.state.r4},{self.state.r5})")
        self._current_triple = None

    # -- introspection ----------------------------------------------------

    def resolve_route(self, scene_idx: int, env_idx: int, instr_idx: int) -> Tuple[int, int, int]:
        """For inference: map original (scene,env,instr) ids to row indices."""
        if scene_idx not in self.state.scene_row:
            raise KeyError(f"Unknown scene id {scene_idx} (seen: {self.state.seen_scenes})")
        if env_idx not in self.state.env_row:
            raise KeyError(f"Unknown env id {env_idx} (seen: {self.state.seen_envs})")
        if instr_idx not in self.state.instr_row:
            raise KeyError(f"Unknown instr id {instr_idx} (seen: {self.state.seen_instrs})")
        return (
            self.state.scene_row[scene_idx],
            self.state.env_row[env_idx],
            self.state.instr_row[instr_idx],
        )

    def num_trainable_params(self) -> int:
        n = 0
        for p in self.model.parameters():
            if p.requires_grad:
                n += p.numel()
        return n
