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
        self.last_expanded: bool = False   # delta_exp of the most recent begin_task

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
        # delta_exp (paper Eq. 15): whether this task expands the shared subspace.
        self.last_expanded = need_expand

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


# ---------------------------------------------------------------------------
# TuKA++ Factor-wise Knowledge Inheritance and Exploration (FKIE) regularizer.
#
# It implements the three FKIE losses and gates them exactly as in the paper's
# overall objective (Eq. 27):
#
#     L_t = L_nav + delta_exp * L_orth + (1 - delta_exp) * (L_Fish + L_con)
#
# where delta_exp = 1 when the current task introduces a NEW factor category
# (exploration -> orthogonality separates the new expert), and delta_exp = 0
# when the task is a new composition of already-seen factors (inheritance ->
# consistency + Fisher-aware regularization stabilize the reused knowledge).
#
# Wire regularization() into the training loss, call update_fisher() after
# loss.backward() and before optimizer.step(), and set_expanded() per task.
# ---------------------------------------------------------------------------
class TuKARegularizer:
    """
    FKIE losses (paper Eq. 13 / 14 / 18):
      * L_orth : orthogonality among the factor experts U3/U4/U5, encouraging
                 decoupled scene / environment / instruction-style knowledge,
                 i.e. || Norm(U_k) Norm(U_k)^T - I ||_F^2 over the existing rows
                 (Eq. 13). Applied when a new factor category appears.
      * L_con  : consistency of the reused factor experts U3/U4/U5 with their
                 value at the start of the current task (Eq. 14), so inherited
                 factor knowledge is only refined, not overwritten.
      * L_Fish : Fisher-aware penalty on the shared subspace {U1, U2, G},
                 sum_i F_i (theta_i - theta_i^old)^2 with a smoothed Fisher
                 F <- omega*F + (1-omega)*grad^2 (Eq. 16-18).

    The gate delta_exp (whether the task expanded the subspace) is provided via
    set_expanded(); it selects the paper's per-task loss combination (Eq. 27).
    """

    _SHARED_TAGS = ("lora_layer.U1", "lora_layer.U2", "lora_layer.G")
    _FACTOR_TAGS = ("lora_layer.U3", "lora_layer.U4", "lora_layer.U5")

    def __init__(self, model, lambda_c=1.0, lambda_o=0.1, lambda_f=0.01, fisher_omega=0.9):
        self.model = model
        self.lambda_c = float(lambda_c)
        self.lambda_o = float(lambda_o)
        self.lambda_f = float(lambda_f)
        self.omega = float(fisher_omega)
        self.expanded = True          # delta_exp for the current task (Eq. 15)
        self._anchor: Dict[str, torch.Tensor] = {}   # theta at task start
        self._fisher: Dict[str, torch.Tensor] = {}    # smoothed Fisher (shared)
        self.snapshot()

    def set_expanded(self, expanded: bool):
        """Set delta_exp for the current task (True iff a new factor appeared)."""
        self.expanded = bool(expanded)

    def _shared_params(self):
        for name, p in self.model.named_parameters():
            if p.requires_grad and any(t in name for t in self._SHARED_TAGS):
                yield name, p

    def _factor_params(self):
        for name, p in self.model.named_parameters():
            if p.requires_grad and any(t in name for t in self._FACTOR_TAGS):
                yield name, p

    def snapshot(self):
        """Anchor shared subspace + factor experts at the start of a task."""
        self._anchor = {
            n: p.detach().clone()
            for n, p in list(self._shared_params()) + list(self._factor_params())
        }

    def orthogonality_loss(self):
        # Eq. 13: separate factor experts using row-normalized Gram matrices.
        loss = None
        for layer in iter_tucker5d_layers(self.model):
            for U in (layer.U3, layer.U4, layer.U5):
                if U.shape[0] > 1:
                    U_norm = U / (U.norm(dim=1, keepdim=True) + 1e-8)
                    G = U_norm @ U_norm.t()
                    I = torch.eye(G.shape[0], device=G.device, dtype=G.dtype)
                    term = ((G - I) ** 2).sum()
                    loss = term if loss is None else loss + term
        return loss if loss is not None else torch.zeros((), device=next(self.model.parameters()).device)

    def consistency_loss(self):
        # Eq. 14: keep reused factor experts close to their pre-task value.
        loss = None
        for name, p in self._factor_params():
            a = self._anchor.get(name)
            if a is not None and a.shape == p.shape:
                term = ((p - a) ** 2).sum()
                loss = term if loss is None else loss + term
        return loss if loss is not None else torch.zeros((), device=next(self.model.parameters()).device)

    def fisher_loss(self):
        # Eq. 18: Fisher-aware penalty on the shared subspace {U1, U2, G}.
        loss = None
        for name, p in self._shared_params():
            f = self._fisher.get(name); a = self._anchor.get(name)
            if f is not None and a is not None and f.shape == p.shape:
                term = (f * (p - a) ** 2).sum()
                loss = term if loss is None else loss + term
        return loss if loss is not None else torch.zeros((), device=next(self.model.parameters()).device)

    @torch.no_grad()
    def update_fisher(self):
        """Smoothed Fisher from squared grads; call after backward(), before step()."""
        for name, p in self._shared_params():
            if p.grad is None:
                continue
            g2 = p.grad.detach() ** 2
            prev = self._fisher.get(name)
            self._fisher[name] = g2 if prev is None or prev.shape != g2.shape \
                else self.omega * prev + (1.0 - self.omega) * g2

    def regularization(self):
        # Paper Eq. 27:
        #   delta_exp == 1 -> orthogonality drives exploration of the new factor
        #   delta_exp == 0 -> Fisher + consistency inherit reused knowledge
        if self.expanded:
            return self.lambda_o * self.orthogonality_loss()
        return (self.lambda_c * self.consistency_loss()
                + self.lambda_f * self.fisher_loss())
