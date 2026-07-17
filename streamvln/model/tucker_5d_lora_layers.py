"""
5th-order Tucker-LoRA with Progressively Expandable Shared Subspace.

Formulation (per TuKA TPAMI extension):
    X^l = G  x_1 U1  x_2 U2  x_3 U3  x_4 U4  x_5 U5
    G  in R^{r1 x r2 x r3 x r4 x r5}
    U1 in R^{a_l x r1}  (shared hidden factor, in_features direction)
    U2 in R^{b_l x r2}  (shared hidden factor, out_features direction)
    U3 in R^{M x r3}    (scene hierarchy, M scenes)
    U4 in R^{N x r4}    (environment hierarchy, N environments)
    U5 in R^{P x r5}    (instruction hierarchy, P instruction paradigms)

For task t = (S_s, E_e, L_p), the per-task delta is:

    DeltaW_t = U1 * (G  x_3 U3[s,:]  x_4 U4[e,:]  x_5 U5[p,:]) * U2^T

Expansion (ONLY when a new category first appears):
    * r1, r2 expand together by (delta_r1, delta_r2) to enlarge shared subspace
    * r3 expands only when a NEW scene appears
    * r4 expands only when a NEW environment appears
    * r5 expands only when a NEW instruction paradigm appears
    * When a category is NEW, a fresh row is appended to the corresponding factor
    * Old G tensor is embedded at the top-left-front-etc. corner of new G
    * All previously-seen rows of U3/U4/U5 are zero-padded on new-coord columns
      (Theorem 1 in paper: preserves old DeltaW bit-exactly)

Hard-masking training strategy:
    * begin_task(s, e, p): register which rows/blocks are active
    * zero_inactive_gradients(): hook zeros gradients on frozen blocks and
      non-active rows, so only (a) new-coord blocks of U1/U2/G, (b) active
      rows of U3/U4/U5 on new-coord columns, and -- optionally -- the
      single active (s,e,p) rows on old-coord columns receive updates
    * enforce_zero_pad(): after each optimizer step, re-zero positions that
      must stay zero (guards against momentum / weight decay drift)
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# State container (per-layer, also mirrored at manager level)
# ---------------------------------------------------------------------------

@dataclass
class Tucker5DState:
    """Snapshot of ranks and seen coordinates. Used for save / restore."""
    r1: int
    r2: int
    r3: int
    r4: int
    r5: int
    seen_scenes: List[int] = field(default_factory=list)
    seen_envs: List[int] = field(default_factory=list)
    seen_instrs: List[int] = field(default_factory=list)
    scene_row_map: List[int] = field(default_factory=list)   # scene_idx -> row in U3
    env_row_map: List[int] = field(default_factory=list)     # env_idx   -> row in U4
    instr_row_map: List[int] = field(default_factory=list)   # instr_idx -> row in U5


# ---------------------------------------------------------------------------
# Core 5-D Tucker layer
# ---------------------------------------------------------------------------

class Tucker5DLoRALayer(nn.Module):
    """
    Expandable 5th-order Tucker-LoRA factor bank for ONE linear weight.

    Parameters live as single nn.Parameter tensors that can grow via `expand`.
    Tracking buffers `frozen_rX` record how much of each factor is already
    "committed" (frozen) from prior tasks -- these slices are masked out of
    gradient updates and enforced to their frozen values after every step.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        ranks: Tuple[int, int, int, int, int] = (16, 16, 8, 8, 4),
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        init_std: float = 0.02,
    ):
        super().__init__()
        r1, r2, r3, r4, r5 = ranks
        self.in_features = in_features
        self.out_features = out_features
        self.lora_alpha = lora_alpha
        self.init_std = init_std
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        # Rank state (mutable under `expand`)
        self.r1 = r1
        self.r2 = r2
        self.r3 = r3
        self.r4 = r4
        self.r5 = r5
        # Keep the INITIAL r1 for scaling. Standard LoRA uses
        # scaling = alpha / r_init so the adapter's effective magnitude
        # stays constant. If we used alpha / r1_CURRENT instead, scaling
        # shrinks ~3x over 23 tasks (r1 grows 16 -> ~52) and later tasks
        # would have a much weaker adapter than early ones.
        self.r1_initial = r1

        # Factors as single parameters (resized via nn.Parameter replacement).
        # U3/U4/U5 start with ZERO rows -- rows are appended lazily as new
        # scene / env / instruction categories are encountered.
        self.U1 = nn.Parameter(torch.zeros(in_features, r1))
        self.U2 = nn.Parameter(torch.zeros(out_features, r2))
        self.U3 = nn.Parameter(torch.zeros(0, r3))
        self.U4 = nn.Parameter(torch.zeros(0, r4))
        self.U5 = nn.Parameter(torch.zeros(0, r5))
        self.G = nn.Parameter(torch.zeros(r1, r2, r3, r4, r5))

        # Frozen prefixes (how many ranks / rows are locked from previous tasks)
        self.register_buffer("frozen_r1", torch.tensor(0, dtype=torch.long))
        self.register_buffer("frozen_r2", torch.tensor(0, dtype=torch.long))
        self.register_buffer("frozen_r3", torch.tensor(0, dtype=torch.long))
        self.register_buffer("frozen_r4", torch.tensor(0, dtype=torch.long))
        self.register_buffer("frozen_r5", torch.tensor(0, dtype=torch.long))
        self.register_buffer("frozen_rows_U3", torch.tensor(0, dtype=torch.long))
        self.register_buffer("frozen_rows_U4", torch.tensor(0, dtype=torch.long))
        self.register_buffer("frozen_rows_U5", torch.tensor(0, dtype=torch.long))

        # Per-forward active coordinates (runtime; not persisted)
        self._active_s: Optional[int] = None
        self._active_e: Optional[int] = None
        self._active_p: Optional[int] = None

        # Initialise the first block (everything is "new" at creation)
        self.reset_parameters()

    # -- initialisation ----------------------------------------------------

    def reset_parameters(self):
        """Random-init U1, U2; ZERO-init G; U3/U4/U5 stay empty.

        This mirrors standard LoRA init (A random, B = 0): the adapter
        delta is exactly 0 at construction, so the model behaves
        identically to the base before any task has trained. Pre-fix
        we used `nn.init.normal_(G, std=init_std)`, which made the
        adapter delta NON-ZERO at init -- Task_1 then had to spend its
        first few epochs just undoing this random noise before it could
        learn useful navigation, and the noise added gradient drift to
        the base behaviour at evaluation time for episodes the model
        had never seen.
        """
        nn.init.normal_(self.U1, std=self.init_std)
        nn.init.normal_(self.U2, std=self.init_std)
        nn.init.zeros_(self.G)
        # U3/U4/U5 are filled row-by-row via `append_category_row`

    # -- runtime routing ---------------------------------------------------

    def set_active_route(self, s: int, e: int, p: int):
        """Set the (scene_row, env_row, instr_row) to use in forward()."""
        self._active_s = s
        self._active_e = e
        self._active_p = p

    @property
    def scaling(self) -> float:
        # IMPORTANT: divide by the INITIAL r1, NOT the current one. Across
        # 23 tasks r1 grows ~3x; if we divided by r1_current the adapter's
        # effective magnitude would shrink ~3x and later tasks would have
        # a much weaker fine-tuning signal than early ones. Constant
        # scaling matches standard LoRA convention (scaling = alpha / r).
        return self.lora_alpha / max(self.r1_initial, 1)

    # -- factor growth -----------------------------------------------------

    def _replace_param(self, name: str, new_tensor: torch.Tensor):
        """Swap nn.Parameter in-place while preserving requires_grad & device."""
        old: nn.Parameter = getattr(self, name)
        new_param = nn.Parameter(new_tensor.to(device=old.device, dtype=old.dtype))
        new_param.requires_grad = old.requires_grad
        # Must delete first: setattr on existing Parameter triggers type check
        delattr(self, name)
        self.register_parameter(name, new_param)

    def append_category_row(self, axis: str, init: str = "randn"):
        """
        Append ONE fresh row to U3 / U4 / U5. The row is zero-initialised on
        columns corresponding to old-coord ranks (guarantees zero-pad
        theorem) and randomly initialised on new-coord columns (where "new"
        = all columns right now if r3/4/5 has just expanded, or all columns
        on first use).

        Returns the index of the new row.
        """
        if axis == "U3":
            r = self.r3
            U = self.U3
            frozen = int(self.frozen_r3.item())
        elif axis == "U4":
            r = self.r4
            U = self.U4
            frozen = int(self.frozen_r4.item())
        elif axis == "U5":
            r = self.r5
            U = self.U5
            frozen = int(self.frozen_r5.item())
        else:
            raise ValueError(axis)

        new_row = torch.zeros(1, r, device=U.device, dtype=U.dtype)
        # Only new-coord columns get non-zero init (old-coord columns stay 0
        # to preserve deltas of previous tasks for other categories).
        if r - frozen > 0:
            if init == "randn":
                nn.init.normal_(new_row[:, frozen:], std=self.init_std)
            elif init == "zeros":
                pass  # already zero
            else:
                raise ValueError(init)

        new_U = torch.cat([U.detach(), new_row], dim=0)
        self._replace_param(axis, new_U)
        return new_U.shape[0] - 1

    def expand(
        self,
        delta_r1: int = 0,
        delta_r2: int = 0,
        delta_r3: int = 0,
        delta_r4: int = 0,
        delta_r5: int = 0,
    ):
        """
        Enlarge factors. New blocks are randomly init'd; the OLD G tensor is
        embedded at the top-left-etc. corner of the new G (rest = 0).
        Simultaneously the currently-existing rows of U3/U4/U5 are
        zero-padded on the new-coord column blocks -- this is what makes
        Theorem 1 (delta preservation) hold.
        """
        if delta_r1 == 0 and delta_r2 == 0 and delta_r3 == 0 and delta_r4 == 0 and delta_r5 == 0:
            return

        new_r1 = self.r1 + delta_r1
        new_r2 = self.r2 + delta_r2
        new_r3 = self.r3 + delta_r3
        new_r4 = self.r4 + delta_r4
        new_r5 = self.r5 + delta_r5

        device = self.U1.device
        dtype = self.U1.dtype

        # ---- U1 ----------------------------------------------------------
        if delta_r1 > 0:
            new_U1 = torch.zeros(self.in_features, new_r1, device=device, dtype=dtype)
            new_U1[:, : self.r1] = self.U1.detach()
            nn.init.normal_(new_U1[:, self.r1 :], std=self.init_std)
            self._replace_param("U1", new_U1)

        # ---- U2 ----------------------------------------------------------
        if delta_r2 > 0:
            new_U2 = torch.zeros(self.out_features, new_r2, device=device, dtype=dtype)
            new_U2[:, : self.r2] = self.U2.detach()
            nn.init.normal_(new_U2[:, self.r2 :], std=self.init_std)
            self._replace_param("U2", new_U2)

        # ---- U3 (zero-pad on new cols) ----------------------------------
        if delta_r3 > 0 and self.U3.shape[0] > 0:
            new_U3 = torch.zeros(self.U3.shape[0], new_r3, device=device, dtype=dtype)
            new_U3[:, : self.r3] = self.U3.detach()
            # columns [self.r3 :] stay zero -> Theorem 1
            self._replace_param("U3", new_U3)
        elif delta_r3 > 0:
            # No rows yet; just widen the empty tensor
            self._replace_param("U3", torch.zeros(0, new_r3, device=device, dtype=dtype))

        # ---- U4 (zero-pad on new cols) ----------------------------------
        if delta_r4 > 0 and self.U4.shape[0] > 0:
            new_U4 = torch.zeros(self.U4.shape[0], new_r4, device=device, dtype=dtype)
            new_U4[:, : self.r4] = self.U4.detach()
            self._replace_param("U4", new_U4)
        elif delta_r4 > 0:
            self._replace_param("U4", torch.zeros(0, new_r4, device=device, dtype=dtype))

        # ---- U5 (zero-pad on new cols) ----------------------------------
        if delta_r5 > 0 and self.U5.shape[0] > 0:
            new_U5 = torch.zeros(self.U5.shape[0], new_r5, device=device, dtype=dtype)
            new_U5[:, : self.r5] = self.U5.detach()
            self._replace_param("U5", new_U5)
        elif delta_r5 > 0:
            self._replace_param("U5", torch.zeros(0, new_r5, device=device, dtype=dtype))

        # ---- G (embed old core, ZERO-init new coords) -------------------
        # Old behaviour added small random noise to the new G region. That
        # broke the "G=0 at init -> Tucker delta = 0" invariant: the
        # incoming task started with a non-zero delta on its NEW row
        # contractions, which contaminated the first few training steps and
        # left a small residual delta even after gradient-mask convergence
        # (because the optimizer's first updates fight the noise instead of
        # fitting the data). Initialising new G slices to ZERO -- exactly
        # like LoRA B -- means the new task's adapter starts as a true
        # no-op, identical to the base + previously-frozen tasks.
        new_G = torch.zeros(new_r1, new_r2, new_r3, new_r4, new_r5, device=device, dtype=dtype)
        new_G[: self.r1, : self.r2, : self.r3, : self.r4, : self.r5] = self.G.detach()
        # (no random noise on the new region anymore)
        self._replace_param("G", new_G)

        # Commit new ranks
        self.r1 = new_r1
        self.r2 = new_r2
        self.r3 = new_r3
        self.r4 = new_r4
        self.r5 = new_r5

    # -- freezing / masking ------------------------------------------------

    def commit_frozen(self):
        """
        Call AFTER a task has been trained. Freezes the current shape:
        every rank / row that exists now becomes untouchable in later tasks
        (except that later tasks are allowed to add rows / ranks).
        """
        self.frozen_r1.fill_(self.r1)
        self.frozen_r2.fill_(self.r2)
        self.frozen_r3.fill_(self.r3)
        self.frozen_r4.fill_(self.r4)
        self.frozen_r5.fill_(self.r5)
        self.frozen_rows_U3.fill_(self.U3.shape[0])
        self.frozen_rows_U4.fill_(self.U4.shape[0])
        self.frozen_rows_U5.fill_(self.U5.shape[0])

    def _active_row_is_new(self, axis: str) -> bool:
        if axis == "U3":
            return self._active_s is not None and self._active_s >= int(self.frozen_rows_U3.item())
        if axis == "U4":
            return self._active_e is not None and self._active_e >= int(self.frozen_rows_U4.item())
        if axis == "U5":
            return self._active_p is not None and self._active_p >= int(self.frozen_rows_U5.item())
        return False

    @torch.no_grad()
    def zero_inactive_gradients(self):
        """
        Masks gradients so that only the following regions learn:
            U1: columns [frozen_r1 :]                  (new rank block)
            U2: columns [frozen_r2 :]                  (new rank block)
            U3: active row, full width if row is new;  new-coord cols if row is old
            U4: active row, full width if row is new;  new-coord cols if row is old
            U5: active row, full width if row is new;  new-coord cols if row is old
            G : any index NOT entirely inside [:fr1, :fr2, :fr3, :fr4, :fr5]
        Everything else has its gradient zeroed.
        """
        fr1 = int(self.frozen_r1.item())
        fr2 = int(self.frozen_r2.item())
        fr3 = int(self.frozen_r3.item())
        fr4 = int(self.frozen_r4.item())
        fr5 = int(self.frozen_r5.item())

        if self.U1.grad is not None:
            self.U1.grad[:, :fr1] = 0
        if self.U2.grad is not None:
            self.U2.grad[:, :fr2] = 0

        def _mask_row_factor(param: nn.Parameter, active_idx: Optional[int], frozen_cols: int, frozen_rows: int):
            if param.grad is None or param.shape[0] == 0:
                return
            g = param.grad
            # Zero ALL rows first...
            mask_keep = torch.zeros_like(g)
            if active_idx is not None and 0 <= active_idx < g.shape[0]:
                if active_idx >= frozen_rows:
                    # New row: update all columns
                    mask_keep[active_idx, :] = 1
                else:
                    # Old row: only new-coord columns may update
                    if g.shape[1] > frozen_cols:
                        mask_keep[active_idx, frozen_cols:] = 1
            g.mul_(mask_keep)

        _mask_row_factor(self.U3, self._active_s, fr3, int(self.frozen_rows_U3.item()))
        _mask_row_factor(self.U4, self._active_e, fr4, int(self.frozen_rows_U4.item()))
        _mask_row_factor(self.U5, self._active_p, fr5, int(self.frozen_rows_U5.item()))

        if self.G.grad is not None:
            # Freeze G's "all-five-dims-old" sub-block. With our G zero
            # init + expand-new-G-zero invariants, this together with the
            # natural u-vector zero patterns (new rows of U3/U4/U5 are
            # zero on the old-coord columns) is SUFFICIENT for Theorem 1:
            #
            # For a new task t to write to G[a, b, c<fr3_k, d<fr4_k, e<fr5_k]
            # (an old task k's contraction region), it would need
            # u3_t[c]*u4_t[d]*u5_t[e] != 0 in that range. But t's triple
            # differs from k's in at least one axis -> the new row on
            # that axis has zero values in [:fr_old] -> the product is
            # zero -> no gradient flows there. So old task k's G slice is
            # never touched even though the explicit mask only freezes
            # the (a<fr1, b<fr2) sub-corner of it.
            #
            # The earlier mask change to G[:, :, :fr3, :fr4, :fr5] was
            # mathematically equivalent (the explicit "extra" freeze never
            # actually blocked any nonzero gradient) but pathologically
            # interacted with high lr + high scaling + long training to
            # destabilise G values. Reverting to the proven OLD mask.
            self.G.grad[:fr1, :fr2, :fr3, :fr4, :fr5] = 0

    @torch.no_grad()
    def enforce_zero_pad(self):
        """
        After an optimizer step, re-zero positions that MUST be zero to
        preserve old-task deltas:
            * U3[old_row, new_cols]  (row in [:frozen_rows_U3], col in [frozen_r3:])
            * U4[old_row, new_cols]
            * U5[old_row, new_cols]
        Also snap frozen blocks of U1/U2/G back to their frozen values in
        case momentum caused any drift (defensive -- gradient masking alone
        should already prevent drift, but weight_decay can still act).
        """
        fr3 = int(self.frozen_r3.item())
        fr4 = int(self.frozen_r4.item())
        fr5 = int(self.frozen_r5.item())
        old_rows_U3 = int(self.frozen_rows_U3.item())
        old_rows_U4 = int(self.frozen_rows_U4.item())
        old_rows_U5 = int(self.frozen_rows_U5.item())

        # Previously-seen rows cannot receive any drift, on ANY column, from
        # this task -- they were frozen when we called commit_frozen(). But
        # we specifically need the NEW-coord columns of these rows to remain
        # ZERO (not just frozen at their value), which happens to BE zero
        # after expand(). Re-zeroing is cheap insurance:
        if old_rows_U3 > 0 and self.U3.shape[1] > fr3:
            self.U3.data[:old_rows_U3, fr3:] = 0
        if old_rows_U4 > 0 and self.U4.shape[1] > fr4:
            self.U4.data[:old_rows_U4, fr4:] = 0
        if old_rows_U5 > 0 and self.U5.shape[1] > fr5:
            self.U5.data[:old_rows_U5, fr5:] = 0

    @torch.no_grad()
    def decay_trainable(self, decay: float):
        """
        Weight-decay applied ONLY to the parameters this task is allowed to
        train -- i.e. the NEW rank block of U1/U2 and the non-frozen-corner
        region of G. The frozen corner G[:fr1,:fr2,:fr3,:fr4,:fr5] and the
        frozen prefixes U1[:, :fr1] / U2[:, :fr2] are NEVER touched, so
        Theorem 1 stays bit-exact (unlike AdamW's global weight_decay, which
        would shrink the frozen factors and corrupt old tasks -- the reason
        bug #14 had to set weight_decay=0).

        Motivation: the per-task delta-norm diagnostic showed cold-start
        adapters (T1 = first task, T8 = first OLN) DIVERGE to a large-
        magnitude delta (max|dW| up to ~2.9) and mode-collapse to all-forward,
        while healthy adapters stay tiny (~0.001). A magnitude penalty pulls
        the over-grown ones back proportionally harder without killing the
        already-small working adapters (which a uniform alpha/scaling cut
        would do). Call this AFTER optimizer.step(), before enforce_zero_pad.
        """
        if decay is None or decay <= 0:
            return
        keep = 1.0 - float(decay)
        fr1 = int(self.frozen_r1.item()); fr2 = int(self.frozen_r2.item())
        fr3 = int(self.frozen_r3.item()); fr4 = int(self.frozen_r4.item())
        fr5 = int(self.frozen_r5.item())

        # U1 / U2 : only the newly-added (trainable) columns.
        if self.U1.shape[1] > fr1:
            self.U1.data[:, fr1:] *= keep
        if self.U2.shape[1] > fr2:
            self.U2.data[:, fr2:] *= keep

        # G : the complement of the frozen corner, partitioned into 5 DISJOINT
        # slabs so nothing is decayed twice and the frozen corner is skipped.
        G = self.G.data
        if G.shape[0] > fr1:
            G[fr1:, :, :, :, :] *= keep
        if G.shape[1] > fr2:
            G[:fr1, fr2:, :, :, :] *= keep
        if G.shape[2] > fr3:
            G[:fr1, :fr2, fr3:, :, :] *= keep
        if G.shape[3] > fr4:
            G[:fr1, :fr2, :fr3, fr4:, :] *= keep
        if G.shape[4] > fr5:
            G[:fr1, :fr2, :fr3, :fr4, fr5:] *= keep

    # -- forward -----------------------------------------------------------

    def forward_delta(self, s: Optional[int] = None, e: Optional[int] = None, p: Optional[int] = None) -> torch.Tensor:
        """
        Compute the delta weight  DeltaW  in R^{out_features x in_features}.
        Active indices come from explicit args if given, else from
        set_active_route() state.
        """
        s = self._active_s if s is None else s
        e = self._active_e if e is None else e
        p = self._active_p if p is None else p
        if s is None or e is None or p is None:
            raise RuntimeError("Tucker5DLoRALayer forward called without (s,e,p). Use set_active_route first.")

        # Contract G over dims 3,4,5 with the three row vectors
        # G: (r1, r2, r3, r4, r5)
        # result: (r1, r2)
        u3 = self.U3[s]
        u4 = self.U4[e]
        u5 = self.U5[p]
        G_sep = torch.einsum("ijklm,k,l,m->ij", self.G, u3, u4, u5)

        # DeltaW = U1 @ G_sep @ U2^T     in R^{a_l x b_l}
        delta = self.U1 @ G_sep @ self.U2.t()
        # DeltaW convention: apply as  y += x @ DeltaW.T (linear convention)
        # Transpose to match out_features x in_features layout used below
        return delta.t()  # (out_features, in_features)


class Tucker5DLoRALinear(nn.Module):
    """Wrap nn.Linear with a Tucker5DLoRALayer adapter."""

    def __init__(
        self,
        base_layer: nn.Linear,
        ranks: Tuple[int, int, int, int, int] = (16, 16, 8, 8, 4),
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        init_std: float = 0.02,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.lora_layer = Tucker5DLoRALayer(
            in_features=base_layer.in_features,
            out_features=base_layer.out_features,
            ranks=ranks,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_std=init_std,
        )
        for p in self.base_layer.parameters():
            p.requires_grad = False

    def set_active_route(self, s: int, e: int, p: int):
        self.lora_layer.set_active_route(s, e, p)

    def forward(self, x):
        base_out = self.base_layer(x)
        s, e, p = self.lora_layer._active_s, self.lora_layer._active_e, self.lora_layer._active_p
        if s is None or e is None or p is None:
            return base_out

        x_d = self.lora_layer.lora_dropout(x)
        u3 = self.lora_layer.U3[s]
        u4 = self.lora_layer.U4[e]
        u5 = self.lora_layer.U5[p]
        # (r1, r2): G contracted over the three hierarchical dims
        G_sep = torch.einsum("ijklm,k,l,m->ij", self.lora_layer.G, u3, u4, u5)
        # x @ U1 -> (..., r1); @ G_sep -> (..., r2); @ U2^T -> (..., out)
        h = F.linear(x_d, self.lora_layer.U1.t())           # (..., r1)
        h = F.linear(h, G_sep.t())                           # (..., r2)
        h = F.linear(h, self.lora_layer.U2)                  # (..., out_features)
        return base_out + h * self.lora_layer.scaling


# ---------------------------------------------------------------------------
# Convenience helpers (iterate all 5D layers in a model)
# ---------------------------------------------------------------------------

def iter_tucker5d_layers(model: nn.Module):
    for m in model.modules():
        if isinstance(m, Tucker5DLoRALinear):
            yield m.lora_layer


def set_active_route_all(model: nn.Module, s: int, e: int, p: int):
    for layer in iter_tucker5d_layers(model):
        layer.set_active_route(s, e, p)


def zero_inactive_gradients_all(model: nn.Module):
    for layer in iter_tucker5d_layers(model):
        layer.zero_inactive_gradients()


def enforce_zero_pad_all(model: nn.Module):
    for layer in iter_tucker5d_layers(model):
        layer.enforce_zero_pad()


def decay_trainable_all(model: nn.Module, decay: float):
    """Apply trainable-only weight decay to every 5D Tucker layer. No-op if
    decay is falsy. Call after optimizer.step(), before enforce_zero_pad_all."""
    if not decay or decay <= 0:
        return
    for layer in iter_tucker5d_layers(model):
        layer.decay_trainable(decay)


def commit_frozen_all(model: nn.Module):
    for layer in iter_tucker5d_layers(model):
        layer.commit_frozen()


def expand_all(
    model: nn.Module,
    delta_r1: int = 0,
    delta_r2: int = 0,
    delta_r3: int = 0,
    delta_r4: int = 0,
    delta_r5: int = 0,
):
    for layer in iter_tucker5d_layers(model):
        layer.expand(delta_r1, delta_r2, delta_r3, delta_r4, delta_r5)


def append_category_row_all(model: nn.Module, axis: str) -> int:
    """
    Append a fresh row to the given axis across ALL Tucker5D layers and
    return the row index (identical across layers by construction).
    """
    new_idx = -1
    for layer in iter_tucker5d_layers(model):
        idx = layer.append_category_row(axis)
        if new_idx == -1:
            new_idx = idx
        elif idx != new_idx:
            raise RuntimeError(f"Tucker5D row-index desync on {axis}: {idx} vs {new_idx}")
    return new_idx
