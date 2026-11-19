"""
Spatio-temporal motion transformer for 3D joint reconstruction (past) and
forecasting (future) from per-frame features, with optional AuxFormer-style
auxiliary tasks (WHAM features only).

Design choices
--------------
- SimpliHuMoN style: past tokens and future learnable queries are concatenated
  into one unified sequence and processed together — no encoder/decoder split,
  no causal mask. Future queries are pure positional slots (no information to
  leak), so full bidirectional attention is fine and lets future steps
  coordinate globally for a coherent trajectory.

- AuxFormer STTrans style: instead of running full attention over all
  (N_past + N_future) * K tokens at once — cost ((15+25)*17)^2 ≈ 462,400 —
  we split into:
    * Temporal attention: (N_past+N_future)^2 per joint = 1,600 per joint
    * Spatial attention:  K^2 per timestep = 289 per timestep
  This is ~17x cheaper and still captures how each joint moves over time
  (temporal) and how joints relate to each other at each frame (spatial).

Input modes
-----------
wham=False (default):
    phi is a generic per-frame feature (e.g. HMR2 pooled, ResNet).
    It is projected to d_model and read out into K per-joint tokens.
    Future tokens = last past token + learnable future queries.
wham=True:
    phi is a WHAM feature = [context (feat_dim - K*3) | joint coords (K*3)].
    Joint coords are made root-relative and embedded per joint by a shared MLP;
    the context vector is projected and broadcast onto every joint.
wham=True, joints_only=True:
    Two-stage baseline: only the WHAM joint coords are used, the context
    vector is ignored (no context_proj is built). This is "estimate the poses,
    then forecast from the skeleton", with the same forecaster.
wham=True, context_only=True:
    Image-only ablation: the WHAM joint coords are dropped and only the
    context vector is used. It is projected exactly like the generic mode
    (Linear -> d_model, then a Linear readout into K per-joint tokens).
    Not compatible with aux tasks (there are no joint tokens to mask).

Auxiliary tasks (aux=True, requires wham=True)
----------------------------------------------
Training only. Each task runs on its own corrupted copy of the input and
shares all transformer weights with the main task (AuxFormer).
  1. Joint masking : random (t, k) joint tokens -> learnable [MASK] token.
                     The frame's image context is kept.
  2. Frame masking : ALL joint tokens of random past frames -> [MASK] token.
                     By default the frame's image context is KEPT, so the
                     model recovers the pose from image evidence (a per-frame
                     task). With context_mask the context of those frames is
                     replaced by a learnable [CTX_MASK] token too, so the pose
                     can ONLY come from the temporal neighbours, i.e. the model
                     has to learn motion dynamics (the skill the future needs).

All aux pose targets are the GROUND TRUTH 3D joints, never the WHAM input.
At test time call model(phi) with no corruption arguments.

Input
-----
phi: (B, N_past, feat_dim) 
Output
------
past_recon:  (B, N_past,   K, 3) — reconstructed 3D joints for observed frames
future_pred: (B, N_future, K, 3) — predicted 3D joints for future frames
"""

import torch
import torch.nn as nn


# ─── spatio-temporal block ───────────────────────────────────────────────────

class STBlock(nn.Module):
    """
    Runs temporal attention (each joint independently across time) followed
    by spatial attention (each timestep independently across joints), each
    with its own pre-norm + residual + MLP.

    Args:
        d_model: token dimension.
        n_heads: number of attention heads.
        dim_feedforward: hidden size of the MLP sublayer.
        dropout: dropout probability.
    """

    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int, dropout: float):
        super().__init__()

        # temporal sublayer
        self.norm_t1 = nn.LayerNorm(d_model)
        self.temp_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_t2 = nn.LayerNorm(d_model)
        self.mlp_t = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

        # spatial sublayer
        self.norm_s1 = nn.LayerNorm(d_model)
        self.spat_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_s2 = nn.LayerNorm(d_model)
        self.mlp_s = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, K, d_model)  T = N_past + N_future, K = num_joints
        Returns:
            x: (B, T, K, d_model)
        """
        B, T, K, C = x.shape

        # ── temporal attention: each joint attends across all T timesteps ──
        # reshape so the "sequence" dimension is T, batch is B*K
        xt = x.permute(0, 2, 1, 3).reshape(B * K, T, C)   # (B*K, T, C)
        xt_n = self.norm_t1(xt)
        xt_a, _ = self.temp_attn(xt_n, xt_n, xt_n)         # full, no mask
        xt = xt + xt_a
        xt = xt + self.mlp_t(self.norm_t2(xt))
        x = xt.reshape(B, K, T, C).permute(0, 2, 1, 3)     # back to (B, T, K, C)

        # ── spatial attention: each timestep attends across all K joints ──
        # reshape so the "sequence" dimension is K, batch is B*T
        xs = x.reshape(B * T, K, C)                         # (B*T, K, C)
        xs_n = self.norm_s1(xs)
        xs_a, _ = self.spat_attn(xs_n, xs_n, xs_n)         # full, no mask
        xs = xs + xs_a
        xs = xs + self.mlp_s(self.norm_s2(xs))
        x = xs.reshape(B, T, K, C)

        return x


# ─── full model ──────────────────────────────────────────────────────────────

class MotionTransformerST(nn.Module):
    """Spatio-temporal motion transformer.

    Args:
        feat_dim: input feature dimension (1280 for HMR2, 2048 for ResNet).
        d_model: internal token dimension.
        n_heads: attention heads (must divide d_model).
        num_layers: number of STBlock layers.
        dim_feedforward: MLP hidden size inside each block.
        dropout: dropout probability.
        num_joints: K, number of body joints (17 for H36M).
        n_past: number of observed past frames.
        n_future: number of future frames to predict.
        wham: if True, treat phi as a WHAM feature = [context | joint coords] instead of a generic per-frame feature.
        aux: if True, enable auxiliary tasks (joint masking, frame masking) for WHAM features.
        aux_mask_context: if True (and aux), also create a [CTX_MASK] token so
            frame masking can hide the image context of masked frames.
        joints_only: if True (requires wham), drop the WHAM context and build
            the tokens from the estimated joint coords only (two-stage baseline).
        context_only: if True (requires wham), drop the WHAM joint coords and
            build the tokens from the context vector only, with the same
            linear projection + per-joint readout as the non-WHAM mode.
    """

    def __init__(
        self,
        feat_dim: int = 1024,
        d_model: int = 256,
        n_heads: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        num_joints: int = 17,
        n_past: int = 15,
        n_future: int = 25,
        wham: bool = False,
        aux: bool = False,
        aux_mask_context: bool = False,
        joints_only: bool = False,
        context_only: bool = False,
    ):
        super().__init__()
        self.n_past = n_past
        self.n_future = n_future
        self.num_joints = num_joints
        self.wham = wham
        self.aux = aux
        self.aux_mask_context = aux and aux_mask_context
        self.d_model = d_model

        if aux and not wham:
            raise ValueError("Auxiliary tasks require wham=True")
        if joints_only and not wham:
            raise ValueError("joints_only requires wham=True")
        if joints_only and self.aux_mask_context:
            raise ValueError("joints_only has no context to mask: drop aux_mask_context")
        if context_only and not wham:
            raise ValueError("context_only requires wham=True")
        if context_only and joints_only:
            raise ValueError("context_only and joints_only are mutually exclusive")
        if context_only and aux:
            raise ValueError("context_only has no joint tokens to mask: drop aux")
        self.joints_only = joints_only
        self.context_only = context_only

        # ── input projection: per-frame feature -> d_model ──────────────────
        if wham and context_only:
            # WHAM context only, projected like the generic (non-WHAM) mode
            self.coord_dim = num_joints * 3
            self.context_dim = feat_dim - self.coord_dim
            self.input_proj = nn.Linear(self.context_dim, d_model)
            self.joint_proj = nn.Linear(d_model, num_joints * d_model)
        elif wham:
            self.coord_dim = num_joints * 3
            self.context_dim = feat_dim - self.coord_dim # 563 - 51 = 512
            self.joint_proj = nn.Sequential(
                nn.Linear(3, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
            )
            # not built with joints_only, so DDP has no unused parameters
            if not joints_only:
                self.context_proj = nn.Sequential(
                    nn.LayerNorm(self.context_dim),
                    nn.Linear(self.context_dim, d_model),
                )
        else:
            self.input_proj = nn.Linear(feat_dim, d_model)
            self.joint_proj = nn.Linear(d_model, num_joints * d_model)

        # -- learnable [MASK] tokens for the joints-masking / frame-masking tasks.
        if aux:
            self.joint_mask_token = nn.Parameter(torch.zeros(d_model))
            nn.init.trunc_normal_(self.joint_mask_token, std=0.02)
        if self.aux_mask_context:
            self.context_mask_token = nn.Parameter(torch.zeros(d_model))
            nn.init.trunc_normal_(self.context_mask_token, std=0.02)

        # ── positional encodings (learned, shared past+future) ───────────────
        # temporal: one vector per time slot (past + future)
        self.time_enc = nn.Parameter(torch.zeros(n_past + n_future, d_model))
        nn.init.trunc_normal_(self.time_enc, std=0.02)

        # spatial: one vector per joint
        self.joint_enc = nn.Parameter(torch.zeros(num_joints, d_model))
        nn.init.trunc_normal_(self.joint_enc, std=0.02)

        # ── learnable future query tokens: one per (future_step, joint) ──────
        self.future_queries = nn.Parameter(torch.zeros(n_future, num_joints, d_model))
        nn.init.trunc_normal_(self.future_queries, std=0.02)

        # ── ST transformer blocks ────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            STBlock(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        # ── shared pose head: d_model -> 3 joints coords ─────────────────────
        self.pose_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 3),
        )

    def _embed_generic(self, phi: torch.Tensor) -> torch.Tensor:
        """Generic feature -> (B, N_past, K, d_model)."""
        B = phi.shape[0]
        past_feat = self.input_proj(phi)                       # (B, N_past, d_model)
        return self.joint_proj(past_feat).view(
            B, self.n_past, self.num_joints, self.d_model
        )
 
    def _embed_wham(self, phi: torch.Tensor, joint_mask: torch.Tensor | None = None, frame_mask: torch.Tensor | None = None,
                    context_mask: torch.Tensor | None = None) -> torch.Tensor:
        """WHAM feature [context | coords] -> (B, N_past, K, d_model).
 
        Corruptions (aux tasks) only ever touch the JOINT tokens; the image
        context is always left intact.
        """
        B = phi.shape[0]
        context = phi[..., :self.context_dim]                   # (B, N_past, context_dim)
        coords = phi[..., self.context_dim:]                    # (B, N_past, K*3)
        coords = coords.reshape(B, self.n_past, self.num_joints, 3)
        coords = coords - coords[..., :1, :]                    # root-relative
 
        joint_tokens = self.joint_proj(coords)                   # (B, N_past, K, d_model)

        # masking tasks: frame masking = mask every joint of the frame,
        # but keep that frame's context token.
        if frame_mask is not None:
            fm = frame_mask[..., None].expand(-1, -1, self.num_joints)  # (B, N_past, K)
            joint_mask = fm if joint_mask is None else (joint_mask | fm)
        if joint_mask is not None:
            joint_tokens = torch.where(
                joint_mask[..., None], self.joint_mask_token, joint_tokens
            )
 
        # two-stage baseline: tokens come from the estimated joints only
        if self.joints_only:
            return joint_tokens                                  # (B, N_past, K, d_model)

        context_tokens = self.context_proj(context)              # (B, N_past, d_model)

        # context masking (B, N_past): hide the image evidence of those frames
        if context_mask is not None:
            context_tokens = torch.where(
                context_mask[..., None], self.context_mask_token, context_tokens
            )

        return joint_tokens + context_tokens.unsqueeze(2)       # (B, N_past, K, d_model)

    def forward(self, phi: torch.Tensor, joint_mask: torch.Tensor | None = None, frame_mask: torch.Tensor | None = None,
                context_mask: torch.Tensor | None = None):
        """
        Args:
            phi: (B, n_past, feat_dim)
            joint_mask / frame_mask : aux-task corruptions (only allowed when aux=True; leave as None for the main task and a test time)

        Returns:
            past_recon:  (B, n_past,   K, 3)
            future_pred: (B, n_future, K, 3)
        """
        B = phi.shape[0]

        corrupted = any(c is not None for c in (joint_mask, frame_mask, context_mask))
        if context_mask is not None and not self.aux_mask_context:
            raise ValueError("context_mask passed but the model was built with "
                             "aux_mask_context=False.")
        if corrupted and not self.aux:
            raise ValueError("Corruption arguments were passed but the model was "
                             "built with aux=False.")

        # 1. project per-frame features into K distinct per-joint tokens
        #    each joint gets its own learned readout of the frame feature,
        #    so joints start differentiated instead of relying on spatial
        #    attention + joint_enc to break the symmetry of a copied feature
        if self.wham and self.context_only:
            # drop the joint coords, keep only the image context
            past_tokens = self._embed_generic(phi[..., :self.context_dim])
        elif self.wham:
            past_tokens = self._embed_wham(phi, joint_mask, frame_mask, context_mask)                  # (B, N_past, K, d_model)
        else:
            past_tokens = self._embed_generic(phi)               # (B, N_past, K, d_model)

        # 2. expand learnable future queries to the batch dimension
        last_token = past_tokens[:, -1:]  # (B, 1, K, d_model)
        future_tokens = (last_token.expand(-1, self.n_future, -1, -1)
                     + self.future_queries.unsqueeze(0)) 
        # (B, N_future, K, d_model)

        # 3. concatenate past and future along the time dimension
        x = torch.cat([past_tokens, future_tokens], dim=1)
        # (B, N_past + N_future, K, d_model)

        # 4. add positional encodings
        #    temporal: (T, d_model) broadcast over B and K
        x = x + self.time_enc.unsqueeze(0).unsqueeze(2)
        #    spatial:  (K, d_model) broadcast over B and T
        x = x + self.joint_enc.unsqueeze(0).unsqueeze(0)

        # 5. spatio-temporal transformer blocks (bidirectional, no mask)
        for block in self.blocks:
            x = block(x)
        # x: (B, T, K, d_model)

        # 6. pose head -> 3D coordinates for every (time, joint) slot
        joints = self.pose_head(x)          # (B, T, K, 3)

        past_recon  = joints[:, :self.n_past]
        future_pred = joints[:, self.n_past:]

        return past_recon, future_pred


 
# ─── corruption helpers (aux tasks) ──────────────────────────────────────────
 
def random_joint_mask(B, T, K, ratio, device, keep_last=True):
    """Mask `ratio` of (t, k) joints; never masks the root (it's always 0).
    With keep_last=True the last observed frame is never masked, because the
    future tokens are initialised by copying it."""
    m = torch.rand(B, T, K, device=device) < ratio
    m[..., 0] = False
    if keep_last:
        m[:, -1] = False
    return m
 
 
def random_frame_mask(B, T, ratio, device, keep_last=True):
    """Mask the joints of `ratio` of past frames (context is kept). With
    keep_last=True the last observed frame is never masked, because the
    future tokens are initialised by copying it."""
    m = torch.rand(B, T, device=device) < ratio
    if keep_last:
        m[:, -1] = False
    return m
 

def random_span_frame_mask(B, T, ratio, device, keep_last=True):
    """Mask ONE contiguous block of past frames per sample (length =
    round(ratio * usable frames), random start). Unlike i.i.d. frame masking,
    the model cannot just average the two neighbours of an isolated hole: it
    has to carry the motion across a gap, which is much closer to forecasting.
    `ratio` may be a float or a (B,) tensor."""
    usable = T - 1 if keep_last else T
    r = torch.as_tensor(ratio, device=device, dtype=torch.float).expand(B)
    L = (r * usable).round().clamp(1, usable).long()                     # (B,)
    start = (torch.rand(B, device=device) * (usable - L + 1).float()).floor().long()
    t = torch.arange(T, device=device)[None]                             # (1, T)
    return (t >= start[:, None]) & (t < (start + L)[:, None])            # (B, T)