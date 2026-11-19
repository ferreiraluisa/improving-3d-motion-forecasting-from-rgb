from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# MANUAL MULTI-HEAD ATTENTION HELPERS (for KV-cached decoding)
#
# These reach into an existing nn.MultiheadAttention module's own
# parameters (in_proj_weight/in_proj_bias/out_proj) and call them by hand
# via F.linear + F.scaled_dot_product_attention, instead of going through
# nn.MultiheadAttention.forward(). This lets the autoregressive rollout
# cache per-layer K/V tensors across steps (so old timesteps are never
# reprocessed) while keeping parameter names/shapes byte-for-byte
# identical to the original module -- existing checkpoints still load.
# ============================================================

def _split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    # (B, L, E) -> (B, num_heads, L, head_dim)
    B, L, E = x.shape
    return x.view(B, L, num_heads, E // num_heads).transpose(1, 2)


def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    # (B, num_heads, L, head_dim) -> (B, L, E)
    B, H, L, D = x.shape
    return x.transpose(1, 2).reshape(B, L, H * D)


def _mha_q_proj(mha: nn.MultiheadAttention, q_in: torch.Tensor) -> torch.Tensor:
    E = mha.embed_dim
    w = mha.in_proj_weight[:E]
    b = mha.in_proj_bias[:E] if mha.in_proj_bias is not None else None
    return F.linear(q_in, w, b)


def _mha_kv_proj(mha: nn.MultiheadAttention, k_in: torch.Tensor, v_in: torch.Tensor):
    E = mha.embed_dim
    w = mha.in_proj_weight
    b = mha.in_proj_bias
    wk, wv = w[E:2 * E], w[2 * E:3 * E]
    bk, bv = (b[E:2 * E], b[2 * E:3 * E]) if b is not None else (None, None)
    k = F.linear(k_in, wk, bk)
    v = F.linear(v_in, wv, bv)
    return k, v


def _mha_attend_cached(
    mha: nn.MultiheadAttention,
    q_in: torch.Tensor,
    new_k_in: Optional[torch.Tensor],
    new_v_in: Optional[torch.Tensor],
    kv_cache: Optional[tuple],
    training: bool,
):
    """
    Single-step attention using (and extending) a cached K/V tensor pair.

    q_in: (B, Lq, E) query input for just the new step(s)
    new_k_in/new_v_in: (B, Lnew, E) raw inputs to project into K/V for the
        new step and append to the cache. Pass None to reuse kv_cache as-is
        (used for cross-attention against a fixed, already-cached memory).
    kv_cache: None on the first call, else the (k, v) tuple returned by a
        previous call -- already head-split tensors of shape
        (B, num_heads, L_so_far, head_dim).

    Returns (attn_output, updated_kv_cache).
    """
    q = _split_heads(_mha_q_proj(mha, q_in), mha.num_heads)

    if new_k_in is not None:
        k_new, v_new = _mha_kv_proj(mha, new_k_in, new_v_in)
        k_new = _split_heads(k_new, mha.num_heads)
        v_new = _split_heads(v_new, mha.num_heads)
        if kv_cache is None:
            k, v = k_new, v_new
        else:
            k = torch.cat([kv_cache[0], k_new], dim=2)
            v = torch.cat([kv_cache[1], v_new], dim=2)
    else:
        # cross-attention against memory: cache is already complete.
        assert kv_cache is not None
        k, v = kv_cache

    dropout_p = mha.dropout if training else 0.0
    attn = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
    attn = mha.out_proj(_merge_heads(attn))

    return attn, (k, v)


# ============================================================
# SPATIO-TEMPORAL ENCODER BLOCK
# ============================================================

class STBlock(nn.Module):

    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int, dropout: float):
        super().__init__()

        # temporal attention: each joint attends over time
        self.norm_t1 = nn.LayerNorm(d_model)
        self.temp_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.drop_t = nn.Dropout(dropout)

        self.norm_t2 = nn.LayerNorm(d_model)
        self.mlp_t = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

        # spatial attention: each timestep attends over joints
        self.norm_s1 = nn.LayerNorm(d_model)
        self.spat_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.drop_s = nn.Dropout(dropout)

        self.norm_s2 = nn.LayerNorm(d_model)
        self.mlp_s = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, K, C = x.shape

        # ---- temporal: (B,T,K,C) -> (B*K,T,C) ----
        xt = x.permute(0, 2, 1, 3).reshape(B * K, T, C)

        xt_norm = self.norm_t1(xt)
        xt_attn, _ = self.temp_attn(xt_norm, xt_norm, xt_norm, need_weights=False)
        xt = xt + self.drop_t(xt_attn)
        xt = xt + self.mlp_t(self.norm_t2(xt))

        x = xt.reshape(B, K, T, C).permute(0, 2, 1, 3)

        # ---- spatial: (B,T,K,C) -> (B*T,K,C) ----
        xs = x.reshape(B * T, K, C)

        xs_norm = self.norm_s1(xs)
        xs_attn, _ = self.spat_attn(xs_norm, xs_norm, xs_norm, need_weights=False)
        xs = xs + self.drop_s(xs_attn)
        xs = xs + self.mlp_s(self.norm_s2(xs))

        return xs.reshape(B, T, K, C)


# ============================================================
# SPATIO-TEMPORAL ENCODER
# ============================================================

class SpatialTemporalEncoder(nn.Module):

    def __init__(self, feat_dim: int, d_model: int, n_heads: int, num_layers: int,
                 dim_feedforward: int, dropout: float, num_joints: int, max_len: int):
        super().__init__()

        self.max_len = max_len
        self.num_joints = num_joints
        self.d_model = d_model

        # input projection: per-frame feature -> K joint tokens
        self.input_proj = nn.Linear(feat_dim, d_model * num_joints)
        self.input_drop = nn.Dropout(dropout)

        # temporal positional encoding
        self.time_enc = nn.Parameter(torch.zeros(max_len, d_model))
        nn.init.trunc_normal_(self.time_enc, std=0.02)

        # joint positional encoding
        self.joint_enc = nn.Parameter(torch.zeros(num_joints, d_model))
        nn.init.trunc_normal_(self.joint_enc, std=0.02)

        self.blocks = nn.ModuleList([
            STBlock(
                d_model=d_model,
                n_heads=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        """
        phi: (B, T, feat_dim)  ->  memory: (B, T, K, C)
        """
        B, T, _ = phi.shape
        K = self.num_joints

        if T > self.max_len:
            raise ValueError(f"Input sequence length {T} exceeds max_len {self.max_len}")

        x = self.input_proj(phi).view(B, T, K, self.d_model)

        # positional encodings (sliced so shorter clips still work)
        x = x + self.time_enc[:T].view(1, T, 1, self.d_model)
        x = x + self.joint_enc.view(1, 1, K, self.d_model)
        x = self.input_drop(x)

        for block in self.blocks:
            x = block(x)

        return self.final_norm(x)


# ============================================================
# AUTOREGRESSIVE TRANSFORMER DECODER LAYER
# ============================================================

class ARDecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int, dropout: float):
        super().__init__()

        self.norm_self = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.drop_self = nn.Dropout(dropout)

        # cross attention: queries read the encoder memory
        self.norm_cross = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.drop_cross = nn.Dropout(dropout)

        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, memory: torch.Tensor, causal_mask: torch.Tensor):
        # causal self attention over future query tokens
        q = self.norm_self(x)
        attn, _ = self.self_attn(q, q, q, attn_mask=causal_mask, need_weights=False)
        x = x + self.drop_self(attn)

        # cross attention: future queries read all past memory tokens
        q = self.norm_cross(x)
        attn, _ = self.cross_attn(q, memory, memory, need_weights=False)
        x = x + self.drop_cross(attn)

        x = x + self.ffn(self.norm_ffn(x))
        return x

    def forward_step(
        self,
        x_new: torch.Tensor,
        self_kv_cache: Optional[tuple],
        cross_kv_cache: tuple,
        training: bool,
    ):
        """
        Incremental (KV-cached) counterpart of forward() for a single new
        timestep block. x_new: (B, K, C) raw hidden state for just the new
        timestep -- not yet layer-normed.

        Because the rollout only ever appends one new timestep at a time and
        each new query attends only to itself plus already-processed
        (earlier) timesteps, this is automatically causal: no mask needed.

        cross_kv_cache is precomputed once from the fixed encoder memory
        before the rollout starts (memory never changes across steps).

        Returns (updated x_new, updated self_kv_cache).
        """
        # ---- self attention against cache of all previous timesteps ----
        q_in = self.norm_self(x_new)
        attn, self_kv_cache = _mha_attend_cached(
            self.self_attn, q_in, q_in, q_in, self_kv_cache, training
        )
        x_new = x_new + self.drop_self(attn)

        # ---- cross attention against cached (fixed) memory K/V ----
        q_in = self.norm_cross(x_new)
        attn, _ = _mha_attend_cached(
            self.cross_attn, q_in, None, None, cross_kv_cache, training
        )
        x_new = x_new + self.drop_cross(attn)

        x_new = x_new + self.ffn(self.norm_ffn(x_new))

        return x_new, self_kv_cache


# ============================================================
# FULL MODEL
# ============================================================

class MotionTransformerSTAR(nn.Module):
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
        use_kv_cache: bool = True,
    ):
        super().__init__()

        self.n_past = n_past
        self.n_future = n_future
        self.num_joints = num_joints
        self.d_model = d_model
        # KV-cached rollout is mathematically equivalent to the naive
        # recompute-everything rollout but avoids O(steps^2) redundant work.
        # Set to False to fall back to the original behavior (e.g. to
        # numerically sanity-check a change).
        self.use_kv_cache = use_kv_cache

        self.encoder = SpatialTemporalEncoder(
            feat_dim=feat_dim,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_joints=num_joints,
            max_len=n_past + n_future,
        )

        # future positional enconding, one per future timestep
        self.future_time_enc = nn.Parameter(torch.zeros(n_future, d_model))
        nn.init.trunc_normal_(self.future_time_enc, std=0.02)

        self.dec_joint_enc = nn.Parameter(torch.zeros(num_joints, d_model))
        nn.init.trunc_normal_(self.dec_joint_enc, std=0.02)

        self.decoder_layers = nn.ModuleList([
            ARDecoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.decoder_norm = nn.LayerNorm(d_model)

        self.query_drop = nn.Dropout(dropout)

        # SHARED 3D regressor: consumes encoder memory AND decoder output
        self.pose_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 3),
        )


    # ---------------- helpers ----------------

    def _causal_mask(self, t_steps: int, k: int, device) -> torch.Tensor:
        """
        Causality is over FUTURE TIMESTEPS, not individual joint tokens.
        Tokens are laid out as t*K + k, so each (T,T) entry is expanded to a
        (K,K) block. Joints within the same frame may attend to each other.
        """
        time_mask = torch.triu(
            torch.ones(t_steps, t_steps, device=device, dtype=torch.bool), diagonal=1
        )
        return time_mask.repeat_interleave(k, dim=0).repeat_interleave(k, dim=1)

    def _embed_queries(self, latent: torch.Tensor, t0: int = 0) -> torch.Tensor:
        """
        latent: (B, T, K, C) previous latent representations -> query tokens (B, T, K, C)
        t0: index of the first timestep, so partial rollouts get the right
            temporal encoding.
        """
        B, T, K, C = latent.shape
        if t0 + T > self.n_future:
            raise ValueError(f"Decoder input exceeds n_future={self.n_future}")
        x = latent

        x = x + self.future_time_enc[t0:t0 + T].view(1, T, 1, C)
        x = x + self.dec_joint_enc.view(1, 1, K, C)

        return self.query_drop(x)

    def decode(self,  memory: torch.Tensor, decoder_inputs: torch.Tensor):
        """
        memory:         (B, N_past, K, C)
        decoder_inputs: (B, T_future, K, C)
        """
        B, T, K, C = decoder_inputs.shape

        x = decoder_inputs.reshape(B, T * K, C)
        mem = memory.reshape(B, -1, C)          # (B, N_past*K, C)

        mask = self._causal_mask(T, K, x.device)

        for layer in self.decoder_layers:
            x = layer(x, mem, causal_mask=mask)
        x = self.decoder_norm(x)

        return x.reshape(B, T, K, C)

    # ---------------- rollout ---------------
    def _rollout(self, memory: torch.Tensor, start_latent: torch.Tensor, steps: int):
        if self.use_kv_cache:
            return self._rollout_cached(memory, start_latent, steps)
        return self._rollout_naive(memory, start_latent, steps)

    def _rollout_naive(
        self,
        memory: torch.Tensor,
        start_latent: torch.Tensor,
        steps: int,
    ):
        """
        Free-running autoregressive generation. Each step re-runs the decoder
        over the prefix (no KV cache with nn.MultiheadAttention) but only the
        last frame is kept.

        This recomputes attention AND the feed-forward/projection layers for
        every earlier timestep at every step, so total work scales as
        O(steps^2) instead of the O(steps) that a real autoregressive decode
        needs. Kept around as a slow-but-obviously-correct reference -- see
        _rollout_cached for the equivalent, much cheaper implementation.
        """
        inputs = [start_latent]  # list of (B, K, C) tensors
        predictions = []
        for step in range(steps):
            """
            build the complete prefix of previous predictions, embed them, and run the decoder
            step 0: inputs = [start_latent] -> prefix = (B, 1, K, C)
            step 1: inputs = [start_latent, pred_0] -> prefix = (B, 2, K, C)
            ...
            """
            prefix = torch.stack(inputs, dim=1)  # (B, t, K, C)
            tokens = self._embed_queries(prefix, t0=0)
            hidden_full = self.decode(memory, tokens)  # (B, t, K, C)
            next_latent = hidden_full[:, -1]  # (B, K, C)
            predictions.append(next_latent)
            #feed the next latent into the next step
            inputs.append(next_latent)

        predicted = torch.stack(predictions, dim=1)  # (B, steps, K, C)

        return predicted

    def _rollout_cached(
        self,
        memory: torch.Tensor,
        start_latent: torch.Tensor,
        steps: int,
    ):
        """
        Equivalent to _rollout_naive but incremental: each decoder layer
        keeps a self-attention K/V cache that only ever grows by one
        timestep block per step (instead of recomputing every previous
        block's attention + FFN from scratch every step), and the
        cross-attention K/V for the fixed encoder `memory` is projected once
        up front and reused for every step/layer.

        This turns the non-attention (linear/FFN/layernorm) work from
        O(steps^2) down to O(steps), and avoids redundant re-projection of
        `memory` at every layer of every step. The O(steps^2) cost of the
        attention score matrix itself is inherent to full causal attention
        and isn't eliminated here, but it's now spent on fused, well-shaped
        scaled_dot_product_attention calls instead of on duplicated
        projections and MLPs.

        Uses the exact same nn.MultiheadAttention parameters as the naive
        path (see _mha_attend_cached), so checkpoints trained with either
        rollout implementation load into either one without changes.
        """
        B, Np, K, C = memory.shape
        memory_flat = memory.reshape(B, Np * K, C)

        # Cross-attention K/V from `memory` never changes across rollout
        # steps, so project it once per layer instead of once per step.
        cross_kv_cache = []
        for layer in self.decoder_layers:
            k_mem, v_mem = _mha_kv_proj(layer.cross_attn, memory_flat, memory_flat)
            k_mem = _split_heads(k_mem, layer.cross_attn.num_heads)
            v_mem = _split_heads(v_mem, layer.cross_attn.num_heads)
            cross_kv_cache.append((k_mem, v_mem))

        self_kv_caches = [None] * len(self.decoder_layers)

        current = start_latent  # (B, K, C)
        predictions = []

        for step in range(steps):
            # embed just the new timestep (positional + joint encoding),
            # equivalent to _embed_queries on a length-1 prefix at t0=step.
            x_new = current + self.future_time_enc[step].view(1, 1, C)
            x_new = x_new + self.dec_joint_enc.view(1, K, C)
            x_new = self.query_drop(x_new)

            for li, layer in enumerate(self.decoder_layers):
                x_new, self_kv_caches[li] = layer.forward_step(
                    x_new, self_kv_caches[li], cross_kv_cache[li], self.training
                )

            x_new = self.decoder_norm(x_new)  # layernorm: safe to apply per-token
            predictions.append(x_new)
            current = x_new

        predicted = torch.stack(predictions, dim=1)  # (B, steps, K, C)

        return predicted

    # ---------------- forward ----------------

    def forward(
        self,
        phi: torch.Tensor,
        future_feats: Optional[torch.Tensor] = None,
        pred_steps: Optional[int] = None,
    ):
        """
        Args:
            phi:(B, N_past, feat_dim)
            future_feats: (B, N_future, feat_dim) Training only. Used to construct the target future latent representation.
            pred_steps: Number of autoregressive future steps.

        Returns:
            past_recon: (B, N_past, K, 3)

            future_pred:  (B, pred_steps, K, 3)

            predicted_latent:  (B, pred_steps, K, C)

            future_target_latent: (B, pred_steps, K, C) only when future_feats is supplied.
        """
        if pred_steps is None:
            pred_steps = self.n_future

        if pred_steps > self.n_future:
            raise ValueError(f"pred_steps={pred_steps} exceeds n_future={self.n_future}")
        
        # 1. encode the past into memory tokens
        memory = self.encoder(phi)                    # (B, N_past, K, C)

        # 2. past reconstruction straight off the latent, via the shared head
        past_recon = self.pose_head(memory)           # (B, N_past, K, 3)

        # 3. future prediction: decoder output through the same shared head
        start_latent = memory[:, -1]                     # (B, K, C)

        predicted_latent = self._rollout(memory, start_latent, pred_steps) 

        # 4. future prediction: decode the predicted latent into 3D joints
        future_pred = self.pose_head(predicted_latent)  # (B, pred_steps, K, 3)

        if future_feats is None:
            return past_recon, future_pred, predicted_latent
        
        # 5. compute the target latent representation for the future, to supervise the predicted latent
        with torch.no_grad():
            window = torch.cat([phi, future_feats[:, :pred_steps]], dim=1)
            full_latent = self.encoder(window)                        # (B, N_past+pred_steps, K, C)
            future_target_latent = full_latent[:, self.n_past:]       # (B, pred_steps, K, C)

        return past_recon, future_pred, predicted_latent, future_target_latent