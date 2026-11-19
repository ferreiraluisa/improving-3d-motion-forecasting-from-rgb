import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# RESIDUAL BLOCK
# ============================================================
# From the paper: both f_movie and f_AR consist of 3 residual blocks.
# Each block: GroupNorm → ReLU → CausalConv1d → GroupNorm → ReLU → CausalConv1d + skip.
# Each 1D convolution uses kernel_size=3.
# 3 blocks × 2 convolutions × (kernel_size - 1) = 3 × 2 × 2 = 12 → receptive field = 13 frames.

class CausalConv1d(nn.Module):
    """1D convolution with left-only (causal) padding using edge replication."""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.left_pad = kernel_size - 1
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size, padding=0,
        )

    def forward(self, x):
        # x: (B, C, T)
        if self.left_pad > 0:
            x = F.pad(x, (self.left_pad, 0), mode="replicate")
        return self.conv(x)

class CausalGroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups, num_channels)

    def forward(self, x):  # (B, C, T)
        B, C, T = x.shape
        y = x.permute(0, 2, 1).reshape(B * T, C)   # each frame is its own "sample"
        y = self.gn(y)
        return y.reshape(B, T, C).permute(0, 2, 1)

class ResidualBlock(nn.Module):
    """GroupNorm → ReLU → CausalConv → Dropout → GroupNorm → ReLU → CausalConv + skip."""
    def __init__(self, hidden=512, kernel=3, dropout=0.2):
        super().__init__()
        num_groups = 32 if hidden >= 32 else 1
        self.gn1 = CausalGroupNorm(num_groups, hidden)
        self.gn2 = CausalGroupNorm(num_groups, hidden)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = CausalConv1d(hidden, hidden, kernel_size=kernel)
        self.conv2 = CausalConv1d(hidden, hidden, kernel_size=kernel)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):  # (B, H, T)
        y = self.gn1(x)
        y = self.relu(y)
        y = self.conv1(y)
        y = self.drop(y)
        y = self.gn2(y)
        y = self.relu(y)
        y = self.conv2(y)
        return x + y


# ============================================================
# CAUSAL TEMPORAL ENCODER (f_movie) / AUTOREGRESSIVE PREDICTOR (f_AR)
# ============================================================
# Both share the same architecture but are separate instances.
# f_movie: maps per-frame ResNet features → movie strips Φ_t
# f_AR:    maps past movie strips → predicted future movie strip
#
# FIX 3: Removed the residual skip connection (feats + delta) that
# anchored the output to single-frame features. f_movie now fully
# transforms the input into a new latent representation.

# [05/10] check causality and make sure teacher forcing is correct.!
class CausalTemporalNet(nn.Module):
    def __init__(self, dim=2048, hidden=512, num_blocks=3, kernel=3, dropout=0.2):
        super().__init__()
        self.pre_ln = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, hidden)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden=hidden, kernel=kernel, dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.post_ln = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, dim)

    def forward(self, feats):
        # feats: (B, T, dim)
        x = self.in_proj(self.pre_ln(feats))  # (B, T, hidden)
        x = x.permute(0, 2, 1)                # (B, hidden, T)
        for b in self.blocks:
            x = b(x)
        x = x.permute(0, 2, 1)                # (B, T, hidden)
        x = self.out_proj(self.post_ln(x))     # (B, T, dim)
        return x  # pure transformation, no residual skip to raw features


# ============================================================
# 3D JOINT REGRESSOR (f_3D) — with delta offset support
# ============================================================
# Adapted from HMR's iterative regressor to output 3D joint coordinates.
#
# FIX 2: Added separate regression heads for different delta_t offsets.
# From the same movie strip Φ_t, the model can predict joints at:
#   delta_t =  0  → joints at time t
#   delta_t = -5  → joints at time t-5
#   delta_t = -10 → joints at time t-10
# This forces f_movie to encode real temporal history, not just
# the current frame's information.

class JointRegressor(nn.Module):
    def __init__(self, latent_dim=2048, joints_num=17, hidden=1024,
                 dropout=0.2, deltas=None):
        super().__init__()
        if deltas is None:
            deltas = [0, -5, -10]
        self.deltas = deltas
        self.joints_num = joints_num

        # One regression head per delta offset
        self.heads = nn.ModuleDict()
        for d in deltas:
            key = f"delta_{d}"
            self.heads[key] = nn.Sequential(
                nn.Linear(latent_dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, joints_num * 3),
            )

    def forward(self, phi, delta_t=0):
        """
        Args:
            phi: movie strip features (B, T, D)
            delta_t: time offset (0, -5, or -10)
        Returns:
            joints: (B, T, joints_num, 3)
        """
        B, T, D = phi.shape
        key = f"delta_{delta_t}"
        y = self.heads[key](phi)
        return y.view(B, T, self.joints_num, 3)


# ============================================================
# PHD MODEL
# ============================================================
# Architecture:
#   1. ResNet-50 backbone extracts per-frame features (preprocessed offline)
#   2. f_movie (CausalTemporalNet) encodes features → movie strips Φ
#   3. f_3D (JointRegressor) decodes movie strips → 3D joints
#   4. f_AR (CausalTemporalNet) predicts future movie strips autoregressively
#
# Training phases:
#   Phase 1: Train f_movie + f_3D on reconstruction (with delta offsets)
#   Phase 2: Freeze f_3D, train f_movie + f_AR with curriculum

class PHDFor3DJoints(nn.Module):
    def __init__(self, latent_dim=2048, joints_num=17, hidden=512,
                 dropout=0.0, receptive_field=13, n_cond=15, deltas=None):
        super().__init__()

        if deltas is None:
            deltas = [0, -5, -10]

        self.latent_dim = latent_dim
        self.receptive_field = receptive_field # context size of f_ar
        self.n_cond = n_cond
        self.deltas = deltas

        self.f_movie = CausalTemporalNet(latent_dim, hidden=hidden, dropout=dropout)
        self.f_AR = CausalTemporalNet(latent_dim, hidden=hidden, dropout=dropout)
        self.f_3D = JointRegressor(latent_dim, joints_num, dropout=dropout, deltas=deltas)

    def forward(self, feats, horizon=25, k_self=None):
        return self.forward_phase2(feats, horizon=horizon, k_self=k_self)

    # ----------------------------------------------------------
    # Phase 1: reconstruction (no future prediction)
    # ----------------------------------------------------------
    def forward_phase1(self, feats):
        """
        Phase 1 forward pass: train f_movie + f_3D.

        Args:
            feats: (B, T, 2048) pre-extracted ResNet features
        Returns:
            phi:        (B, T, D)    movie strips
            joints_dict: dict        {delta_t: (B, T', J, 3)} joints per offset
        """
        # [05/10] run with this to see
        # torch.Size([512, 40, 1024]) 40 
        # print(feats.shape) 
        phi = self.f_movie(feats)  # (B, T, D)

        joints_dict = {}
        for d in self.deltas:
            joints_dict[d] = self.f_3D(phi, delta_t=d)

        return phi, joints_dict

    # ----------------------------------------------------------
    # Phase 2: autoregressive future prediction with curriculum
    # ----------------------------------------------------------
    def forward_phase2(self, feats, horizon=25, k_self=None):
        """
        Phase 2 forward pass: train f_movie + f_AR (f_3D is frozen).

        Implements autoregressive self-feeding: f_AR predicts one future
        movie strip at a time, then feeds its own prediction back as
        input for the next step.

        Args:
            feats:      (B, T, 2048) pre-extracted ResNet features
            pred_steps: number of future steps to predict (curriculum)
        Returns:
            phi:         (B, T, D)          real movie strips from f_movie
            phi_hat:     (B, pred_steps, D) predicted future movie strips
            joints_phi:  (B, T, J, 3)       joints from real movie strips
            joints_hat:  (B, pred_steps, J, 3) joints from predicted strips
        """
        if k_self == None:
            k_self = horizon
        R = self.receptive_field

        phi = self.f_movie(feats)  # (B, T, D)

        # Joints from real movie strips (for L_movie_strip target)
        joints_phi = self.f_3D(phi, delta_t=0)

        preds = []
        ctx = None
        for s in range(horizon):
            t = self.n_cond + s # idx being predicted
            if s % k_self == 0:
                # (re)start from ground-truth strips phi[t-R : t]  (all strictly < t)
                lo = max(0, t - R)
                ctx = phi[:, lo:t, :]
            out = self.f_AR(ctx)[:, -1:, :]        # prediction for index t
            preds.append(out)
            ctx = torch.cat([ctx, out], dim=1)[:, -R:, :]
        
        phi_hat = torch.cat(preds, dim=1)          # (B, horizon, D)
        joints_hat = self.f_3D(phi_hat, delta_t=0)
        return phi, phi_hat, joints_phi, joints_hat

    # ----------------------------------------------------------
    # Inference: predict future from conditioning frames
    # ----------------------------------------------------------
    @torch.no_grad()
    def predict_future(self, feats, pred_steps=25):
        self.eval()
        feats = feats[:, :self.n_cond] # keep only observed frames
        phi = self.f_movie(feats)
        context = phi[:, -self.receptive_field:, :]

        predictions = []
        for _ in range(pred_steps):
            next_phi = self.f_AR(context)[:, -1:, :]
            predictions.append(next_phi)

            context = torch.cat([context, next_phi], dim=1)[:, -self.receptive_field:, :]

        phi_hat = torch.cat(predictions, dim=1)
        predicted_joints = self.f_3D(phi_hat, delta_t=0)

        return predicted_joints


# ============================================================
# LOSS FUNCTIONS
# ============================================================

def mpjpe(predicted, target):
    """
    Mean Per Joint Position Error.
    Args:
        predicted: (B, T, J, 3)
        target:    (B, T, J, 3)
    Returns:
        scalar loss
    """
    return torch.mean(torch.norm(predicted - target, dim=-1))


def compute_phase1_loss(model, feats, gt_joints):
    """
    Phase 1 loss: L_movie = Σ_t L_3D(t) + Σ_Δt L_3D(t+Δt)

    Trains f_movie + f_3D. The delta losses force f_movie to encode
    temporal history into the movie strips.

    Args:
        model:     PHDFor3DJoints
        feats:     (B, T, 2048)
        gt_joints: (B, T, J, 3)
    Returns:
        total_loss, loss_dict
    """
    phi, joints_dict = model.forward_phase1(feats)

    loss_dict = {}

    # Main loss at delta_t = 0: predict joints at time t
    loss_dict["L_3D"] = mpjpe(joints_dict[0], gt_joints)

    # Delta offset losses: from Φ_t, also predict joints at t-5 and t-10
    for d in model.deltas:
        if d == 0:
            continue
        offset = abs(d)
        # joints_dict[d] at position t predicts gt at position t + d (= t - offset)
        # So we compare prediction[:, offset:] with gt[:, :-offset]
        pred = joints_dict[d][:, offset:, :]
        target = gt_joints[:, :-offset, :]
        loss_dict[f"L_3D_delta{d}"] = mpjpe(pred, target)

    total_loss = sum(loss_dict.values())
    return total_loss, loss_dict


def compute_phase2_loss(model, feats, gt_joints, pred_steps):
    """
    Phase 2 loss: L_AR = Σ_t L_3D(t) + L_movie_strip + L_const

    Trains f_movie + f_AR (f_3D must be frozen before calling this).

    Args:
        model:      PHDFor3DJoints
        feats:      (B, T, 2048)
        gt_joints:  (B, T, J, 3)
        pred_steps: number of future steps (curriculum)
    Returns:
        total_loss, loss_dict
    """
    n_cond = model.n_cond
    phi, phi_hat, joints_phi, joints_hat = model.forward_phase2(feats, pred_steps)

    loss_dict = {}

    # L_3D on predicted future joints
    gt_future = gt_joints[:, n_cond:n_cond + pred_steps, :]
    loss_dict["L_3D_future"] = mpjpe(joints_hat, gt_future)

    # L_movie_strip: predicted movie strips should match real ones
    real_future_strips = phi[:, n_cond:n_cond + pred_steps, :]
    loss_dict["L_movie_strip"] = F.mse_loss(phi_hat, real_future_strips)

    # L_3D on reconstruction (real movie strips → joints)
    loss_dict["L_3D_recon"] = mpjpe(joints_phi, gt_joints)

    total_loss = sum(loss_dict.values())
    return total_loss, loss_dict


@torch.no_grad()
def check_causality(model, T=40, D=2048, horizon=25):
    model.eval()
    n_cond = model.n_cond
    x = torch.randn(2, T, D)

    # --- Test 1: f_movie is causal ---
    x2 = x.clone()
    x2[:, n_cond:] = torch.randn_like(x2[:, n_cond:]) * 10
    p1 = model.f_movie(x)[:, :n_cond]
    p2 = model.f_movie(x2)[:, :n_cond]
    print("1) f_movie past change     :", (p1 - p2).abs().max().item(), "(should be 0)")

    # --- Test 2: predict_future ignores future frames ---
    j1 = model.predict_future(x, horizon)
    j2 = model.predict_future(x2, horizon)
    print("2) predict_future change   :", (j1 - j2).abs().max().item(), "(should be 0)")

    # --- Test 3: training and inference give the same predictions ---
    # forward_phase2 with full self-feeding should equal predict_future
    _, phi_hat, _, joints_hat = model.forward_phase2(x, horizon=horizon, k_self=horizon)
    print("3) train vs inference diff :", (joints_hat - j1).abs().max().item(), "(should be ~0)")

    # --- Test 4: prediction s depends only on frames < n_cond + s ---
    s = 5
    t = n_cond + s
    x3 = x.clone()
    x3[:, t:] = torch.randn_like(x3[:, t:]) * 10        # perturb from frame t onward
    _, ph1, _, _ = model.forward_phase2(x,  horizon=horizon, k_self=1)  # teacher forcing
    _, ph3, _, _ = model.forward_phase2(x3, horizon=horizon, k_self=1)
    print(f"4) teacher-forced pred {s} change:",
          (ph1[:, s] - ph3[:, s]).abs().max().item(), "(should be 0)")
    print(f"   teacher-forced pred {s+1} change:",
          (ph1[:, s + 1] - ph3[:, s + 1]).abs().max().item(), "(should be > 0)")


if __name__ == "__main__":
    torch.manual_seed(0)
    check_causality(PHDFor3DJoints())
