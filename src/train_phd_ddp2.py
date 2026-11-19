import os
import sys
import csv
import argparse
import random
from datetime import datetime
import wandb

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist

# model
from models.phd_model import PHDFor3DJoints

# dataset
from dataset.losd_data_h36 import Fusion as FusionH36M
from dataset.losd_data_3dpw import Fusion3DPW
from dataset.dataloader import Human36mDataset, ThreeDPWDataset

# utils
from utils.mpjpe import mpjpe_3d, loss_velocity
from utils.config_tools import get_config as get_config_aux
from utils.ddp_utils import reduce_tensor
from utils.utils import dtw_aligned_metrics

from visualize.visualize import helper_plot_3d_2im


SPLITS_3DPW = {'train': 'train', 'val': 'validation', 'test': 'test'}

H36M_ACTIONS = [
    "walking", "eating", "smoking", "discussion", "directions",
    "greeting", "phoning", "posing", "purchases", "sitting",
    "sittingdown", "takingphoto", "waiting", "walkingdog", "walkingtogether",
]

# CLI args that the yaml config must NOT override (the yaml is merged on top of
# the CLI below, and an H36M yaml could otherwise silently reset the dataset).
DATASET_ARGS = ('dataset', 'dataset_3dpw_path', 'dataset_3dpw_feats', 'img_dir_3dpw')


# ─────────────────────────── dataset selection ───────────────────────────────

def build_raw_dataset(args):
    """Returns (Fusion class, raw dataset object)."""
    if args.dataset == 'h36m':
        return FusionH36M, Human36mDataset(args.dataset_path)
    if args.dataset == '3dpw':
        return Fusion3DPW, ThreeDPWDataset(args.dataset_3dpw_path)
    raise ValueError(args.dataset)


def eval_groups(args, dataset, mode_t):
    """
    List of (name, extra kwargs for the Fusion constructor) used for val/test.
      H36M : one group per action       -> action=<act>
      3DPW : one group per source video -> sequences=[<seq>]
    """
    if args.dataset == 'h36m':
        return [(a, dict(action=a)) for a in H36M_ACTIONS]

    split = SPLITS_3DPW[mode_t]
    seqs = sorted({e['source_seq'] for e in dataset[split].values()})
    return [(s, dict(sequences=[s])) for s in seqs]


def setup_dataset_args(args):
    """Dataset-specific checks / fixed settings (call after the yaml merge)."""
    if args.dataset == '3dpw':
        missing = [n for n in ('dataset_3dpw_path', 'dataset_3dpw_feats') if not getattr(args, n)]
        if missing:
            raise ValueError(f'--dataset 3dpw needs: {", ".join("--" + m for m in missing)}')
        args.out_all = True          # loader must return all T frames of gt_3D
        args.split_train_3dpw = SPLITS_3DPW['train']
        args.split_val_3dpw = SPLITS_3DPW['val']
        args.split_test_3dpw = SPLITS_3DPW['test']
        args.normalize_2d_3dpw = True
    return args


def build_eval_loaders(args, Fusion, dataset, mode_t, world_size, rank, workers):
    """One DataLoader per eval group. Groups with no full window (3DPW segments
    shorter than seq_len) are skipped so they never enter the average as 0 mm."""
    loaders, sets = {}, {}
    for name, kw in eval_groups(args, dataset, mode_t):
        ds = Fusion(args, dataset, train=False, mode_t=mode_t, **kw)
        if len(ds) == 0:
            print(f"[{mode_t}] skip group '{name}': no window of {args.seq_len} frames")
            continue
        sampler = torch.utils.data.distributed.DistributedSampler(
            ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        loaders[name] = torch.utils.data.DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            num_workers=workers, pin_memory=True, sampler=sampler)
        sets[name] = ds
    if not loaders:
        raise RuntimeError(f'No {mode_t} group has a window of {args.seq_len} frames.')
    return loaders, sets


def check_shapes(args, probe_ds):
    """Adapt num_joints / latent_dim to the real data (3DPW features and joints
    differ from H36M) and check the window length."""
    sample = probe_ds[0]
    real_feat_dim = int(sample[3].shape[-1])
    real_joints = int(sample[1].shape[-2])
    real_T = int(sample[1].shape[0])
    if real_joints != args.num_joints:
        print(f"[warn] num_joints={args.num_joints} but GT has {real_joints}; using {real_joints}.")
        args.num_joints = real_joints
    if real_feat_dim != args.latent_dim:
        print(f"[warn] --latent_dim={args.latent_dim} but features are {real_feat_dim}-d; "
              f"using {real_feat_dim}.")
        args.latent_dim = real_feat_dim
    assert real_T == args.seq_len, (
        f"Dataset returns {real_T} frames per window but past+future={args.seq_len}.")


# ─────────────────────────── phase-2 helpers ────────────────────────────────

def curriculum_steps(epoch, max_steps, curriculum_epochs):
    """Linearly ramp pred_steps from 1 → max_steps over curriculum_epochs."""
    if max_steps <= 1:
        return 1
    if curriculum_epochs <= 1:
        return max_steps
    alpha = min(1.0, max(0.0, epoch / float(curriculum_epochs - 1)))
    return int(max(1, min(max_steps, 1 + round(alpha * (max_steps - 1)))))


def phase2_losses(joints_phi, joints_hat, phi, phi_hat, gt_joints,
                  n_cond, pred_steps, w_recon_3d, w_future_3d, w_latent, w_vel):
    """
    Phase 2 losses.

    Args:
        joints_phi: (B, T, J, 3) joints from real movie strips (all frames)
        joints_hat: (B, pred_steps, J, 3) joints from predicted future movie strips
        phi:        (B, T, D) real movie strips from f_movie
        phi_hat:    (B, pred_steps, D) predicted future movie strips from f_AR
        gt_joints:  (B, T, J, 3) ground truth joints for full sequence
        n_cond:     number of conditioning frames
        pred_steps: number of predicted future steps
        w_*:        loss weights
    """
    # L_3D on reconstruction (real movie strips → joints, all frames)
    recon_3d = mpjpe_3d(joints_phi, gt_joints)

    # L_3D on predicted future joints
    gt_future = gt_joints[:, n_cond:n_cond + pred_steps, :, :]
    future_3d = mpjpe_3d(joints_hat, gt_future)

    # L_movie_strip: predicted movie strips should match real future movie strips
    real_future_strips = phi[:, n_cond:n_cond + pred_steps, :]
    latent = nn.functional.mse_loss(phi_hat, real_future_strips)

    # Velocity loss on predicted future
    vel = loss_velocity(joints_hat, gt_future)

    total = w_recon_3d * recon_3d + w_future_3d * future_3d + w_latent * latent + w_vel * vel

    stats = {
        'loss':      total.item(),
        'recon_3d':  recon_3d.item(),
        'future_3d': future_3d.item(),
        'latent':    latent.item(),
        'vel':       vel.item(),
    }
    return total, stats


def freeze_module(module):
    module.eval()
    for p in module.parameters():
        p.requires_grad = False


def load_phase1_weights(model, ckpt_path):
    """Load phase-1 checkpoint into model.module, handling both checkpoint formats."""
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
    # strip DDP 'module.' prefix if present
    state = {k[7:] if k.startswith('module.') else k: v for k, v in state.items()}
    missing, unexpected = model.module.load_state_dict(state, strict=False)
    if missing:
        print(f"[phase1 load] Missing keys: {missing}")
    if unexpected:
        print(f"[phase1 load] Unexpected keys: {unexpected}")
    print(f"Loaded phase-1 weights from: {ckpt_path}")


# ─────────────────────────── train / val ────────────────────────────────────

def train_epoch(model, args, dataloader, optimizer, epoch, writer, local_rank):
    model.train()
    # keep f_3D frozen and in eval mode throughout
    freeze_module(model.module.f_3D)
    torch.cuda.set_device(local_rank)

    pred_steps = curriculum_steps(epoch, args.max_pred_steps, args.curriculum_epochs)
    n_cond = model.module.n_cond  # 15

    meter = {k: 0.0 for k in ('loss', 'recon_3d', 'future_3d', 'latent', 'vel')}
    counter = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch + 1} [horizon={pred_steps}]',
                disable=(dist.get_rank() != 0))

    for data in pbar:
        batch_size  = data[1].size(0)
        feat_window = data[3].to(local_rank, non_blocking=True)   # (B, T, 2048)
        gt_3D       = data[1].to(local_rank, non_blocking=True)   # (B, T, J, 3)

        T = feat_window.shape[1]
        if n_cond + pred_steps > T:
            continue  # skip if sequence is too short

        optimizer.zero_grad()

        # Use updated model API: forward_phase2 handles autoregressive rollout
        phi, phi_hat, joints_phi, joints_hat = model(
            feat_window, horizon=pred_steps)

        loss, stats = phase2_losses(
            joints_phi, joints_hat, phi, phi_hat, gt_3D,
            n_cond, pred_steps,
            args.w_recon_3d, args.w_future_3d, args.w_latent, args.w_vel)

        loss.backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], args.grad_clip)

        optimizer.step()

        for k in meter:
            meter[k] += stats[k] * batch_size
        counter += batch_size

        pbar.set_postfix({k: f'{stats[k]:.4f}' for k in ('loss', 'future_3d', 'latent')})

    n = max(1, counter)
    avgs = {k: meter[k] / n for k in meter}

    if dist.is_initialized():
        dev = torch.device(f'cuda:{local_rank}')
        ws  = dist.get_world_size()
        avgs = {k: reduce_tensor(torch.tensor(v, device=dev), ws).item() for k, v in avgs.items()}

    return avgs, pred_steps


def validate_epoch(model, args, dataloader, local_rank):
    model.eval()
    torch.cuda.set_device(local_rank)

    n_cond = model.module.n_cond  # 15

    meter = {k: 0.0 for k in ('loss', 'recon_3d', 'future_3d', 'latent', 'vel')}
    counter = 0

    with torch.no_grad():
        for data in dataloader:
            batch_size  = data[1].size(0)
            feat_window = data[3].to(local_rank, non_blocking=True)
            gt_3D       = data[1].to(local_rank, non_blocking=True)

            T = feat_window.shape[1]
            if n_cond + args.max_pred_steps > T:
                continue

            # Use updated model API
            phi, phi_hat, joints_phi, joints_hat = model(
                feat_window, horizon=args.max_pred_steps)

            _, stats = phase2_losses(
                joints_phi, joints_hat, phi, phi_hat, gt_3D,
                n_cond, args.max_pred_steps,
                args.w_recon_3d, args.w_future_3d, args.w_latent, args.w_vel)

            for k in meter:
                meter[k] += stats[k] * batch_size
            counter += batch_size

    n = max(1, counter)
    avgs = {k: meter[k] / n for k in meter}

    if dist.is_initialized():
        dev = torch.device(f'cuda:{local_rank}')
        ws  = dist.get_world_size()
        avgs = {k: reduce_tensor(torch.tensor(v, device=dev), ws).item() for k, v in avgs.items()}

    return avgs


# ─────────────────────────── test ───────────────────────────────────────────

def procrustes_align_batched(pred, gt, eps=1e-8):
    """
    Rigid alignment of pred to gt with rotation + translation + uniform scale,
    computed independently for every frame. Follows the Umeyama / Procrustes
    formulation used in VideoPose3D's p_mpjpe.

    Args:
        pred: (..., J, 3)  source
        gt:   (..., J, 3)  target
    Returns:
        pred_aligned: same shape as pred — minimises ||gt - pred_aligned||² in
                      the least-squares sense over each frame's 7 rigid d.o.f.
    """
    muX = gt.mean(dim=-2, keepdim=True)           # (..., 1, 3)
    muY = pred.mean(dim=-2, keepdim=True)
    X0  = gt   - muX                               # centered target
    Y0  = pred - muY                               # centered source

    normX = torch.sqrt((X0 ** 2).sum(dim=(-1, -2), keepdim=True))  # (..., 1, 1)
    normY = torch.sqrt((Y0 ** 2).sum(dim=(-1, -2), keepdim=True))

    X0n = X0 / (normX + eps)
    Y0n = Y0 / (normY + eps)

    # cross-covariance H = X0ᵀ Y0  shape (..., 3, 3)
    H = torch.matmul(X0n.transpose(-1, -2), Y0n)
    U, s, Vh = torch.linalg.svd(H)
    V = Vh.transpose(-1, -2)

    # reflection check: ensure det(R) = +1
    R_init   = torch.matmul(V, U.transpose(-1, -2))
    sign_det = torch.sign(torch.det(R_init)).view(*R_init.shape[:-2], 1, 1)

    V = V.clone()
    V[..., :, -1:] = V[..., :, -1:] * sign_det
    s = s.clone()
    s[..., -1] = s[..., -1] * sign_det.view(*sign_det.shape[:-2])

    R     = torch.matmul(V, U.transpose(-1, -2))                       # (..., 3, 3)
    tr    = s.sum(dim=-1).view(*s.shape[:-1], 1, 1)                    # (..., 1, 1)
    scale = tr * normX / (normY + eps)                                  # (..., 1, 1)
    t     = muX - scale * torch.matmul(muY, R)                          # (..., 1, 3)

    pred_aligned = scale * torch.matmul(pred, R) + t
    return pred_aligned


def _print_test_block(label, m, dtw_horizons, pred_steps):
    """Print a per-action (or aggregated) test-result block. m is the dict returned by test_epoch."""
    print(f"  [{label}] recon mpjpe_3d: {m['recon_mm']:.2f} mm  "
          f"(Procrustes: {m['recon_aligned_mm']:.2f} mm)  |  "
          f"future mpjpe_3d (mean over all {pred_steps} steps): "
          f"{m['future_mm']:.2f} mm  |  "
          f"Procrustes future: {m['future_aligned_mm']:.2f} mm")
    print("  future MPJPE per step (mm)   [raw  |  Procrustes]:")
    for k, (v, va) in enumerate(zip(m['per_step_mm'], m['per_step_aligned_mm']), start=1):
        print(f"    step {k:>2}:  {v:7.2f}  |  {va:7.2f}")
    key_steps = [s for s in (1, 5, 10, 20, 25, 30) if s <= len(m['per_step_mm'])]
    key_raw = '  '.join(f's{s}={m["per_step_mm"][s-1]:.2f}'         for s in key_steps)
    key_pa  = '  '.join(f's{s}={m["per_step_aligned_mm"][s-1]:.2f}' for s in key_steps)
    print(f"  key horizons raw        : {key_raw}")
    print(f"  key horizons Procrustes : {key_pa}")
    key_dtw_raw = '  '.join(f's{h}={m["dtw_mpjpe_mm"][h]:.2f}' for h in dtw_horizons)
    key_dtw_pa  = '  '.join(f's{h}={m["dtw_pa_mm"][h]:.2f}'    for h in dtw_horizons)
    print(f"  key horizons DTW raw    : {key_dtw_raw}")
    print(f"  key horizons DTW PA     : {key_dtw_pa}")


def test_epoch(model, args, dataloader, local_rank, act):
    """
    Phase-2 test: evaluate autoregressive future-prediction quality.
    Returns mean future MPJPE (in mm) and per-step MPJPE (list of mm values).
    """
    model.eval()
    torch.cuda.set_device(local_rank)

    n_cond     = model.module.n_cond
    pred_steps = args.max_pred_steps

    per_step_sum         = torch.zeros(pred_steps, device=local_rank)
    per_step_aligned_sum = torch.zeros(pred_steps, device=local_rank)
    recon_sum            = torch.zeros(1, device=local_rank)
    recon_aligned_sum    = torch.zeros(1, device=local_rank)
    future_sum           = torch.zeros(1, device=local_rank)
    future_aligned_sum   = torch.zeros(1, device=local_rank)
    counter              = 0
    saved_vis            = False

    # ── per-horizon DTW accumulators (paper §4.2 metric) ──
    # For each horizon h: run DTW on pred[:h] and gt[:h], take mean of path errors.
    dtw_horizons     = [h for h in (1, 5, 10, 20, 25, 30) if h <= pred_steps]
    dtw_mpjpe_sum    = {h: torch.zeros(1, device=local_rank) for h in dtw_horizons}
    dtw_pa_sum       = {h: torch.zeros(1, device=local_rank) for h in dtw_horizons}

    with torch.no_grad():
        for data in dataloader:
            batch_size  = data[1].size(0)
            feat_window = data[3].to(local_rank, non_blocking=True)
            gt_3D       = data[1].to(local_rank, non_blocking=True)

            T = feat_window.shape[1]
            if n_cond + pred_steps > T:
                continue

            _, _, joints_phi, joints_hat = model(
                feat_window, horizon=pred_steps)

            gt_future = gt_3D[:, n_cond:n_cond + pred_steps]  # (B, P, J, 3)

            # ── raw MPJPE (camera/world frame) ──
            per_step = torch.norm(joints_hat - gt_future, dim=-1).mean(dim=(0, 2))  # (P,)
            per_step_sum += per_step * batch_size
            recon_sum    += mpjpe_3d(joints_phi, gt_3D)       * batch_size
            future_sum   += mpjpe_3d(joints_hat, gt_future)   * batch_size

            # ── Procrustes-aligned MPJPE (paper's "Reconst." metric) ──
            # Per-frame rigid alignment (rotation + translation + uniform scale) of
            # the 17-joint pred to the 17-joint gt via Umeyama / Procrustes.
            # This is the metric reported in the paper's Table 1 and Supp. Table 3.
            recon_aligned      = procrustes_align_batched(joints_phi, gt_3D)
            recon_aligned_sum += mpjpe_3d(recon_aligned, gt_3D) * batch_size

            pred_aligned = procrustes_align_batched(joints_hat, gt_future)

            per_step_aligned = torch.norm(
                pred_aligned - gt_future, dim=-1).mean(dim=(0, 2))  # (P,)
            per_step_aligned_sum += per_step_aligned * batch_size
            future_aligned_sum   += mpjpe_3d(pred_aligned, gt_future) * batch_size

            # ── per-horizon DTW (paper §4.2): for each horizon h, run DTW on
            # pred[:h] and gt[:h] and take the mean of per-pair errors along the
            # warping path. NumPy / per-sample loop — slow but only at test time.
            pred_np = joints_hat.detach().cpu().numpy()   # (B, P, J, 3) in metres
            gt_np   = gt_future.detach().cpu().numpy()
            for b in range(batch_size):
                for h in dtw_horizons:
                    dtw_mpjpe_seq, dtw_pa_seq = dtw_aligned_metrics(
                        pred_np[b, :h], gt_np[b, :h])
                    dtw_mpjpe_sum[h] += float(dtw_mpjpe_seq.mean())   # already mm
                    dtw_pa_sum[h]    += float(dtw_pa_seq.mean())

            counter += batch_size

            # # save one visualization per walking-style action
            # if not saved_vis and dist.get_rank() == 0 and \
            #    act in ['walking', 'walkingdog', 'walkingtogether']:
            #     i = 0
            #     os.makedirs('pic_phase2', exist_ok=True)
            #     helper_plot_3d_2im(
            #         joints_hat[i].cpu(),  # (P, J, 3) predicted future
            #         gt_future[i].cpu(),   # (P, J, 3) GT future
            #         name=f'pic_phase2/phase2_future_{act}.png',
            #         n_joint=17,
            #         axis_limit=1.0,
            #     )
            #     print(f"[VIS] Saved phase-2 future prediction for {act}")
            #     saved_vis = True

    counter          = max(1, counter)
    per_step         = per_step_sum         / counter
    per_step_aligned = per_step_aligned_sum / counter
    recon            = recon_sum            / counter
    recon_aligned    = recon_aligned_sum    / counter
    future           = future_sum           / counter
    future_aligned   = future_aligned_sum   / counter
    dtw_mpjpe        = {h: dtw_mpjpe_sum[h] / counter for h in dtw_horizons}
    dtw_pa           = {h: dtw_pa_sum[h]    / counter for h in dtw_horizons}

    if dist.is_initialized():
        ws = dist.get_world_size()
        per_step         = reduce_tensor(per_step,         ws)
        per_step_aligned = reduce_tensor(per_step_aligned, ws)
        recon            = reduce_tensor(recon,            ws)
        recon_aligned    = reduce_tensor(recon_aligned,    ws)
        future           = reduce_tensor(future,           ws)
        future_aligned   = reduce_tensor(future_aligned,   ws)
        dtw_mpjpe        = {h: reduce_tensor(dtw_mpjpe[h], ws) for h in dtw_horizons}
        dtw_pa           = {h: reduce_tensor(dtw_pa[h],    ws) for h in dtw_horizons}

    per_step_mm         = [v * 1000 for v in per_step.tolist()]
    per_step_aligned_mm = [v * 1000 for v in per_step_aligned.tolist()]

    metrics = {
        'recon_mm':            recon.item() * 1000,
        'recon_aligned_mm':    recon_aligned.item() * 1000,
        'future_mm':           future.item() * 1000,
        'future_aligned_mm':   future_aligned.item() * 1000,
        'per_step_mm':         per_step_mm,
        'per_step_aligned_mm': per_step_aligned_mm,
        'dtw_mpjpe_mm':        {h: dtw_mpjpe[h].item() for h in dtw_horizons},
        'dtw_pa_mm':           {h: dtw_pa[h].item()    for h in dtw_horizons},
        'dtw_horizons':        dtw_horizons,
    }

    if dist.get_rank() == 0:
        _print_test_block(act, metrics, dtw_horizons, pred_steps)

    return metrics


# ─────────────────────────── checkpoint helpers ──────────────────────────────

def save_checkpoint(model, optimizer, scheduler, epoch, best_val, path):
    if dist.get_rank() != 0:
        return
    torch.save({
        'epoch':                epoch,
        'model_state_dict':     model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss':                 best_val,
    }, path)
    print(f"Checkpoint saved: {path}")


def load_checkpoint_for_resume(model, optimizer, scheduler, path):
    ckpt = torch.load(path, map_location='cpu')
    state = ckpt['model_state_dict']
    state = {k[7:] if k.startswith('module.') else k: v for k, v in state.items()}
    model.module.load_state_dict(state)
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler and ckpt.get('scheduler_state_dict'):
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    start_epoch = ckpt['epoch'] + 1
    best_val    = ckpt['loss']
    print(f"Resumed from epoch {ckpt['epoch']}, best val: {best_val:.6f}")
    return start_epoch, best_val


# ─────────────────────────── main ────────────────────────────────────────────

from torch.distributed.elastic.multiprocessing.errors import record

@record
def main():
    seed = 1126
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark     = False
    torch.backends.cudnn.deterministic = True

    parser = argparse.ArgumentParser()

    # ── dataset choice ──
    parser.add_argument('--dataset', type=str, default='h36m', choices=['h36m', '3dpw'])

    # ── 3DPW paths (splits are always train / validation / test, eval per video) ──
    parser.add_argument('--dataset_3dpw_path',  type=str, default=None,
                        help='3DPW meta file read by dataloader.ThreeDPWDataset.')
    parser.add_argument('--dataset_3dpw_feats', type=str, default=None,
                        help='3DPW features .npy (--out of losd_data_3dpw.py).')
    parser.add_argument('--img_dir_3dpw',       type=str, default=None,
                        help='<3DPW root>/imageFiles (only needed for frame_info / viz).')

    # ── H36M paths ──
    parser.add_argument('--dataset_path',         type=str, default='data/data_3d_h36m.npz')
    parser.add_argument('--dataset_2d_path',      type=str, default='data/data_2d_h36m_gt.npz')
    parser.add_argument('--dataset_2d_vit_path',  type=str, default='data/compatible_keypoints.pkl')
    parser.add_argument('--features_path', type=str, default='output/checkpoints')

    # ── checkpoints ──
    parser.add_argument('--phase1_ckpt',           type=str, default='server_models/best_phd_model_phase1.pth',
                        help='Path to phase-1 checkpoint to initialise weights from')
    parser.add_argument('--checkpoint_dir',        type=str, default='work_dir/checkpoints_phase2')
    parser.add_argument('--checkpoint_save_name',  type=str, default='best_phase2.pth')
    parser.add_argument('--resume',                type=str, default='',
                        help='Path to phase-2 checkpoint to resume training from')

    # ── work dir ──
    parser.add_argument('--work_dir',  type=str, default='./work_dir_phase2')
    parser.add_argument('--log_csv',   type=str, default='')

    # ── config ──
    parser.add_argument('--config_path', type=str, default='config/MB_train_h36m_short.yaml')

    # ── training hyper-params ──
    parser.add_argument('--epochs',      type=int,   default=40)
    parser.add_argument('--batch_size',  type=int,   default=16)
    parser.add_argument('--lr',          type=float, default=1e-4)
    parser.add_argument('--grad_clip',   type=float, default=1.0)
    parser.add_argument('--save_freq',   type=int,   default=5)
    parser.add_argument('--val_freq',    type=int,   default=1)
    parser.add_argument('--workers',     type=int,   default=2)

    # ── dataset config ──
    parser.add_argument('--subjects_train', type=str, default='S1,S5,S6,S7,S8')
    parser.add_argument('--subjects_val',   type=str, default='S11')
    parser.add_argument('--subjects_test',  type=str, default='S9')
    parser.add_argument('--data_augmentation', action='store_true', default=False)
    parser.add_argument('--reverse_augmentation', action='store_true', default=False)
    parser.add_argument('--downsample', type=int, default=2)
    parser.add_argument('--stride',     type=int, default=2)
    parser.add_argument('--img_size', nargs=2, type=int, default=[192, 256])

    # ── model ──
    parser.add_argument('--latent_dim', type=int,   default=2048)
    parser.add_argument('--dropout',    type=float, default=0.0)
    parser.add_argument('--mode',       type=str,   default='test')

    # ── phase-2 specific ──
    parser.add_argument('--max_pred_steps',    type=int,   default=25,
                        help='Maximum future steps to predict at full curriculum')
    parser.add_argument('--curriculum_epochs', type=int,   default=15,
                        help='Epochs to ramp pred_steps from 1 to max_pred_steps')
    parser.add_argument('--w_recon_3d',  type=float, default=1.0)
    parser.add_argument('--w_future_3d', type=float, default=1.0)
    parser.add_argument('--w_latent',    type=float, default=1.0)
    parser.add_argument('--w_vel',       type=float, default=0.1)

    # ──── wandb ────
    parser.add_argument('--wandb_project', type=str, default='baseline_hmr')
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--wandb_mode', type=str, default='online', choices=['online', 'offline', 'disabled'])

    args = parser.parse_args()

    # merge yaml config (CLI dataset choice / 3DPW paths are kept)
    cli_dataset_args = {k: getattr(args, k) for k in DATASET_ARGS}
    args2 = get_config_aux(args.config_path)
    args_dict = vars(args)
    args_dict.update(vars(args2))
    args_dict.update(cli_dataset_args)
    args = argparse.Namespace(**args_dict)

    # Phase 2 sequence length: 15 conditioning + 25 future = 40
    args.past_length   = 15
    args.future_length = 25
    args.seq_len       = args.past_length + args.future_length  # 40
    args.frames        = args.seq_len
    args.root_joint    = 0
    args = setup_dataset_args(args)

    # sanity check: need enough frames for conditioning + max prediction
    n_cond = 15
    if n_cond + args.max_pred_steps > args.seq_len:
        raise ValueError(
            f"n_cond ({n_cond}) + max_pred_steps ({args.max_pred_steps}) "
            f"must be <= seq_len ({args.seq_len})")

    # ── DDP init ──
    world_size = int(os.environ['WORLD_SIZE'])
    rank       = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=world_size, rank=rank,
                            device_id=torch.device(f'cuda:{local_rank}'))
    dist.barrier()

    if rank == 0:
        print(f"DDP initialized, world_size={world_size}", flush=True)

    if dist.get_rank() != 0:
        f = open(os.devnull, 'w')
        sys.stdout = f
        sys.stderr = f

    # ── dirs / tensorboard ──
    if dist.get_rank() == 0:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        os.makedirs(args.work_dir,       exist_ok=True)
        log_dir = os.path.join(args.work_dir, 'logs', datetime.now().strftime('%Y%m%d_%H%M%S'))
        writer  = SummaryWriter(log_dir)

        wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or datetime.now().strftime('%Y%m%d_%H%M%S'),
        config=vars(args),
        mode=args.wandb_mode,
        )
    else:
        writer = None

    # ── dataset ──
    Fusion, dataset = build_raw_dataset(args)

    if args.mode == 'train':
        train_data = Fusion(args, dataset, train=True)
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_data, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, sampler=train_sampler,
            persistent_workers=args.workers > 0,
            prefetch_factor=2 if args.workers > 0 else None, drop_last=True)

        val_loaders, _ = build_eval_loaders(
            args, Fusion, dataset, 'val', world_size, rank, args.workers)
        acts = list(val_loaders)
        probe_ds = train_data
    else:
        test_loaders, test_sets = build_eval_loaders(
            args, Fusion, dataset, 'test', world_size, rank, args.workers)
        acts = list(test_loaders)
        probe_ds = test_sets[acts[0]]

    print(f"[{args.dataset}] {len(acts)} eval group(s): "
          f"{'actions' if args.dataset == 'h36m' else 'videos'}")
    check_shapes(args, probe_ds)

    # ── model ──
    model = PHDFor3DJoints(
        latent_dim=args.latent_dim,
        joints_num=args.num_joints,
        dropout=args.dropout,
        n_cond=15
    ).to(local_rank)
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank,
        find_unused_parameters=True)

    # ── load phase-1 weights, then freeze f_3D ──
    load_phase1_weights(model, args.phase1_ckpt)
    freeze_module(model.module.f_3D)

    # Only train f_movie + f_AR (f_3D is frozen)
    trainable = (
        list(model.module.f_movie.parameters()) +
        list(model.module.f_AR.parameters())
    )
    print(f"Trainable params: {sum(p.numel() for p in trainable):,}  "
          f"(f_movie + f_AR only)")

    optimizer = optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # ── test branch ──
    if args.mode != 'train':
        ckpt_path = args.resume or os.path.join(args.checkpoint_dir, args.checkpoint_save_name)
        print(f"Loading phase-2 checkpoint for test: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state = ckpt.get('model_state_dict', ckpt)
        state = {k[7:] if k.startswith('module.') else k: v for k, v in state.items()}
        model.module.load_state_dict(state)

        print("Starting phase-2 inference...")
        all_metrics = []
        for act in acts:
            print('-' * 50)
            print(f"{'Action' if args.dataset == 'h36m' else 'Video'}: {act}")
            m = test_epoch(model, args, test_loaders[act], local_rank, act)
            all_metrics.append(m)

        if dist.get_rank() == 0 and all_metrics:
            dtw_horizons = all_metrics[0]['dtw_horizons']
            P = len(all_metrics[0]['per_step_mm'])
            n = len(all_metrics)

            # mean over actions (unweighted, matching paper convention)
            avg = {
                'recon_mm':           sum(m['recon_mm']          for m in all_metrics) / n,
                'recon_aligned_mm':   sum(m['recon_aligned_mm']  for m in all_metrics) / n,
                'future_mm':          sum(m['future_mm']         for m in all_metrics) / n,
                'future_aligned_mm':  sum(m['future_aligned_mm'] for m in all_metrics) / n,
                'per_step_mm':         [sum(m['per_step_mm'][k]         for m in all_metrics) / n for k in range(P)],
                'per_step_aligned_mm': [sum(m['per_step_aligned_mm'][k] for m in all_metrics) / n for k in range(P)],
                'dtw_mpjpe_mm':        {h: sum(m['dtw_mpjpe_mm'][h] for m in all_metrics) / n for h in dtw_horizons},
                'dtw_pa_mm':           {h: sum(m['dtw_pa_mm'][h]    for m in all_metrics) / n for h in dtw_horizons},
            }

            print('=' * 50)
            print(f"[{args.dataset}] All ({n} {'actions' if args.dataset == 'h36m' else 'videos'})")
            _print_test_block('All', avg, dtw_horizons, args.max_pred_steps)

            if dist.get_rank() == 0:
                wandb.finish()

        dist.barrier()
        dist.destroy_process_group()
        return

    # ── csv log ──
    log_csv = args.log_csv or os.path.join(args.work_dir, 'train_phase2_log.csv')
    csv_header_written = os.path.exists(log_csv)

    # ── resume or fresh start ──
    start_epoch  = 0
    best_val     = float('inf')

    if args.resume and os.path.exists(args.resume):
        start_epoch, best_val = load_checkpoint_for_resume(
            model, optimizer, scheduler, args.resume)

    if dist.get_rank() == 0:
        print("===== Phase 2 =====")
        print(f"Phase1 ckpt:       {args.phase1_ckpt}")
        print(f"seq_len:           {args.seq_len} (past={args.past_length} + future={args.future_length})")
        print(f"receptive_field:   {model.module.receptive_field}")
        print(f"max_pred_steps:    {args.max_pred_steps}")
        print(f"curriculum_epochs: {args.curriculum_epochs}")
        print(f"Loss weights:  recon={args.w_recon_3d}  future={args.w_future_3d}  "
              f"latent={args.w_latent}  vel={args.w_vel}")
        print("===================")

    # ── training loop ──
    for epoch in range(start_epoch, args.epochs):
        args.epoch = epoch
        train_sampler.set_epoch(epoch)

        print(f'\n=== Epoch {epoch + 1}/{args.epochs} ===')

        train_stats, pred_steps = train_epoch(
            model, args, train_loader, optimizer, epoch, writer, local_rank)

        # validation: aggregate over all actions
        val_stats_agg = {k: 0.0 for k in ('loss', 'recon_3d', 'future_3d', 'latent', 'vel')}
        for act in acts:
            act_stats = validate_epoch(model, args, val_loaders[act], local_rank)
            for k in val_stats_agg:
                val_stats_agg[k] += act_stats[k]
        val_stats = {k: v / len(acts) for k, v in val_stats_agg.items()}

        print(
            f"train loss={train_stats['loss']:.5f} "
            f"(recon={train_stats['recon_3d']:.5f} future={train_stats['future_3d']:.5f} "
            f"latent={train_stats['latent']:.5f} vel={train_stats['vel']:.5f})  |  "
            f"val future={val_stats['future_3d'] * 1000:.2f} mm"
        )

        if dist.get_rank() == 0:
            writer.add_scalar('Train/Loss',      train_stats['loss'],      epoch)
            writer.add_scalar('Train/Recon3D',   train_stats['recon_3d'],  epoch)
            writer.add_scalar('Train/Future3D',  train_stats['future_3d'], epoch)
            writer.add_scalar('Train/Latent',    train_stats['latent'],    epoch)
            writer.add_scalar('Train/Vel',       train_stats['vel'],       epoch)
            writer.add_scalar('Train/PredSteps', pred_steps,               epoch)
            writer.add_scalar('Val/Loss',        val_stats['loss'],        epoch)
            writer.add_scalar('Val/Future3D',    val_stats['future_3d'],   epoch)
            writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], epoch)

            wandb.log({
                'train/loss':       train_stats['loss'],
                'train/recon_3d':   train_stats['recon_3d'],
                'train/future_3d':  train_stats['future_3d'],
                'train/latent':     train_stats['latent'],
                'train/vel':        train_stats['vel'],
                'train/pred_steps': pred_steps,
                'train/lr':         optimizer.param_groups[0]['lr'],
                'val/loss':         val_stats['loss'],
                'val/recon_3d':     val_stats['recon_3d'],
                'val/future_3d':    val_stats['future_3d'],
                'val/latent':       val_stats['latent'],
                'val/vel':          val_stats['vel'],
                'val/future_3d_mm': val_stats['future_3d'] * 1000,
            }, step=epoch)

            # save best based on future prediction error
            if val_stats['future_3d'] < best_val:
                best_val = val_stats['future_3d']
                save_checkpoint(
                    model, optimizer, scheduler, epoch, best_val,
                    os.path.join(args.checkpoint_dir, args.checkpoint_save_name))
                print(f"New best model: val future MPJPE = {best_val * 1000:.2f} mm")

            if (epoch + 1) % args.save_freq == 0:
                save_checkpoint(
                    model, optimizer, scheduler, epoch, train_stats['loss'],
                    os.path.join(args.checkpoint_dir, f'phase2_epoch_{epoch + 1}.pth'))

            # csv log
            with open(log_csv, 'a', newline='') as f:
                w = csv.writer(f)
                if not csv_header_written:
                    w.writerow([
                        'epoch', 'lr', 'pred_steps',
                        'train_loss', 'train_recon_3d', 'train_future_3d', 'train_latent', 'train_vel',
                        'val_loss',   'val_recon_3d',   'val_future_3d',   'val_latent',   'val_vel',
                    ])
                    csv_header_written = True
                w.writerow([
                    epoch + 1,
                    optimizer.param_groups[0]['lr'],
                    pred_steps,
                    train_stats['loss'], train_stats['recon_3d'], train_stats['future_3d'],
                    train_stats['latent'], train_stats['vel'],
                    val_stats['loss'],  val_stats['recon_3d'],  val_stats['future_3d'],
                    val_stats['latent'],  val_stats['vel'],
                ])

        scheduler.step()

    if rank == 0:
        writer.close()
        wandb.finish()
        print(f"Training completed. Best val future MPJPE: {best_val * 1000:.2f} mm")
        print(f"Best checkpoint: {os.path.join(args.checkpoint_dir, args.checkpoint_save_name)}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()