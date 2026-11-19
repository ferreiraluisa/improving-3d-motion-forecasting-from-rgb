import os
import sys
import argparse
import random
from datetime import datetime

import numpy as np
from tqdm import tqdm
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.distributed.elastic.multiprocessing.errors import record

# model
from models.motionst import (MotionTransformerST, random_joint_mask, random_frame_mask,
                             random_span_frame_mask)

# dataset
from dataset.losd_data_h36 import Fusion as FusionH36M
from dataset.losd_data_3dpw import Fusion3DPW
from dataset.dataloader import Human36mDataset, ThreeDPWDataset

# viz
from visualize.plot_3d import render_test_clips

# utils
from utils.mpjpe import mpjpe_3d, bone_length_loss, loss_velocity
from utils.config_tools import get_config as get_config_aux
from utils.utils import dtw_aligned_metrics

from src.train_phd_ddp2 import procrustes_align_batched


HORIZONS = [1, 5, 10, 20, 25]   # future frames to report at test time

SPLITS_3DPW = {'train': 'train', 'val': 'validation', 'test': 'test'}

H36M_ACTIONS = [
    "walking", "eating", "smoking", "discussion", "directions",
    "greeting", "phoning", "posing", "purchases", "sitting",
    "sittingdown", "takingphoto", "waiting", "walkingdog", "walkingtogether",
]


# ─────────────────────────── helpers ─────────────────────────────────────────

def is_main():
    return (not dist.is_initialized()) or dist.get_rank() == 0


def all_reduce_sum(value, device):
    """Sum a python number across ranks and return it as a float."""
    t = torch.tensor(float(value), device=device, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item()


def split_batch(data, n_past, device):
    """Split a full-window batch into past features + past/future GT joints.
    Works for both loaders: index 1 = gt_3D, index 3 = feat_window."""
    feat_window = data[3].to(device, non_blocking=True).float()   # (B, T, feat_dim)
    gt_3D = data[1].to(device, non_blocking=True).float()         # (B, T, J, 3)

    feat_past = feat_window[:, :n_past]
    gt_past = gt_3D[:, :n_past]
    gt_future = gt_3D[:, n_past:]
    return feat_past, gt_past, gt_future


def finite_diff(x):
    """x: (B, T, K, 3) -> (B, T-1, K, 3)"""
    return x[:, 1:] - x[:, :-1]


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


# ─────────────────────────── train / val / test ──────────────────────────────

def compute_losses(past_recon, future_pred, gt_past, gt_future, args,
                   past_mask=None, future_weight=1.0):
    """Past + future loss. Used for the main task AND for every aux task
    (all targets are ground truth). Returns (total_loss, parts dict).

    past_mask    : (B, T, K) bool. If given, the past term is ONLY the MPJPE on
                   the masked tokens (MAE-style aux loss: the task is "recover
                   what was hidden", not "re-copy what you can see").
    future_weight: extra scale on the future term (aux copies use
                   --aux_future_weight; 0 = aux tasks never train the future).
    """
    # ── past reconstruction losses ──
    l_past_3d = mpjpe_3d(past_recon, gt_past)
    l_past_bone = bone_length_loss(past_recon, gt_past)
    l_past_vel = loss_velocity(past_recon, gt_past)
    if past_mask is not None:
        err = (past_recon - gt_past).norm(dim=-1)                  # (B, T, K)
        l_past = err[past_mask].mean() if past_mask.any() else err.sum() * 0.0
    else:
        l_past_acc = loss_velocity(finite_diff(past_recon), finite_diff(gt_past))
        l_past = (l_past_3d + args.lambda_bone * l_past_bone
                  + args.lambda_vel * l_past_vel + args.lambda_acc * l_past_acc)

    # ── future prediction losses ──
    l_future_3d = mpjpe_3d(future_pred, gt_future)
    loss = l_past
    if future_weight > 0:
        l_future_bone = bone_length_loss(future_pred, gt_future)
        # include the last past frame so the past→future step is also checked
        fp = torch.cat([past_recon[:, -1:], future_pred], dim=1)
        fg = torch.cat([gt_past[:, -1:], gt_future], dim=1)
        l_future_vel = loss_velocity(fp, fg)
        l_future_acc = loss_velocity(finite_diff(fp), finite_diff(fg))
        l_future = (l_future_3d
                    + args.lambda_bone * l_future_bone
                    + args.lambda_future_vel * l_future_vel
                    + args.lambda_future_acc * l_future_acc)
        loss = loss + future_weight * args.lambda_future * l_future

    parts = dict(past_3d=l_past_3d, future_3d=l_future_3d,
                 past_bone=l_past_bone, past_vel=l_past_vel)
    return loss, parts


AUX_TASKS = ('joint', 'frame')


def active_aux_tasks(args):
    """Ordered list of enabled aux tasks, e.g. --aux_tasks joint,noise.
    'both' is kept as an alias of 'joint,frame'."""
    if not args.aux:
        return []
    spec = 'joint,frame' if args.aux_tasks == 'both' else args.aux_tasks
    tasks = [t.strip() for t in spec.split(',') if t.strip()]
    bad = [t for t in tasks if t not in AUX_TASKS]
    if bad or not tasks:
        raise ValueError(f'--aux_tasks: unknown {bad}; choose from {AUX_TASKS}')
    return tasks


def _sample_ratio(max_ratio, B, device, random_ratio):
    """Per-sample masking ratio: fixed, or U(0, max_ratio) if random_ratio."""
    if random_ratio:
        return torch.rand(B, device=device) * max_ratio
    return torch.full((B,), max_ratio, device=device)


def build_aux_batch(feat_past, args, device):
    """Stack [clean | <one corrupted copy per enabled aux task>] along the batch
    dim so all tasks go through ONE DDP forward (calling a DDP model several
    times before a single backward breaks gradient sync).

    joint : random (t, k) joints -> [MASK]
    frame : all joints of random past frames -> [MASK]
            --aux_frame_mode iid  : independent frames;  span : one contiguous block
            --aux_frame_mask_context : also hide the image context of those frames
    noise : Gaussian noise on the WHAM joint coords (root untouched), no mask.
            This matches the test-time situation (WHAM joints are noisy, not
            missing), unlike masking.
    Returns phi (n*B, T, F), joint_mask (n*B, T, K), per-copy masks list,
    context_mask (n*B, T) or None."""
    B, T, K = feat_past.size(0), args.past_length, args.num_joints
    phis = [feat_past]
    masks = [torch.zeros(B, T, K, dtype=torch.bool, device=device)]   # clean copy
    ctx = [torch.zeros(B, T, dtype=torch.bool, device=device)]
    for task in active_aux_tasks(args):
        c = torch.zeros(B, T, dtype=torch.bool, device=device)
        if task == 'joint':
            r = _sample_ratio(args.aux_joint_ratio, B, device, args.aux_random_ratio)
            m = random_joint_mask(B, T, K, r.view(B, 1, 1), device)
            phis.append(feat_past)
        elif task == 'frame':
            r = _sample_ratio(args.aux_frame_ratio, B, device, args.aux_random_ratio)
            if args.aux_frame_mode == 'span':
                f = random_span_frame_mask(B, T, r, device)
            else:
                f = random_frame_mask(B, T, r.view(B, 1), device)
            m = f[..., None].expand(-1, -1, K)
            if args.aux_frame_mask_context:
                c = f
            phis.append(feat_past)
        else:  # noise
            phi = feat_past.clone()
            coords = phi[..., -K * 3:].reshape(B, T, K, 3)
            noise = torch.randn_like(coords) * args.aux_noise_std
            noise[..., 0, :] = 0.0                     # keep root (model is root-relative)
            phi[..., -K * 3:] = (coords + noise).reshape(B, T, K * 3)
            m = torch.zeros(B, T, K, dtype=torch.bool, device=device)
            phis.append(phi)
        masks.append(m)
        ctx.append(c)

    phi = torch.cat(phis, dim=0)                    # (n*B, T, feat_dim)
    joint_mask = torch.cat(masks, dim=0)            # (n*B, T, K)
    context_mask = torch.cat(ctx, dim=0) if args.aux_frame_mask_context else None
    return phi, joint_mask, masks, context_mask


def aux_weight_at(base, epoch, args):
    """Aux-loss weight for this epoch. 'decay': linear base -> 0 over training
    (aux shapes the representation early, main task gets the last epochs)."""
    if args.aux_schedule == 'decay':
        return base * max(0.0, 1.0 - epoch / max(1, args.epochs - 1))
    return base


def train_epoch(model, args, dataloader, optimizer, epoch, device):
    model.train()

    tasks = active_aux_tasks(args)                  # e.g. [], ['frame'], ['joint', 'noise']
    aux_weight = {'joint': args.lambda_aux_joint, 'frame': args.lambda_aux_frame,
                  'noise': args.lambda_aux_noise}
    aux_weight = {t: aux_weight_at(w, epoch, args) for t, w in aux_weight.items()}

    sums = dict(loss=0.0, past_mpjpe=0.0, future_mpjpe=0.0, bone=0.0, vel=0.0)
    sums.update({f'aux_{t}_future': 0.0 for t in tasks})
    counter = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch + 1}', disable=not is_main())

    for data in pbar:
        batch_size = data[1].size(0)
        feat_past, gt_past, gt_future = split_batch(data, args.past_length, device)

        optimizer.zero_grad(set_to_none=True)

        if tasks:
            phi, joint_mask, copy_masks, context_mask = build_aux_batch(feat_past, args, device)
            past_all, future_all = model(phi, joint_mask=joint_mask, context_mask=context_mask)
            past_chunks = past_all.chunk(len(tasks) + 1, dim=0)
            future_chunks = future_all.chunk(len(tasks) + 1, dim=0)
            past_recon, future_pred = past_chunks[0], future_chunks[0]
        else:
            past_recon, future_pred = model(feat_past)

        # ── main task (clean input) ──
        loss, parts = compute_losses(past_recon, future_pred, gt_past, gt_future, args)

        # ── aux tasks (corrupted input, ground-truth targets) ──
        aux_parts = {}
        for i, t in enumerate(tasks, start=1):
            pm = copy_masks[i] if (args.aux_masked_only and t != 'noise') else None
            l_aux, aux_parts[t] = compute_losses(past_chunks[i], future_chunks[i],
                                                 gt_past, gt_future, args,
                                                 past_mask=pm,
                                                 future_weight=args.aux_future_weight)
            loss = loss + aux_weight[t] * l_aux

        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        sums['loss'] += loss.item() * batch_size
        sums['past_mpjpe'] += parts['past_3d'].item() * batch_size
        sums['future_mpjpe'] += parts['future_3d'].item() * batch_size
        sums['bone'] += parts['past_bone'].item() * batch_size
        sums['vel'] += parts['past_vel'].item() * batch_size
        for t in tasks:
            sums[f'aux_{t}_future'] += aux_parts[t]['future_3d'].item() * batch_size
        counter += batch_size

        postfix = {
            'loss':    f'{loss.item():.4f}',
            'past':    f'{parts["past_3d"].item():.4f}',
            'future':  f'{parts["future_3d"].item():.4f}',
            'bone':    f'{parts["past_bone"].item():.4f}',
            'LR':      f'{optimizer.param_groups[0]["lr"]:.2e}',
        }
        for t in tasks:
            postfix[f'{t[0]}m_fut'] = f'{aux_parts[t]["future_3d"].item():.4f}'
        pbar.set_postfix(postfix)

    n = max(all_reduce_sum(counter, device), 1.0)
    return {k: all_reduce_sum(v, device) / n for k, v in sums.items()}


@torch.no_grad()
def validate_epoch(model, args, dataloader, device):
    model.eval()
    total_past = 0.0
    total_future = 0.0
    counter = 0

    for data in dataloader:
        batch_size = data[1].size(0)
        feat_past, gt_past, gt_future = split_batch(data, args.past_length, device)
        past_recon, future_pred = model.module(feat_past)

        total_past += mpjpe_3d(past_recon, gt_past).item() * batch_size
        total_future += mpjpe_3d(future_pred, gt_future).item() * batch_size
        counter += batch_size

    n = max(all_reduce_sum(counter, device), 1.0)
    return all_reduce_sum(total_past, device) / n, all_reduce_sum(total_future, device) / n


@torch.no_grad()
def test_epoch(model, args, dataloader, device, act):
    model.eval()

    horizons = [h for h in HORIZONS if h <= args.future_length]
    total_past = 0.0
    horizon_totals = {h: 0.0 for h in horizons}
    dtw_pa_totals = {h: 0.0 for h in horizons}
    counter = 0

    # ── collapse diagnostics ──
    diag = dict(pred_fut=0.0, zero_vel=0.0,
                mot_pred_fut=0.0, mot_gt_fut=0.0,
                mot_pred_past=0.0, mot_gt_past=0.0)
    motion = lambda x: (x[:, 1:] - x[:, :-1]).norm(dim=-1).mean().item()

    for data in dataloader:
        batch_size = data[1].size(0)
        feat_past, gt_past, gt_future = split_batch(data, args.past_length, device)

        past_recon, future_pred = model.module(feat_past)
        total_past += mpjpe_3d(past_recon, gt_past).item() * batch_size

        zv = gt_past[:, -1:].expand_as(gt_future)               # zero-velocity baseline
        diag['pred_fut']      += mpjpe_3d(future_pred, gt_future).item() * batch_size
        diag['zero_vel']      += mpjpe_3d(zv, gt_future).item() * batch_size
        diag['mot_pred_fut']  += motion(future_pred) * batch_size
        diag['mot_gt_fut']    += motion(gt_future) * batch_size
        diag['mot_pred_past'] += motion(past_recon) * batch_size
        diag['mot_gt_past']   += motion(gt_past) * batch_size

        for h in horizons:
            horizon_totals[h] += (
                mpjpe_3d(future_pred[:, h - 1:h], gt_future[:, h - 1:h]).item() * batch_size)

            if args.skip_dtw:
                continue

            pred_h_aligned = procrustes_align_batched(future_pred[:, :h], gt_future[:, :h])
            dtw_sum = 0.0
            for b in range(batch_size):
                _, dtw_pa_seq = dtw_aligned_metrics(
                    pred_h_aligned[b].cpu().numpy(),
                    gt_future[b, :h].cpu().numpy(),
                )
                dtw_sum += float(dtw_pa_seq.mean())
            dtw_pa_totals[h] += dtw_sum

        counter += batch_size

    n = max(all_reduce_sum(counter, device), 1.0)
    avg_past_mm = all_reduce_sum(total_past, device) / n * 1000
    horizon_mm = {h: all_reduce_sum(horizon_totals[h], device) / n * 1000 for h in horizons}
    dtw_pa_mm = {h: all_reduce_sum(dtw_pa_totals[h], device) / n for h in horizons}
    d = {k: all_reduce_sum(v, device) / n * 1000 for k, v in diag.items()}   # mm

    if is_main():
        print(f"  [{act}]  past recon MPJPE: {avg_past_mm:.2f} mm")
        print(f"  [{act}]  future horizons  raw (mm): " +
              '  '.join(f's{h}={horizon_mm[h]:.2f}' for h in horizons))
        if not args.skip_dtw:
            print(f"  [{act}]  future horizons DTW+PA(mm): " +
                  '  '.join(f's{h}={dtw_pa_mm[h]:.2f}' for h in horizons))
        print(f"  [{act}]  DIAG future MPJPE pred={d['pred_fut']:.1f}  zero-vel={d['zero_vel']:.1f} mm")
        print(f"  [{act}]  DIAG motion/frame  future pred={d['mot_pred_fut']:.2f}  gt={d['mot_gt_fut']:.2f}"
              f" | past pred={d['mot_pred_past']:.2f}  gt={d['mot_gt_past']:.2f} mm")

    return avg_past_mm, horizon_mm, dtw_pa_mm


# ─────────────────────────── checkpoint helpers ──────────────────────────────

def save_checkpoint(model, optimizer, scheduler, epoch, loss, best_val_loss, path):
    if not is_main():
        return
    torch.save({
        'epoch': epoch,
        'wham': model.module.wham,
        'joints_only': model.module.joints_only,
        'context_only': model.module.context_only,
        'aux': model.module.aux,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'best_val_loss': best_val_loss,
    }, path)
    print(f"Checkpoint saved: {path}")


def load_model_weights(model, path):
    """Load a checkpoint into model.module, checking the wham flag matches."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    ckpt_wham = ckpt.get('wham')
    if ckpt_wham is not None and ckpt_wham != model.module.wham:
        raise ValueError(f"Checkpoint was trained with wham={ckpt_wham} but model has "
                         f"wham={model.module.wham}; {'add' if ckpt_wham else 'remove'} --wham.")
    ckpt_jo = ckpt.get('joints_only', False)        # old checkpoints: full WHAM
    if ckpt_jo != model.module.joints_only:
        raise ValueError(f"Checkpoint was trained with joints_only={ckpt_jo} but model has "
                         f"joints_only={model.module.joints_only}; "
                         f"{'add' if ckpt_jo else 'remove'} --joints_only.")
    ckpt_co = ckpt.get('context_only', False)       # old checkpoints: full WHAM
    if ckpt_co != model.module.context_only:
        raise ValueError(f"Checkpoint was trained with context_only={ckpt_co} but model has "
                         f"context_only={model.module.context_only}; "
                         f"{'add' if ckpt_co else 'remove'} --context_only.")
    sd =ckpt['model_state_dict']
    if any(k.startswith('module.') for k in sd):
        sd = {k[7:]: v for k, v in sd.items()}

    # joint_mask_token only exists when aux=True and is never used at test
    # time, so aux / non-aux checkpoints are interchangeable:
    #   aux ckpt  -> non-aux model : drop the token
    #   plain ckpt -> aux model    : token keeps its fresh init
    aux_only = ('joint_mask_token', 'context_mask_token')
    for k in aux_only:
        if not hasattr(model.module, k):
            sd.pop(k, None)
    missing, unexpected = model.module.load_state_dict(sd, strict=False)
    missing = [k for k in missing if k not in aux_only]
    if missing or unexpected:
        raise RuntimeError(f"State dict mismatch. missing={missing} unexpected={unexpected}")
    return ckpt


def load_checkpoint_for_resume(model, optimizer, scheduler, path):
    ckpt = load_model_weights(model, path)
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler and ckpt.get('scheduler_state_dict'):
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    best = ckpt.get('best_val_loss', float('inf'))
    print(f"Resumed from epoch {ckpt['epoch']} (best val future MPJPE: {best})")
    return ckpt['epoch'] + 1, best


# ─────────────────────────── args ────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser('MotionTransformerST training')

    # ── config ──
    p.add_argument('--config_path', type=str, default='config/MB_train_h36m_long.yaml')

    # ── H36M paths ──
    p.add_argument('--dataset_path', type=str, default='data/data_3d_h36m.npz')
    p.add_argument('--dataset_2d_path', type=str, default='data/data_2d_h36m_gt.npz')
    p.add_argument('--features_path', type=str, default=None,
                   help='H36M: .npy with dict[subject][action] -> features.')

    # ── 3DPW paths (splits are always train / validation / test, eval per video) ──
    p.add_argument('--dataset_3dpw_path', type=str, default=None,
                   help='3DPW meta file read by dataloader.ThreeDPWDataset.')
    p.add_argument('--dataset_3dpw_feats', type=str, default=None,
                   help='3DPW features .npy (--out of feats_3dpw.py).')
    p.add_argument('--img_dir_3dpw', type=str, default=None,
                   help='<3DPW root>/imageFiles (only needed for frame_info / viz).')

    # ── checkpoint / dirs ──
    p.add_argument('--checkpoint_dir', type=str, default='work_dir/checkpoints')
    p.add_argument('--checkpoint_save_name', type=str, default='best_st_model.pth')
    p.add_argument('--resume', type=str, default='')
    p.add_argument('--work_dir', type=str, default='./work_dir')
    p.add_argument('--video_dir', type=str, default='pic')

    # ── training hyper-params ──
    p.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-3)
    p.add_argument('--warmup_epochs', type=int, default=5)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--lambda_bone', type=float, default=1.0)
    p.add_argument('--lambda_vel', type=float, default=0.5)
    p.add_argument('--lambda_acc', type=float, default=0.5)
    p.add_argument('--lambda_future', type=float, default=1.0)
    p.add_argument('--lambda_future_vel', type=float, default=3.0)
    p.add_argument('--lambda_future_acc', type=float, default=1.0)
    p.add_argument('--save_freq', type=int, default=10)
    p.add_argument('--val_freq', type=int, default=1)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--workers_val', type=int, default=2)
    p.add_argument('--seed', type=int, default=1126)
    p.add_argument('--skip_dtw', action='store_true')

    # ── test videos (visualize/plot_3d.py) ──
    p.add_argument('--viz', action='store_true',
                   help='Test mode: render cropped video | 3D GT + recon/pred .mp4 clips.')
    p.add_argument('--viz_num', type=int, default=3, help='Clips per action / group.')
    p.add_argument('--viz_dir', type=str, default=None,
                   help='Output folder (default <work_dir>/<video_dir>).')
    p.add_argument('--viz_scale', type=float, default=1.2,
                   help='Crop side = scale * extent of 2D joints + head top / soles.')
    p.add_argument('--viz_crop', type=int, default=384, help='Crop size in pixels.')
    p.add_argument('--viz_fps', type=float, default=None,
                   help='Default: 50/downsample (H36M), 30/downsample (3DPW).')
    p.add_argument('--viz_elev', type=float, default=15.0)
    p.add_argument('--viz_azim', type=float, default=-70.0)
    p.add_argument('--h36m_root', type=str, default=None,
                   help='H36M root with <subject>/Videos/<action>.<cam_id>.mp4.')
    p.add_argument('--h36m_video_template', type=str,
                   default='{root}/{subject}/Videos/{action}.{cam_id}.mp4',
                   help='Fields: root, subject, action, cam_id, cam_ind.')

    # ── dataset ──
    p.add_argument('--dataset', type=str, default='h36m', choices=['h36m', '3dpw'])
    p.add_argument('--subjects_train', type=str, default='S1,S6,S7,S8,S9')
    p.add_argument('--subjects_val', type=str, default='S11')
    p.add_argument('--subjects_test', type=str, default='S5')
    p.add_argument('--data_augmentation', action='store_true')
    p.add_argument('--reverse_augmentation', action='store_true')
    p.add_argument('--downsample', type=int, default=2)
    p.add_argument('--stride', type=int, default=2)
    p.add_argument('--img_size', nargs=2, type=int, default=[192, 256])
    p.add_argument('--past_length', type=int, default=15)
    p.add_argument('--future_length', type=int, default=25)

    # ── model ──
    p.add_argument('--feat_dim', type=int, default=1024)
    p.add_argument('--wham', action='store_true',
                   help='Features are WHAM = [context | num_joints*3 joint coords]; '
                        'the model embeds the joints directly.')
    p.add_argument('--joints_only', action='store_true',
                   help='Two-stage baseline: with --wham, build the tokens from the WHAM '
                        'joint coords only and ignore the WHAM context vector.')
    p.add_argument('--context_only', action='store_true',
                   help='Image-only ablation: with --wham, drop the WHAM joint coords and '
                        'project the context vector linearly (same as the non-WHAM mode).')
    p.add_argument('--aux', action='store_true',
                   help='Train with AuxFormer-style auxiliary tasks (joint masking + '
                        'frame masking). Requires --wham. Training only.')
    p.add_argument('--aux_tasks', type=str, default='both',
                   help="Comma list of aux tasks from {joint, frame, noise}, e.g. "
                        "'frame,noise'. 'both' = 'joint,frame'.")
    p.add_argument('--lambda_aux_noise', type=float, default=1.0,
                   help='Weight of the denoising aux loss.')
    p.add_argument('--aux_noise_std', type=float, default=0.02,
                   help='Std (same units as the joints, m) of the noise in the denoising task.')
    p.add_argument('--aux_future_weight', type=float, default=1.0,
                   help='Scale of the FUTURE loss inside aux copies (0 = aux copies only '
                        'train past reconstruction).')
    p.add_argument('--aux_masked_only', action='store_true',
                   help='Masking tasks: past loss only on masked tokens. NOT recommended with '
                        'WHAM input: visible tokens are noisy, so recovering their GT is a real task.')
    p.add_argument('--aux_frame_mode', type=str, default='iid', choices=['iid', 'span'],
                   help='Frame masking: independent frames, or one contiguous block per sample.')
    p.add_argument('--aux_frame_mask_context', action='store_true',
                   help='Frame masking also hides the image context of masked frames '
                        '(pose must come from temporal neighbours only).')
    p.add_argument('--aux_random_ratio', action='store_true',
                   help='Sample the masking ratio per sample from U(0, ratio).')
    p.add_argument('--aux_schedule', type=str, default='const', choices=['const', 'decay'],
                   help="Aux weight: constant, or linear decay to 0 over training.")
    p.add_argument('--lambda_aux_joint', type=float, default=1.0,
                   help='Weight of the joint-masking aux loss.')
    p.add_argument('--lambda_aux_frame', type=float, default=1.0,
                   help='Weight of the frame-masking aux loss.')
    p.add_argument('--aux_joint_ratio', type=float, default=0.7,
                   help='Fraction of (t, k) joints masked in the joint-masking task.')
    p.add_argument('--aux_frame_ratio', type=float, default=0.5,
                   help='Fraction of past frames whose joints are masked in the frame-masking task.')
    p.add_argument('--num_joints', type=int, default=17)
    p.add_argument('--d_model', type=int, default=256)
    p.add_argument('--n_heads', type=int, default=4)
    p.add_argument('--num_layers', type=int, default=6)
    p.add_argument('--dim_ff', type=int, default=1024)
    p.add_argument('--dropout', type=float, default=0.1)

    # ── wandb ──
    p.add_argument('--wandb_project', type=str, default='motion_transformer_st')
    p.add_argument('--wandb_run_name', type=str, default=None)
    p.add_argument('--wandb_mode', type=str, default='online',
                   choices=['online', 'offline', 'disabled'])
    return p


def parse_args():
    """Precedence: command line > yaml > argparse defaults."""
    parser = build_parser()

    pre_args, _ = parser.parse_known_args()
    cfg = get_config_aux(pre_args.config_path)
    cfg = dict(cfg) if isinstance(cfg, dict) else dict(vars(cfg))

    known = {a.dest for a in parser._actions}
    parser.set_defaults(**{k: v for k, v in cfg.items() if k in known and k != 'config_path'})
    args = parser.parse_args()

    for k, v in cfg.items():
        if k not in known:
            setattr(args, k, v)

    # dataset-specific required paths
    if args.dataset == 'h36m':
        if args.features_path is None:
            raise ValueError('features_path is not set: pass --features_path or put it in the yaml.')
    else:
        missing = [n for n in ('dataset_3dpw_path', 'dataset_3dpw_feats') if not getattr(args, n)]
        if missing:
            raise ValueError(f'--dataset 3dpw needs: {", ".join("--" + m for m in missing)}')

    if args.aux and not args.wham:
        raise ValueError('--aux only works with WHAM features: add --wham.')
    if args.joints_only and not args.wham:
        raise ValueError('--joints_only only works with WHAM features: add --wham.')
    if args.context_only and not args.wham:
        raise ValueError('--context_only only works with WHAM features: add --wham.')
    if args.context_only and args.joints_only:
        raise ValueError('--context_only and --joints_only are mutually exclusive.')
    if args.context_only and args.aux:
        raise ValueError('--context_only has no joint tokens to mask: drop --aux.')
    if args.joints_only and getattr(args, 'aux_frame_mask_context', False):
        raise ValueError('--joints_only has no context to mask: drop --aux_frame_mask_context.')
    if 'frame' not in active_aux_tasks(args):   # also validates --aux_tasks
        if getattr(args, 'aux_frame_mask_context', False):
            # the [CTX_MASK] token would never be used -> DDP unused-param error
            raise ValueError('--aux_frame_mask_context needs frame in --aux_tasks.')

    if args.viz and args.mode == 'test':
        if args.dataset == 'h36m' and not args.h36m_root:
            raise ValueError('--viz on H36M needs --h36m_root (folder with <subject>/Videos/).')
        if args.dataset == '3dpw' and not args.img_dir_3dpw:
            raise ValueError('--viz on 3DPW needs --img_dir_3dpw (<3DPW root>/imageFiles).')
    if args.viz_dir is None:
        args.viz_dir = os.path.join(args.work_dir, args.video_dir)

    for k in ('subjects_train', 'subjects_val', 'subjects_test'):
        val = getattr(args, k)
        if isinstance(val, (list, tuple)):
            val = ','.join(val)
        setattr(args, k, ','.join(s.strip() for s in str(val).split(',') if s.strip()))

    args.seq_len = args.past_length + args.future_length
    args.frames = args.seq_len
    args.root_joint = 0
    args.out_all = True          # both loaders must return all T frames of gt_3D

    # fixed 3DPW settings (set after the yaml so an old config can't override them;
    # Fusion3DPW still reads these from args)
    args.split_train_3dpw = SPLITS_3DPW['train']
    args.split_val_3dpw = SPLITS_3DPW['val']
    args.split_test_3dpw = SPLITS_3DPW['test']
    args.normalize_2d_3dpw = True
    return args


# ─────────────────────────── main ────────────────────────────────────────────

@record
def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # ── DDP init ──
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=world_size, rank=rank,
                            device_id=torch.device(f'cuda:{local_rank}'))
    dist.barrier()
    device = torch.device(f'cuda:{local_rank}')

    if rank == 0:
        print(f"DDP initialised — world_size={world_size}", flush=True)
        print('\n── effective config ──')
        for k in sorted(vars(args)):
            print(f'  {k}: {getattr(args, k)}')
        print(f'  effective batch size: {args.batch_size * world_size}\n', flush=True)
    else:
        f = open(os.devnull, 'w')
        sys.stdout = f
        sys.stderr = f

    # ── dirs / logging ──
    writer = None
    if is_main():
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        os.makedirs(args.work_dir, exist_ok=True)
        log_dir = os.path.join(args.work_dir, 'logs', datetime.now().strftime('%Y%m%d_%H%M%S'))
        writer = SummaryWriter(log_dir)
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or datetime.now().strftime('%Y%m%d_%H%M%S'),
            config=vars(args),
            mode=args.wandb_mode,
        )

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
            prefetch_factor=2 if args.workers > 0 else None,
            drop_last=True)

        groups = eval_groups(args, dataset, 'val')
        val_loaders = {}
        for name, kw in groups:
            val_data = Fusion(args, dataset, train=False, mode_t='val', **kw)
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_data, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
            val_loaders[name] = torch.utils.data.DataLoader(
                val_data, batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers_val, pin_memory=True, sampler=val_sampler)
        probe_ds = train_data
    else:
        groups = eval_groups(args, dataset, 'test')
        test_loaders = {}
        test_sets = {}
        probe_ds = None
        for name, kw in groups:
            test_data = Fusion(args, dataset, train=False, mode_t='test', **kw)
            test_sets[name] = test_data
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                test_data, num_replicas=world_size, rank=rank, shuffle=False)
            test_loaders[name] = torch.utils.data.DataLoader(
                test_data, batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True, sampler=test_sampler)
            if probe_ds is None:
                probe_ds = test_data

    group_names = [g for g, _ in groups]

    # ── sanity-check shapes against the real data ──
    sample = probe_ds[0]
    real_feat_dim = int(sample[3].shape[-1])
    real_joints = int(sample[1].shape[-2])
    real_T = int(sample[1].shape[0])
    if real_feat_dim != args.feat_dim:
        print(f"[warn] --feat_dim={args.feat_dim} but features are {real_feat_dim}-d; using {real_feat_dim}.")
        args.feat_dim = real_feat_dim
    if real_joints != args.num_joints:
        print(f"[warn] --num_joints={args.num_joints} but GT has {real_joints}; using {real_joints}.")
        args.num_joints = real_joints
    assert real_T == args.seq_len, (
        f"Dataset returns {real_T} frames per window but past+future={args.seq_len}.")
    if args.wham:
        assert args.feat_dim > args.num_joints * 3, (
            f"--wham expects [context | {args.num_joints}*3 joints] features, "
            f"but features are only {args.feat_dim}-d. Wrong --features_path?")
        if is_main():
            print(f"WHAM features: context={args.feat_dim - args.num_joints * 3}-d "
                  f"+ joints={args.num_joints}x3"
                  + ("  [joints_only: context IGNORED]" if args.joints_only else "")
                  + ("  [context_only: joints IGNORED]" if args.context_only else ""))

    # ── model ──
    model = MotionTransformerST(
        feat_dim=args.feat_dim,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_ff,
        dropout=args.dropout,
        num_joints=args.num_joints,
        n_past=args.past_length,
        n_future=args.future_length,
        wham=args.wham,
        aux=args.aux,
        aux_mask_context=args.aux_frame_mask_context,
        joints_only=args.joints_only,
        context_only=args.context_only,
    ).to(device)

    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank,
        find_unused_parameters=False)

    if is_main():
        print(f"MotionTransformerST params: {sum(p.numel() for p in model.parameters()):,}"
              f"  (wham={args.wham}, joints_only={args.joints_only}, "
              f"context_only={args.context_only}, aux={args.aux})")

    # if args.mode == 'train':
    #     g = train_data.generator
    #     for i in range(0, len(g.pairs), 500):
    #         s = g.bounds_1[i] - g.pad
    #         n = g.poses_2d_GT[tuple((g.pairs[i][0], g.pairs[i][1], int(g.pairs[i][2])))].shape[0]
    #         assert s >= 0 and s + g.seq_len <= n, (i, s, n)
    #     print("no padded windows ✓")

    # ── optimizer + scheduler ──
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs)
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs - args.warmup_epochs), eta_min=args.lr * 0.01)
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[args.warmup_epochs])

    # ─────────── train ───────────
    if args.mode == 'train':
        start_epoch = 0
        best_val_loss = float('inf')

        if args.resume and os.path.exists(args.resume):
            print(f"Resuming from: {args.resume}")
            start_epoch, best_val_loss = load_checkpoint_for_resume(
                model, optimizer, scheduler, args.resume)

        for epoch in range(start_epoch, args.epochs):
            args.epoch = epoch
            train_sampler.set_epoch(epoch)
            print(f'\n=== Epoch {epoch + 1}/{args.epochs} ===')

            train_r = train_epoch(model, args, train_loader, optimizer, epoch, device)

            did_validate = (epoch + 1) % args.val_freq == 0
            avg_past_val = float('nan')
            avg_future_val = float('nan')
            if did_validate:
                p_sum = f_sum = 0.0
                for name in group_names:
                    p_, f_ = validate_epoch(model, args, val_loaders[name], device)
                    p_sum += p_
                    f_sum += f_
                avg_past_val = p_sum / len(group_names)
                avg_future_val = f_sum / len(group_names)

            print(f"Train | loss={train_r['loss']:.4f}  past={train_r['past_mpjpe']:.4f}"
                  f"  future={train_r['future_mpjpe']:.4f}  bone={train_r['bone']:.4f}"
                  f"  vel={train_r['vel']:.4f}")
            if args.aux:
                print("Aux   | " + "  ".join(
                    f"{t}-mask future={train_r[f'aux_{t}_future']:.4f}"
                    for t in active_aux_tasks(args)))
            if did_validate:
                print(f"Val   | past_MPJPE={avg_past_val * 1000:.2f} mm"
                      f"  future_MPJPE={avg_future_val * 1000:.2f} mm")

            if is_main():
                metrics = {
                    'Train/Loss': train_r['loss'],
                    'Train/Past_MPJPE': train_r['past_mpjpe'],
                    'Train/Future_MPJPE': train_r['future_mpjpe'],
                    'Train/BoneLoss': train_r['bone'],
                    'Train/VelLoss': train_r['vel'],
                    'Train/LR': optimizer.param_groups[0]['lr'],
                    'epoch': epoch,
                }
                for t in active_aux_tasks(args):
                    metrics[f'Train/Aux_{t.capitalize()}Mask_Future_MPJPE'] = train_r[f'aux_{t}_future']
                if did_validate:
                    metrics['Val/Past_MPJPE_m'] = avg_past_val
                    metrics['Val/Future_MPJPE_m'] = avg_future_val
                for k, v in metrics.items():
                    writer.add_scalar(k, v, epoch)
                wandb.log(metrics, step=epoch)

                if did_validate and avg_future_val < best_val_loss:
                    best_val_loss = avg_future_val
                    save_checkpoint(
                        model, optimizer, scheduler, epoch, avg_future_val, best_val_loss,
                        os.path.join(args.checkpoint_dir, args.checkpoint_save_name))
                    print(f"New best model — future MPJPE: {best_val_loss * 1000:.2f} mm")

            if (epoch + 1) % args.save_freq == 0:
                save_checkpoint(
                    model, optimizer, scheduler, epoch, train_r['loss'], best_val_loss,
                    os.path.join(args.checkpoint_dir, f'st_checkpoint_epoch_{epoch + 1}.pth'))

            scheduler.step()

        if is_main():
            writer.close()
            wandb.finish()
        dist.barrier()
        dist.destroy_process_group()
        print("Training completed!")

    # ─────────── test ───────────
    else:
        ckpt_path = os.path.join(args.checkpoint_dir, args.checkpoint_save_name)
        print(f"Loading checkpoint: {ckpt_path}")
        load_model_weights(model, ckpt_path)

        print("\nStarting inference...")
        horizons = [h for h in HORIZONS if h <= args.future_length]
        all_past_mm = []
        all_horizon = {h: [] for h in horizons}
        all_horizon_dtw_pa = {h: [] for h in horizons}

        for name in group_names:
            print('-' * 50)
            past_mm, horizon_mm, dtw_pa_mm = test_epoch(
                model, args, test_loaders[name], device, name)
            all_past_mm.append(past_mm)
            for h in horizons:
                all_horizon[h].append(horizon_mm[h])
                all_horizon_dtw_pa[h].append(dtw_pa_mm[h])

        if is_main():
            label = 'actions' if args.dataset == 'h36m' else 'groups'
            print('\n' + '=' * 60)
            print(f'SUMMARY [{args.dataset}] — average over all {label}')
            print('=' * 60)
            print(f"  Past reconstruction MPJPE : {np.mean(all_past_mm):.2f} mm")
            print("  Future horizons (mm)      : " +
                  '  '.join(f's{h}={np.mean(all_horizon[h]):.2f}' for h in horizons))
            if not args.skip_dtw:
                print("  Future horizons DTW+PA(mm): " +
                      '  '.join(f's{h}={np.mean(all_horizon_dtw_pa[h]):.2f}' for h in horizons))
            print('=' * 60)

        # ── videos: cropped RGB | 3D GT + past recon / future pred ──
        # groups are split round-robin over ranks so no rank sits in a long
        # barrier (NCCL timeout) while another renders everything
        if args.viz:
            my_groups = group_names[rank::world_size]
            print(f"\nRendering {args.viz_num} clip(s) per group -> {args.viz_dir}")
            for name in my_groups:
                try:
                    render_test_clips(model.module, test_sets[name], args, name,
                                      args.viz_dir, device, n=args.viz_num)
                except Exception as e:      # never lose the metrics over a plot
                    print(f"[viz] {name} failed: {type(e).__name__}: {e}", file=sys.__stdout__)
            dist.barrier()

        if writer is not None:
            writer.close()
        if is_main():
            wandb.finish()
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()