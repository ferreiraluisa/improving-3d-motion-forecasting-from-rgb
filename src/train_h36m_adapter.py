"""Train the COCO-17 -> H36M-17 adapter on Human3.6M WHAM features.

Uses the chunked loader (dataset/losd_data_h36.py): fixed-length clips, no
padding, no masking. Loss = MPJPE + bone-length. Evaluation follows the
standard per-action H36M protocol.

    python train_adapter_chunked.py \
        --dataset_path data/data_3d_h36m.npz \
        --dataset_2d_path data/data_2d_h36m_gt.npz \
        --features_path data/wham_feats_h36m.npy \
        --fit_linear --epochs 30 --wandb_project h36-adapter
"""

import argparse
import os
import time
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.dataloader import Human36mDataset
from dataset.losd_data_h36 import Fusion
from models.h36_adapter import Coco2H36mAdapter

# Must match SKELETON_3D in dataset/dataloader.py
BONES = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8),
         (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16)]

H36M_NAMES = ['pelvis', 'r_hip', 'r_knee', 'r_ank', 'l_hip', 'l_knee', 'l_ank',
              'spine', 'thorax', 'neck', 'head', 'l_sho', 'l_elb', 'l_wri',
              'r_sho', 'r_elb', 'r_wri']

ACTIONS = ['walking', 'eating', 'smoking', 'discussion', 'directions',
           'greeting', 'phoning', 'posing', 'purchases', 'sitting',
           'sittingdown', 'waiting', 'walkingtogether', 'takingphoto',
           'walkingdog']


# --------------------------------------------------------------------------- #
# np.load cache
# --------------------------------------------------------------------------- #
def patch_np_load_cache():
    """Fusion re-loads features_path in every instance. We build 16 of them
    (1 train + 15 per-action test), so memoise np.load by path."""
    cache = {}
    original = np.load

    def cached(path, *args, **kwargs):
        key = str(path)
        if key not in cache:
            cache[key] = original(path, *args, **kwargs)
        return cache[key]

    np.load = cached
    return cache


# --------------------------------------------------------------------------- #
# logging: train/loss and val/loss only
# --------------------------------------------------------------------------- #
class Logger:
    def __init__(self, opt):
        self.run = None
        if not opt.wandb_project:
            return
        try:
            import wandb
        except ImportError:
            print('[wandb] not installed, logging to stdout only')
            return
        self.wandb = wandb
        self.run = wandb.init(project=opt.wandb_project,
                              entity=opt.wandb_entity or None,
                              name=opt.wandb_name or None,
                              config=vars(opt))

    def log(self, data, step):
        if self.run is not None:
            self.wandb.log(data, step=step)

    def finish(self):
        if self.run is not None:
            self.run.finish()


# --------------------------------------------------------------------------- #
# args
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    # data
    p.add_argument('--dataset_path', type=str, default='data/data_3d_h36m.npz')
    p.add_argument('--dataset_2d_path', type=str, default='data/data_2d_h36m_gt.npz')
    p.add_argument('--features_path', type=str, required=True,
                   help='dict[subject][action] -> (n_cams, N, 563) WHAM features')
    p.add_argument('--subjects_train', type=str, default='S1,S5,S6,S7,S8')
    p.add_argument('--subjects_val', type=str, default='S9,S11')
    p.add_argument('--subjects_test', type=str, default='S9,S11')
    p.add_argument('--downsample', type=int, default=1, help='50fps -> 25fps')
    p.add_argument('--seq_len', type=int, default=81)
    p.add_argument('--mode', type=str, default='train')
    # model
    p.add_argument('--d_ctx', type=int, default=512)
    p.add_argument('--hidden', type=int, default=512)
    p.add_argument('--blocks', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--use_ctx', action='store_true')
    p.add_argument('--no_temporal', action='store_true')
    # optim
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--lr_A', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--clip', type=float, default=1.0)
    p.add_argument('--w_bone', type=float, default=1.0)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--log_every', type=int, default=50)
    # baseline
    p.add_argument('--fit_linear', action='store_true')
    p.add_argument('--ridge', type=float, default=1e-3)
    # wandb
    p.add_argument('--wandb_project', type=str, default='coco2h36m-adapter',
                   help='empty string disables wandb')
    p.add_argument('--wandb_entity', type=str, default='')
    p.add_argument('--wandb_name', type=str, default='')
    # misc
    p.add_argument('--ckpt', type=str, default='checkpoints/h36_adapter')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--eval_only', type=str, default='')

    opt = p.parse_args()

    # Attributes Fusion / ChunkedGenerator require.
    opt.stride = opt.seq_len          # non-overlapping clips: 1 pass per frame
    opt.out_all = True                # gt_3D must be the whole window, not 1 frame
    opt.img_size = [192, 256]
    # HARD OFF: get_batch flips gt_3D/2D but __getitem__ never flips feat_window.
    opt.data_augmentation = False
    opt.reverse_augmentation = False
    return opt


# --------------------------------------------------------------------------- #
# loss: MPJPE + bone length
# --------------------------------------------------------------------------- #
def bone_lengths(kp):
    i, j = zip(*BONES)
    return (kp[..., list(i), :] - kp[..., list(j), :]).norm(dim=-1)


def adapter_loss(pred, gt, w_bone, eps=1e-9):
    """pred/gt: (B, T, 17, 3) in metres. Returned loss is in metres."""
    # eps before sqrt: ||e|| has an infinite gradient at e == 0
    mpjpe = (pred - gt).pow(2).sum(-1).add(eps).sqrt().mean()
    total = mpjpe
    logs = {'mpjpe_mm': mpjpe.item() * 1000}

    if w_bone > 0:
        bone = (bone_lengths(pred) - bone_lengths(gt)).abs().mean()
        total = total + w_bone * bone
        logs['bone_mm'] = bone.item() * 1000

    logs['loss'] = total.item()
    return total, logs


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def build_loaders(opt):
    dataset = Human36mDataset(opt.dataset_path)

    train_set = Fusion(opt, dataset, train=True)
    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True,
                              num_workers=opt.workers, pin_memory=True,
                              drop_last=True)

    # get_action() has no empty-action guard here, so one dataset per action.
    test_loaders = {}
    for a in ACTIONS:
        s = Fusion(opt, dataset, train=False, mode_t='test', action=a)
        test_loaders[a] = DataLoader(s, batch_size=opt.batch_size, shuffle=False,
                                     num_workers=opt.workers, pin_memory=True)
    return train_loader, test_loaders


def unpack(batch, device, d_ctx):
    """-> feats (B,T,563), gt (B,T,17,3), meta"""
    cam, gt_3D, _, _, feats, action, subject, cam_ind, frame_idx, flip = batch
    feats = feats.to(device, non_blocking=True).float()
    gt = gt_3D.to(device, non_blocking=True).float()
    assert feats.shape[-1] == d_ctx + 51, \
        f'expected {d_ctx + 51}-d features, got {feats.shape[-1]}'
    return feats, gt, (action, subject, cam_ind, frame_idx)


@torch.no_grad()
def sanity_check(loader, d_ctx):
    """Catch unit / normalisation bugs before wasting an hour."""
    feats, gt, _ = unpack(next(iter(loader)), 'cpu', d_ctx)
    kp = feats[..., d_ctx:].reshape(-1, 17, 3)
    g = gt.reshape(-1, 17, 3)

    coco_span = (kp.max(1).values - kp.min(1).values).mean(0)
    gt_span = (g.max(1).values - g.min(1).values).mean(0)
    print(f'[sanity] coco bbox {[round(v, 3) for v in coco_span.tolist()]}  '
          f'mid-hip |r| = {kp[:, 11:13].mean(1).norm(dim=-1).mean():.4f}')
    print(f'[sanity] h36m bbox {[round(v, 3) for v in gt_span.tolist()]}  '
          f'pelvis |r| = {g[:, 0].norm(dim=-1).mean():.4f}  (should be ~0)')
    ratio = (gt_span.mean() / coco_span.mean()).item()
    print(f'[sanity] gt/coco scale ratio = {ratio:.3f}')
    if not 0.5 < ratio < 2.0:
        print('[sanity] WARNING: large scale gap — check metres vs mm, or whether '
              'your WHAM features are scale-normalised.')


# --------------------------------------------------------------------------- #
# closed-form linear baseline
# --------------------------------------------------------------------------- #
@torch.no_grad()
def fit_linear_A(loader, d_ctx, ridge, max_frames=300_000):
    """Ridge-fit the 17x17 regressor. One matrix shared by x/y/z => equivariant."""
    XtX = torch.zeros(17, 17, dtype=torch.float64)
    XtY = torch.zeros(17, 17, dtype=torch.float64)
    seen = 0

    for batch in loader:
        feats, gt, _ = unpack(batch, 'cpu', d_ctx)
        kp = feats[..., d_ctx:].reshape(-1, 17, 3)
        y = gt.reshape(-1, 17, 3)
        if seen + kp.shape[0] > max_frames:
            keep = torch.randperm(kp.shape[0])[:max_frames - seen]
            kp, y = kp[keep], y[keep]
        # the 3 axes are independent observations of the same 17x17 map
        Xf = kp.permute(0, 2, 1).reshape(-1, 17).double()
        Yf = y.permute(0, 2, 1).reshape(-1, 17).double()
        XtX += Xf.T @ Xf
        XtY += Xf.T @ Yf
        seen += kp.shape[0]
        if seen >= max_frames:
            break

    W = torch.linalg.solve(
        XtX + ridge * max(seen, 1) * torch.eye(17, dtype=torch.float64), XtY)
    A = W.T.float()                                     # y_i = sum_j A[i,j] * x_j

    ref = [[11, 12], [12], [14], [16], [11], [13], [15],
           [5, 6, 11, 12], [5, 6], [0, 1, 2], [0, 1, 2, 3, 4],
           [5], [7], [9], [6], [8], [10]]
    print(f'[linear] ridge fit on {seen} frames')
    print('[linear] dominant COCO source per H36M joint — verify joint ordering:')
    bad = 0
    for i in range(17):
        top = torch.argsort(A[i].abs(), descending=True)[:2]
        ok = int(top[0]) in ref[i]
        bad += (not ok)
        print(f'   {H36M_NAMES[i]:<7s} <- coco[{int(top[0]):2d}] {A[i, top[0]]:+.2f} , '
              f'coco[{int(top[1]):2d}] {A[i, top[1]]:+.2f}'
              f'{"" if ok else "   <-- UNEXPECTED"}')
    if bad > 3:
        print(f'[linear] {bad}/17 rows unexpected: your WHAM keypoints are probably '
              'NOT in standard COCO-17 order. Fix coco2h36m_init before training.')
    return A


# --------------------------------------------------------------------------- #
# eval
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, test_loaders, device, d_ctx, w_bone):
    """Val loss (same objective as training) + per-action / per-joint MPJPE."""
    model.eval()
    joint_sum = torch.zeros(17, device=device)
    n_total = 0
    loss_sum, loss_n = 0.0, 0
    per_action = {}

    for act, loader in test_loaders.items():
        act_sum, act_n = 0.0, 0
        seen = defaultdict(set)                      # (subj, action, cam) -> frames

        for batch in loader:
            feats, gt, (actions, subjects, cam_inds, frame_idx) = \
                unpack(batch, device, d_ctx)

            pred = model.forward_joints(feats)
            pred = pred - pred[..., :1, :]           # root-relative, like the GT

            # val loss: identical objective to training, over whole clips
            _, logs = adapter_loss(pred, gt, w_bone)
            loss_sum += logs['loss'] * feats.shape[0]
            loss_n += feats.shape[0]

            # MPJPE: chunk bounds are edge-padded, so drop frames already counted
            err = (pred - gt).norm(dim=-1) * 1000.0  # (B, T, 17) in mm
            keep = torch.zeros(frame_idx.shape, dtype=torch.bool)
            for b in range(frame_idx.shape[0]):
                s = seen[(subjects[b], actions[b], int(cam_inds[b]))]
                for t, f in enumerate(frame_idx[b].tolist()):
                    if f not in s:
                        s.add(f)
                        keep[b, t] = True

            sel = err[keep.to(device)]               # (n_valid, 17)
            joint_sum += sel.sum(0)
            act_sum += sel.mean(dim=-1).sum().item()
            act_n += sel.shape[0]
            n_total += sel.shape[0]

        per_action[act] = act_sum / max(act_n, 1)

    per_joint = (joint_sum / max(n_total, 1)).cpu()
    return {'loss': loss_sum / max(loss_n, 1),
            'mpjpe': float(np.mean(list(per_action.values()))),
            'per_joint': per_joint,
            'per_action': per_action,
            'n_frames': n_total}


def print_eval(tag, res):
    print(f'{tag} MPJPE = {res["mpjpe"]:.2f} mm  (action-averaged, '
          f'{res["n_frames"]} frames)')
    worst = torch.argsort(res['per_joint'], descending=True)[:5]
    print('   worst joints: ' + ', '.join(
        f'{H36M_NAMES[i]} {res["per_joint"][i]:.1f}' for i in worst))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    opt = parse_args()
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(os.path.dirname(opt.ckpt) or '.', exist_ok=True)

    patch_np_load_cache()
    logger = Logger(opt)

    train_loader, test_loaders = build_loaders(opt)
    sanity_check(train_loader, opt.d_ctx)

    model = Coco2H36mAdapter(d_ctx=opt.d_ctx, hidden=opt.hidden,
                             n_blocks=opt.blocks, dropout=opt.dropout,
                             use_ctx=opt.use_ctx,
                             temporal=not opt.no_temporal).to(device)
    print(f'params: {sum(p.numel() for p in model.parameters()):,}')

    if opt.eval_only:
        model.load_state_dict(torch.load(opt.eval_only, map_location=device)['model'])
        print_eval('[eval]', evaluate(model, test_loaders, device,
                                      opt.d_ctx, opt.w_bone))
        logger.finish()
        return

    # ---- baseline 1: analytic map (MLP head is zero-init, contributes 0) ---- #
    print_eval('[analytic init]',
               evaluate(model, test_loaders, device, opt.d_ctx, opt.w_bone))

    # ---- baseline 2: ridge-fitted linear map -------------------------------- #
    if opt.fit_linear:
        model.set_A(fit_linear_A(train_loader, opt.d_ctx, opt.ridge))
        print_eval('[ridge linear]',
                   evaluate(model, test_loaders, device, opt.d_ctx, opt.w_bone))
        print('   ^ if the trained MLP does not beat this by a useful margin, '
              'ship the 17x17 matrix and drop the MLP.')

    # ---- train -------------------------------------------------------------- #
    rest = [p for n, p in model.named_parameters() if n != 'A']
    optim = torch.optim.AdamW(
        [{'params': rest, 'lr': opt.lr},
         {'params': [model.A], 'lr': opt.lr_A, 'weight_decay': 0.0}],
        weight_decay=opt.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=opt.epochs)

    best, step = float('inf'), 0
    for epoch in range(opt.epochs):
        model.train()
        t0, agg, nb = time.time(), defaultdict(float), 0

        for batch in train_loader:
            feats, gt, _ = unpack(batch, device, opt.d_ctx)

            pred = model.forward_joints(feats)
            pred = pred - pred[..., :1, :]
            loss, logs = adapter_loss(pred, gt, opt.w_bone)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
            optim.step()

            for k, v in logs.items():
                agg[k] += v
            nb += 1
            step += 1

            if step % opt.log_every == 0:
                logger.log({'train/loss': logs['loss']}, step=step)

        sched.step()
        res = evaluate(model, test_loaders, device, opt.d_ctx, opt.w_bone)
        logger.log({'train/loss_epoch': agg['loss'] / max(nb, 1),
                    'val/loss': res['loss']}, step=step)

        msg = ' '.join(f'{k}={agg[k] / max(nb, 1):.4f}' for k in sorted(agg))
        print(f'[ep {epoch + 1:03d}] {msg} | val loss {res["loss"]:.4f} '
              f'| val MPJPE {res["mpjpe"]:.2f} mm | {time.time() - t0:.0f}s')

        if res['mpjpe'] < best:
            best = res['mpjpe']
            torch.save({'model': model.state_dict(), 'opt': vars(opt),
                        'epoch': epoch, 'mpjpe': best}, opt.ckpt + '_best.pth')

    print(f'\nbest val MPJPE {best:.2f} mm -> {opt.ckpt}_best.pth')
    model.load_state_dict(torch.load(opt.ckpt + '_best.pth',
                                     map_location=device)['model'])
    res = evaluate(model, test_loaders, device, opt.d_ctx, opt.w_bone)
    print_eval('[final]', res)
    for a in ACTIONS:
        print(f'   {a:<16s} {res["per_action"][a]:.2f}')
    logger.finish()


if __name__ == '__main__':
    main()