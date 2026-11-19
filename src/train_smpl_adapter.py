"""Train the COCO-17 -> SMPL-24 adapter on 3DPW WHAM features.

Data (written by losd_data_3dpw.py --extractor wham):
    wham.npy       dict[split][segment] = [actor (n, 563)]   563 = 512 ctx + 17x3 COCO
    wham_meta.npy  dict[split][segment]['joints_cam'] = [actor (n, 24, 3)]  metres, camera

Each (segment, actor) is cut into fixed-length clips. Segments shorter than
--seq_len are edge-padded and the padded frames are masked out of loss and
metrics. Loss = MPJPE + bone-length on the SMPL kinematic tree, root-relative.
Eval: MPJPE + PA-MPJPE (24 SMPL joints), every frame counted once.

    python train_adapter_3dpw.py \
        --feats_path /media/data/ferreiral1/3DPW/feats/wham.npy \
        --meta_path  /media/data/ferreiral1/3DPW/feats/wham_meta.npy \
        --fit_linear --epochs 30 --wandb_project smpl-adapter-3dpw
"""

import argparse
import os
import time
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from models.smpl_adapter import Coco2SmplAdapter, SMPL_BONES, SMPL_NAMES

N_COCO, N_SMPL = 17, 24

# dominant COCO source(s) expected for each SMPL joint (sanity check of the ridge fit)
# COCO: 0 nose 1-2 eyes 3-4 ears 5/6 sho 7/8 elb 9/10 wri 11/12 hip 13/14 knee 15/16 ank
SMPL_REF = [[11, 12], [11], [12], [5, 6, 11, 12], [13], [14], [5, 6, 11, 12],
            [15], [16], [5, 6, 11, 12], [15, 13], [16, 14], [0, 1, 2, 3, 4, 5, 6],
            [5, 6, 0], [6, 5, 0], [0, 1, 2, 3, 4], [5], [6], [7], [8], [9], [10],
            [9, 7], [10, 8]]


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
    p.add_argument('--feats_path', type=str,
                   default='/media/data/ferreiral1/3DPW/feats/wham.npy')
    p.add_argument('--meta_path', type=str,
                   default='/media/data/ferreiral1/3DPW/feats/wham_meta.npy')
    p.add_argument('--split_train', type=str, default='train')
    p.add_argument('--split_val', type=str, default='validation')
    p.add_argument('--split_test', type=str, default='test')
    p.add_argument('--seq_len', type=int, default=81)
    p.add_argument('--train_stride', type=int, default=27,
                   help='clip stride on train (eval always uses seq_len)')
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
    p.add_argument('--warmup_steps', type=int, default=500,
                   help='linear LR warmup; Adam\'s first steps on the zero-init '
                        'head otherwise jump far from the init')
    p.add_argument('--w_bone', type=float, default=1.0)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--log_every', type=int, default=50)
    # baseline
    p.add_argument('--fit_linear', action='store_true')
    p.add_argument('--ridge', type=float, default=1e-3)
    # wandb
    p.add_argument('--wandb_project', type=str, default='coco2smpl-adapter-3dpw',
                   help='empty string disables wandb')
    p.add_argument('--wandb_entity', type=str, default='')
    p.add_argument('--wandb_name', type=str, default='')
    # misc
    p.add_argument('--ckpt', type=str, default='checkpoints/smpl_adapter_3dpw')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--eval_only', type=str, default='')
    return p.parse_args()


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_dict(path):
    d = np.load(path, allow_pickle=True)
    return d.item() if isinstance(d, np.ndarray) and d.dtype == object else d


class Clips3DPW(Dataset):
    """Fixed-length clips over every (segment, actor) of one split.

    item -> feats (T, 563), gt (T, 24, 3) root-relative, mask (T,) real frames,
            eval_mask (T,) real frames not already counted by a previous clip,
            track id (int)
    """

    def __init__(self, feats_all, meta_all, split, seq_len, stride, d_ctx):
        if split not in feats_all:
            raise KeyError(f'split "{split}" not in features (have {list(feats_all)})')
        self.T = seq_len
        self.tracks, self.names, self.clips = [], [], []
        d_in = d_ctx + N_COCO * 3

        for seg in sorted(feats_all[split]):
            persons = feats_all[split][seg]
            meta = meta_all[split][seg]
            for a, f in enumerate(persons):
                f = np.asarray(f, np.float32)
                gt = np.asarray(meta['joints_cam'][a], np.float32).reshape(-1, N_SMPL, 3)
                assert f.shape[-1] == d_in, \
                    f'{split}/{seg}: expected {d_in}-d features, got {f.shape[-1]}'
                assert f.shape[0] == gt.shape[0], \
                    f'{split}/{seg} actor {a}: {f.shape[0]} feats vs {gt.shape[0]} GT'
                gt = gt - gt[:, :1]                      # root-relative (pelvis)
                tid = len(self.tracks)
                self.tracks.append((f, gt))
                self.names.append(f'{seg}#{meta["actor_ids"][a]}')

                n = f.shape[0]
                starts = list(range(0, max(n - seq_len, 0) + 1, stride))
                if starts[-1] + seq_len < n:             # cover the tail
                    starts.append(n - seq_len)
                covered = 0
                for s in starts:
                    self.clips.append((tid, s, covered))
                    covered = min(s + seq_len, n)

        self.n_frames = sum(t[0].shape[0] for t in self.tracks)
        print(f'[data] {split}: {len(self.tracks)} tracks, {self.n_frames} frames, '
              f'{len(self.clips)} clips (T={seq_len}, stride={stride})')

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i):
        tid, s, covered = self.clips[i]
        f, gt = self.tracks[tid]
        n = f.shape[0]
        idx = np.arange(s, s + self.T)
        mask = idx < n
        eval_mask = mask & (idx >= covered)
        idx = np.minimum(idx, n - 1)                     # edge-pad short segments
        return (torch.from_numpy(f[idx]), torch.from_numpy(gt[idx]),
                torch.from_numpy(mask), torch.from_numpy(eval_mask), tid)


def build_loaders(opt):
    feats_all = load_dict(opt.feats_path)
    meta_all = load_dict(opt.meta_path)
    print(f'[data] splits in features: {list(feats_all)}')

    mk = lambda split, stride: Clips3DPW(feats_all, meta_all, split,
                                         opt.seq_len, stride, opt.d_ctx)
    train_set = mk(opt.split_train, opt.train_stride)
    val_set = mk(opt.split_val, opt.seq_len)
    test_set = mk(opt.split_test, opt.seq_len)

    kw = dict(batch_size=opt.batch_size, num_workers=opt.workers, pin_memory=True)
    return (DataLoader(train_set, shuffle=True, drop_last=True, **kw),
            DataLoader(val_set, shuffle=False, **kw),
            DataLoader(test_set, shuffle=False, **kw))


def unpack(batch, device):
    feats, gt, mask, eval_mask, tid = batch
    return (feats.to(device, non_blocking=True).float(),
            gt.to(device, non_blocking=True).float(),
            mask.to(device, non_blocking=True),
            eval_mask.to(device, non_blocking=True), tid)


# --------------------------------------------------------------------------- #
# loss: MPJPE + bone length (masked)
# --------------------------------------------------------------------------- #
def bone_lengths(kp):
    i, j = zip(*SMPL_BONES)
    return (kp[..., list(i), :] - kp[..., list(j), :]).norm(dim=-1)


def adapter_loss(pred, gt, mask, w_bone, eps=1e-9):
    """pred/gt: (B, T, 24, 3) metres, mask (B, T). Returned loss is in metres."""
    m = mask.float()
    denom = m.sum().clamp(min=1)
    # eps before sqrt: ||e|| has an infinite gradient at e == 0
    err = (pred - gt).pow(2).sum(-1).add(eps).sqrt().mean(-1)        # (B, T)
    mpjpe = (err * m).sum() / denom
    total = mpjpe
    logs = {'mpjpe_mm': mpjpe.item() * 1000}

    if w_bone > 0:
        bone = (bone_lengths(pred) - bone_lengths(gt)).abs().mean(-1)
        bone = (bone * m).sum() / denom
        total = total + w_bone * bone
        logs['bone_mm'] = bone.item() * 1000

    logs['loss'] = total.item()
    return total, logs


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def pa_align(pred, gt):
    """Procrustes-align pred to gt (scale, rotation, translation). (N, J, 3)."""
    X, Y = gt.double(), pred.double()
    muX, muY = X.mean(1, keepdim=True), Y.mean(1, keepdim=True)
    X0, Y0 = X - muX, Y - muY
    nX = X0.pow(2).sum((1, 2), keepdim=True).sqrt().clamp(min=1e-8)
    nY = Y0.pow(2).sum((1, 2), keepdim=True).sqrt().clamp(min=1e-8)
    X0, Y0 = X0 / nX, Y0 / nY
    U, s, Vh = torch.linalg.svd(X0.transpose(1, 2) @ Y0)
    V = Vh.transpose(1, 2)
    d = torch.sign(torch.linalg.det(V @ U.transpose(1, 2)))
    V = torch.cat([V[..., :2], V[..., 2:] * d[:, None, None]], -1)
    s = torch.cat([s[:, :2], s[:, 2:] * d[:, None]], -1)
    R = V @ U.transpose(1, 2)
    a = s.sum(-1)[:, None, None] * nX / nY
    t = muX - a * (muY @ R)
    return (a * (Y @ R) + t).to(pred.dtype)


@torch.no_grad()
def sanity_check(loader):
    """Catch unit / normalisation / frame bugs before wasting an hour."""
    feats, gt, mask, _, _ = unpack(next(iter(loader)), 'cpu')
    d_ctx = feats.shape[-1] - N_COCO * 3
    kp = feats[..., d_ctx:][mask].reshape(-1, N_COCO, 3)
    g = gt[mask].reshape(-1, N_SMPL, 3)

    coco_span = (kp.max(1).values - kp.min(1).values).mean(0)
    gt_span = (g.max(1).values - g.min(1).values).mean(0)
    print(f'[sanity] coco bbox {[round(v, 3) for v in coco_span.tolist()]}  '
          f'mid-hip |r| = {kp[:, 11:13].mean(1).norm(dim=-1).mean():.4f}')
    print(f'[sanity] smpl bbox {[round(v, 3) for v in gt_span.tolist()]}  '
          f'pelvis |r| = {g[:, 0].norm(dim=-1).mean():.4f}  (should be ~0)')
    ratio = (gt_span.mean() / coco_span.mean()).item()
    print(f'[sanity] gt/coco scale ratio = {ratio:.3f}')
    if not 0.5 < ratio < 2.0:
        print('[sanity] WARNING: large scale gap — check metres vs mm, or whether '
              'your WHAM keypoints are scale-normalised.')
    # per-axis extents should roughly agree if both live in the camera frame
    print('[sanity] if the per-axis bboxes disagree a lot (e.g. y tall in one, '
          'z tall in the other) the keypoints are not in camera coordinates.')


# --------------------------------------------------------------------------- #
# closed-form linear baseline
# --------------------------------------------------------------------------- #
@torch.no_grad()
def fit_linear_A(loader, d_ctx, ridge, max_frames=300_000):
    """Ridge-fit the 24x17 regressor. One matrix shared by x/y/z => equivariant.

    3DPW kps live in camera coords (depth varies a lot), so the fit is done on
    kps centred on their joint mean. Afterwards every row gets a constant added
    so it sums to 1: on centred data that changes nothing (the constant hits a
    zero-mean vector), and it makes A translation-equivariant, so the
    root-relative subtraction at eval removes the camera offset exactly.
    """
    XtX = torch.zeros(N_COCO, N_COCO, dtype=torch.float64)
    XtY = torch.zeros(N_COCO, N_SMPL, dtype=torch.float64)
    seen = 0

    for batch in loader:
        feats, gt, mask, _, _ = unpack(batch, 'cpu')
        kp = feats[..., d_ctx:][mask].reshape(-1, N_COCO, 3)
        kp = kp - kp.mean(1, keepdim=True)
        y = gt[mask].reshape(-1, N_SMPL, 3)
        if seen + kp.shape[0] > max_frames:
            keep = torch.randperm(kp.shape[0])[:max_frames - seen]
            kp, y = kp[keep], y[keep]
        # the 3 axes are independent observations of the same 24x17 map
        Xf = kp.permute(0, 2, 1).reshape(-1, N_COCO).double()
        Yf = y.permute(0, 2, 1).reshape(-1, N_SMPL).double()
        XtX += Xf.T @ Xf
        XtY += Xf.T @ Yf
        seen += kp.shape[0]
        if seen >= max_frames:
            break

    W = torch.linalg.solve(
        XtX + ridge * max(seen, 1) * torch.eye(N_COCO, dtype=torch.float64), XtY)
    A = W.T.float()                                     # y_i = sum_j A[i,j] * x_j
    A = A + (1.0 - A.sum(1, keepdim=True)) / N_COCO     # rows sum to 1

    print(f'[linear] ridge fit on {seen} frames')
    print('[linear] dominant COCO source per SMPL joint — verify joint ordering:')
    bad = 0
    for i in range(N_SMPL):
        top = torch.argsort(A[i].abs(), descending=True)[:2]
        ok = int(top[0]) in SMPL_REF[i]
        bad += (not ok)
        print(f'   {SMPL_NAMES[i]:<8s} <- coco[{int(top[0]):2d}] {A[i, top[0]]:+.2f} , '
              f'coco[{int(top[1]):2d}] {A[i, top[1]]:+.2f}'
              f'{"" if ok else "   <-- UNEXPECTED"}')
    if bad > 4:
        print(f'[linear] {bad}/24 rows unexpected: your WHAM keypoints are probably '
              'NOT in standard COCO-17 order. Fix coco2smpl_init before training.')
    return A


# --------------------------------------------------------------------------- #
# eval
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, loader, device, w_bone):
    """Val loss (same objective as training) + MPJPE / PA-MPJPE, each frame once."""
    model.eval()
    joint_sum = torch.zeros(N_SMPL, device=device, dtype=torch.float64)
    mpjpe_sum = pa_sum = 0.0
    n = 0
    loss_sum, loss_n = 0.0, 0

    for batch in loader:
        feats, gt, mask, eval_mask, _ = unpack(batch, device)

        pred = model.forward_joints(feats)
        pred = pred - pred[..., :1, :]               # root-relative, like the GT

        _, logs = adapter_loss(pred, gt, mask, w_bone)
        loss_sum += logs['loss'] * feats.shape[0]
        loss_n += feats.shape[0]

        p, g = pred[eval_mask], gt[eval_mask]        # (n_valid, 24, 3)
        if p.shape[0] == 0:
            continue
        err = (p - g).norm(dim=-1) * 1000.0          # mm
        pa = (pa_align(p, g) - g).norm(dim=-1) * 1000.0
        joint_sum += err.sum(0).double()
        mpjpe_sum += err.mean(-1).sum().item()
        pa_sum += pa.mean(-1).sum().item()
        n += p.shape[0]

    n_ = max(n, 1)
    return {'loss': loss_sum / max(loss_n, 1),
            'mpjpe': mpjpe_sum / n_,
            'pa_mpjpe': pa_sum / n_,
            'per_joint': (joint_sum / n_).float().cpu(),
            'n_frames': n}


def print_eval(tag, res):
    print(f'{tag} MPJPE = {res["mpjpe"]:.2f} mm | PA-MPJPE = {res["pa_mpjpe"]:.2f} mm'
          f'  ({res["n_frames"]} frames)')
    worst = torch.argsort(res['per_joint'], descending=True)[:5]
    print('   worst joints: ' + ', '.join(
        f'{SMPL_NAMES[i]} {res["per_joint"][i]:.1f}' for i in worst))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    opt = parse_args()
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(os.path.dirname(opt.ckpt) or '.', exist_ok=True)

    logger = Logger(opt)
    train_loader, val_loader, test_loader = build_loaders(opt)
    sanity_check(train_loader)

    model = Coco2SmplAdapter(d_ctx=opt.d_ctx, hidden=opt.hidden,
                             n_blocks=opt.blocks, dropout=opt.dropout,
                             use_ctx=opt.use_ctx,
                             temporal=not opt.no_temporal).to(device)
    print(f'params: {sum(p.numel() for p in model.parameters()):,}')

    if opt.eval_only:
        model.load_state_dict(torch.load(opt.eval_only, map_location=device)['model'])
        print_eval('[eval val]', evaluate(model, val_loader, device, opt.w_bone))
        print_eval('[eval test]', evaluate(model, test_loader, device, opt.w_bone))
        logger.finish()
        return

    # ---- baseline 1: analytic map (MLP head is zero-init, contributes 0) ---- #
    print_eval('[analytic init]', evaluate(model, val_loader, device, opt.w_bone))

    # ---- baseline 2: ridge-fitted linear map -------------------------------- #
    if opt.fit_linear:
        model.set_A(fit_linear_A(train_loader, opt.d_ctx, opt.ridge))
        print_eval('[ridge linear]', evaluate(model, val_loader, device, opt.w_bone))
        print('   ^ if the trained MLP does not beat this by a useful margin, '
              'ship the 24x17 matrix and drop the MLP.')

    # ---- train -------------------------------------------------------------- #
    rest = [p for n, p in model.named_parameters() if n != 'A']
    optim = torch.optim.AdamW(
        [{'params': rest, 'lr': opt.lr},
         {'params': [model.A], 'lr': opt.lr_A, 'weight_decay': 0.0}],
        weight_decay=opt.weight_decay)
    total = max(opt.epochs * len(train_loader), 1)
    warm = min(opt.warmup_steps, total // 2)

    def lr_lambda(t):                                 # per step: warmup + cosine
        if t < warm:
            return (t + 1) / warm
        return 0.5 * (1 + np.cos(np.pi * (t - warm) / max(total - warm, 1)))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    best, step = float('inf'), 0
    for epoch in range(opt.epochs):
        model.train()
        t0, agg, nb = time.time(), defaultdict(float), 0

        for batch in train_loader:
            feats, gt, mask, _, _ = unpack(batch, device)

            pred = model.forward_joints(feats)
            pred = pred - pred[..., :1, :]
            loss, logs = adapter_loss(pred, gt, mask, opt.w_bone)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
            optim.step()
            sched.step()

            for k, v in logs.items():
                agg[k] += v
            nb += 1
            step += 1

            if step % opt.log_every == 0:
                logger.log({'train/loss': logs['loss']}, step=step)

        res = evaluate(model, val_loader, device, opt.w_bone)
        logger.log({'train/loss_epoch': agg['loss'] / max(nb, 1),
                    'val/loss': res['loss']}, step=step)

        msg = ' '.join(f'{k}={agg[k] / max(nb, 1):.4f}' for k in sorted(agg))
        print(f'[ep {epoch + 1:03d}] {msg} | val loss {res["loss"]:.4f} '
              f'| val MPJPE {res["mpjpe"]:.2f} PA {res["pa_mpjpe"]:.2f} mm '
              f'| {time.time() - t0:.0f}s')

        if res['mpjpe'] < best:
            best = res['mpjpe']
            torch.save({'model': model.state_dict(), 'opt': vars(opt),
                        'epoch': epoch, 'mpjpe': best}, opt.ckpt + '_best.pth')

    print(f'\nbest val MPJPE {best:.2f} mm -> {opt.ckpt}_best.pth')
    model.load_state_dict(torch.load(opt.ckpt + '_best.pth',
                                     map_location=device)['model'])
    print_eval('[final val]', evaluate(model, val_loader, device, opt.w_bone))
    print_eval('[final test]', evaluate(model, test_loader, device, opt.w_bone))
    logger.finish()


if __name__ == '__main__':
    main()