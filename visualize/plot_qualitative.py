"""
Qualitative figure: one test sample, GT (gray) + PHD (orange) + MotionST (blue),
predictions Procrustes + DTW aligned to the GT (same alignment as DTW-PA-MPJPE).

Saves, for one window of the test set:
  <out_dir>/<name>_pred_f01.png, _f05, _f10, _f20, _f25   (3D poses, white bg, no box)
  <out_dir>/<name>_rgb_f15.png                             (last observed frame, cropped)

Runs on a single GPU, no torchrun needed. Example (H36M):

  python plot_qualitative.py --config_path config/MB_train_h36m_long.yaml \
      --wham --features_path <wham_h36m_feats.npy> \
      --motionst_ckpt work_dir/checkpoints/best_st_model.pth \
      --phd_ckpt server_models/best_phd_model_phase2.pth \
      --phd_features_path <feats used by PHD, if different> \
      --group walking --sample_idx 120 --out_dir figs/qual --h36m_root <H36M root>

Same flags as train_motionst.py (yaml, --dataset, --wham, ...) plus the ones in
add_plot_args().
"""
import os
import copy
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

import src.train_motionst as T
from models.motionst import MotionTransformerST
import visualize.plot_3d as P3


PLOT_FRAMES = [1, 5, 10, 20, 25]          # predicted frames to save (1-based)

COLORS = dict(gt='#b5b5b5', phd='#E69F00', motionst='#0072B2')
WIDTHS = dict(gt=4.5, phd=2.2, motionst=2.2)
DRAW_ORDER = ('gt', 'phd', 'motionst')   # GT first = thick, in the background
LABELS = dict(gt='Ground truth', motionst='Our method', phd='PHD (baseline)')



# ─────────────────────────── args ────────────────────────────────────────────

def add_plot_args(p):
    p.add_argument('--motionst_ckpt', type=str, required=True)
    p.add_argument('--phd_ckpt', type=str, required=True)
    p.add_argument('--phd_features_path', type=str, default=None,
                   help='Features PHD was trained on (e.g. ResNet-50). Default: same as MotionST.')
    p.add_argument('--group', type=str, required=True,
                   help='H36M action (e.g. walking) or 3DPW sequence name.')
    p.add_argument('--sample_idx', type=int, default=0, help='Window index inside the group.')
    p.add_argument('--out_dir', type=str, default='figs/qualitative')
    p.add_argument('--name', type=str, default=None, help='File prefix (default <group>_<idx>).')
    p.add_argument('--dpi', type=int, default=300)
    p.add_argument('--align', type=str, default='raw', choices=['raw', 'dtw_pa'],
                   help='raw: predictions as output by the models (MPJPE view). '
                        'dtw_pa: Procrustes + DTW aligned to the GT (DTW-PA-MPJPE view).')
    p.add_argument('--no_legend', action='store_true', help='Do not draw the legend.')
    return p


def parse_args():
    """Reuse train_motionst.parse_args (yaml + checks) with the extra flags."""
    orig = T.build_parser
    T.build_parser = lambda: add_plot_args(orig())
    args = T.parse_args()
    T.build_parser = orig
    args.mode = 'test'
    return args


# ─────────────────────────── models ──────────────────────────────────────────

class _Wrap:
    """load_model_weights() expects a DDP-like object with .module."""
    def __init__(self, m):
        self.module = m


def infer_motionst_dims(args):
    """The checkpoint does not store d_model / dim_ff / num_layers, so read them
    from the weight shapes (n_heads cannot be inferred: it must match training)."""
    sd = torch.load(args.motionst_ckpt, map_location='cpu', weights_only=False)['model_state_dict']
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
    d_model = sd['pose_head.0.weight'].shape[0]
    dim_ff = sd['blocks.0.mlp_t.0.weight'].shape[0]
    num_layers = 1 + max(int(k.split('.')[1]) for k in sd if k.startswith('blocks.'))
    for k, v in dict(d_model=d_model, dim_ff=dim_ff, num_layers=num_layers).items():
        if getattr(args, k) != v:
            print(f'[ckpt] {k}: {getattr(args, k)} -> {v} (from checkpoint)')
            setattr(args, k, v)
    print(f'[ckpt] n_heads={args.n_heads} (not stored, must match training)')


def load_motionst(args, device):
    infer_motionst_dims(args)
    model = MotionTransformerST(
        feat_dim=args.feat_dim, d_model=args.d_model, n_heads=args.n_heads,
        num_layers=args.num_layers, dim_feedforward=args.dim_ff, dropout=args.dropout,
        num_joints=args.num_joints, n_past=args.past_length, n_future=args.future_length,
        wham=args.wham, aux=False, aux_mask_context=False, joints_only=args.joints_only,
    ).to(device)
    T.load_model_weights(_Wrap(model), args.motionst_ckpt)
    return model.eval()


def load_phd(args, feat_dim, device):
    """Phase-2 PHD checkpoint, built exactly as in train_phd_ddp2.py
    (latent_dim = feature dim, dropout 0, n_cond 15)."""
    from models.phd_model import PHDFor3DJoints
    model = PHDFor3DJoints(latent_dim=feat_dim, joints_num=args.num_joints,
                           dropout=0.0, n_cond=args.past_length).to(device)
    ckpt = torch.load(args.phd_ckpt, map_location='cpu', weights_only=False)
    sd = ckpt.get('model_state_dict', ckpt)
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
    model.load_state_dict(sd)
    return model.eval()


@torch.no_grad()
def phd_predict(model, feat_past, n_future):
    """-> (B, n_future, J, 3). predict_future only sees the n_cond observed
    frames and rolls out autoregressively from the last 13 strips; this is the
    same prediction as forward_phase2 with k_self = horizon used in test_epoch
    (check 3 in phd_model.check_causality)."""
    return model.predict_future(feat_past, pred_steps=n_future)


# ─────────────────────────── data ───────────────────────────────────────────

def build_test_set(args, dataset, Fusion):
    groups = dict(T.eval_groups(args, dataset, 'test'))
    if args.group not in groups:
        raise ValueError(f'--group {args.group!r} not found. Options: {sorted(groups)}')
    return Fusion(args, dataset, train=False, mode_t='test', **groups[args.group])


def args_with_features(args, feats):
    a = copy.copy(args)
    if args.dataset == 'h36m':
        a.features_path = feats
    else:
        a.dataset_3dpw_feats = feats
    return a


def get_frame_and_2d(test_set, sample, idx, t, args):
    """RGB image (H, W, 3) and 2D GT joints (J, 2) in pixels for frame t of
    window idx, found exactly as in visualize/plot_3d.render_test_clips.
    Called right after the sample is loaded: building a second Fusion (the PHD
    features) changes shared generator state, so test_set must not be indexed
    again afterwards."""
    J = int(np.asarray(sample[1]).shape[-2])
    if args.dataset == '3dpw':
        fi = test_set.frame_info(idx)
        if 'paths' not in fi:
            raise ValueError('set --img_dir_3dpw to <3DPW root>/imageFiles')
        if not fi['is_real'][t]:
            print(f'[warn] frame {t + 1} of this window is edge padding, not a real frame')
        img = P3.read_images([fi['paths'][t]])[0]
        uv = np.asarray(fi['joints2d_px'], np.float32)[t]
    else:
        action, subject = str(sample[4]), str(sample[5])
        cam_ind = int(P3._np(sample[6]))
        vid_id = int(P3._np(sample[7]).astype(int)[t] * args.downsample)
        path = P3.find_h36m_video(args.h36m_root, subject, action, cam_ind,
                                  args.h36m_video_template)
        img = P3.read_video_frames(path, [vid_id])[0]
        uv = P3._np(sample[2]).astype(np.float32).reshape(-1, J, 2)[t]
        if np.abs(uv).max() <= 2.0:                       # normalized screen coords
            h, w = img.shape[:2]
            uv = P3.denormalize_2d(uv, w, h)
    return img, uv


def crop_around_subject(img, uv, scale, size):
    """Same crop as the test videos: square box = scale * extent of the 2D
    joints + estimated head top / soles (black outside the image)."""
    J = uv.shape[0]
    box = P3.square_boxes(P3.extend_2d(uv[None], J), scale)[0]
    return Image.fromarray(P3.crop_square(img, *box, out=size))


# ─────────────────────────── alignment (DTW + PA) ────────────────────────────

def procrustes(pred, gt):
    """Similarity Procrustes (rotation + translation + uniform scale) of one
    pose (J, 3) onto gt (J, 3)."""
    mu_p, mu_g = pred.mean(0), gt.mean(0)
    p, g = pred - mu_p, gt - mu_g
    U, S, Vt = np.linalg.svd(p.T @ g)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = (S * np.diag(D)).sum() / (p ** 2).sum()
    return s * p @ R + mu_g


def dtw_path(cost):
    """Classic DTW on a (N, M) cost matrix -> list of (i, j) pairs."""
    N, M = cost.shape
    acc = np.full((N + 1, M + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            acc[i, j] = cost[i - 1, j - 1] + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
    i, j, path = N, M, []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step = np.argmin([acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1]])
        i, j = (i - 1, j - 1) if step == 0 else (i - 1, j) if step == 1 else (i, j - 1)
    return path[::-1]


def dtw_pa_align(pred, gt):
    """pred, gt: (F, J, 3). Every predicted frame is Procrustes-aligned to the
    GT, then the sequences are aligned in time with DTW. Returns (F, J, 3):
    for every GT frame j, the aligned predicted frame matched to it (if DTW
    matches several, the closest one).
    Note: the metric runs DTW on the prefix [:h] per horizon; for plotting we
    run it once on the full 25-frame window."""
    F = gt.shape[0]
    # PA of pred frame i onto gt frame j, for every pair (needed by DTW)
    pa = np.stack([[procrustes(pred[i], gt[j]) for j in range(F)] for i in range(F)])
    cost = np.linalg.norm(pa - gt[None], axis=-1).mean(-1)            # (F, F)
    out = np.empty_like(gt)
    path = dtw_path(cost)
    for j in range(F):
        cands = [i for i, jj in path if jj == j]
        i_best = min(cands, key=lambda i: cost[i, j])
        out[j] = pa[i_best, j]
    return out


# ─────────────────────────── plotting ────────────────────────────────────────

def to_plot(x):
    """camera coords (x right, y down, z forward) -> plot (x, z, -y), z up,
    same convention as the test videos."""
    return P3._to_plot(x)


def plot_poses(poses, parents, lims, args, path):
    """poses: dict name -> (J, 3) already in plot axes. The GT is drawn first,
    thick and light, so it shows around the predictions drawn on top of it."""
    fig = plt.figure(figsize=(4, 4), facecolor='white')
    ax = fig.add_subplot(111, projection='3d', facecolor='white')
    for name in (n for n in DRAW_ORDER if n in poses):
        P = poses[name]
        for j, pj in enumerate(parents):
            if pj < 0:
                continue
            ax.plot(*zip(P[j], P[pj]), color=COLORS[name], lw=WIDTHS[name],
                    solid_capstyle='round')
    (cx, cy, cz), r = lims
    ax.set_xlim(cx - r, cx + r)
    ax.set_ylim(cy - r, cy + r)
    ax.set_zlim(cz - r, cz + r)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=args.viz_elev, azim=args.viz_azim)
    ax.set_axis_off()                         # no box, no panes, no ticks
    def handle(k):
        return Line2D([0], [0], marker='o', linestyle='none', markersize=9,
                      markerfacecolor=COLORS[k], markeredgecolor='none', label=LABELS[k])
    handles = [handle(k) for k in ('gt', 'motionst', 'phd') if k in poses]
    if not args.no_legend:
        fig.legend(handles=handles, loc='upper right', ncol=1, frameon=True,
                   fancybox=True, framealpha=1.0, edgecolor='#d0d0d0',
                   fontsize=11, handlelength=1.6, handletextpad=0.6, borderpad=0.6, labelspacing=0.5)
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(path, dpi=args.dpi, facecolor='white', bbox_inches='tight', pad_inches=0)
    plt.close(fig)


# ─────────────────────────── main ────────────────────────────────────────────

@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    # ── data: same window for both models (same split / stride / group) ──
    Fusion, dataset = T.build_raw_dataset(args)
    test_set = build_test_set(args, dataset, Fusion)
    idx = args.sample_idx
    name = args.name or f'{args.group}_{idx}_{args.align}'
    sample = test_set[idx]
    args.feat_dim = int(sample[3].shape[-1])
    args.num_joints = int(sample[1].shape[-2])

    # last observed RGB frame: read NOW, before the PHD dataset is built
    t_last = args.past_length - 1               # frame 15 of the window
    img, uv = get_frame_and_2d(test_set, sample, idx, t_last, args)

    gt = torch.as_tensor(sample[1]).float()[None].to(device)           # (1, T, J, 3)
    feat_st = torch.as_tensor(sample[3]).float()[None, :args.past_length].to(device)

    if args.phd_features_path:
        a_phd = args_with_features(args, args.phd_features_path)
        F2, ds2 = T.build_raw_dataset(a_phd)
        s_phd = build_test_set(a_phd, ds2, F2)[idx]
        assert np.allclose(np.asarray(s_phd[1]), np.asarray(sample[1])), \
            'PHD and MotionST test sets return different windows for this index.'
        feat_phd = torch.as_tensor(s_phd[3]).float()[None, :args.past_length].to(device)
    else:
        feat_phd = feat_st

    # ── predictions ──
    motionst = load_motionst(args, device)
    _, pred_st = motionst(feat_st)                                     # (1, F, J, 3)
    phd = load_phd(args, int(feat_phd.shape[-1]), device)
    pred_phd = phd_predict(phd, feat_phd, args.future_length)

    gt_fut = gt[0, args.past_length:].cpu().numpy()
    al_st, al_phd = pred_st[0].cpu().numpy(), pred_phd[0].cpu().numpy()
    if args.align == 'dtw_pa':
        al_st = dtw_pa_align(al_st, gt_fut)
        al_phd = dtw_pa_align(al_phd, gt_fut)

    # ── 3D plots, identical limits in every png ──
    parents = P3.skeleton_parents(args.num_joints)
    idx = [f - 1 for f in PLOT_FRAMES]
    allp = np.concatenate([to_plot(x)
                           for x in [*gt_fut[idx], *al_st[idx], *al_phd[idx]]])
    r = 0.55 * (allp.max(0) - allp.min(0)).max()
    for f, i in zip(PLOT_FRAMES, idx):
        g = to_plot(gt_fut[i])
        poses = dict(gt=g,
                     phd=to_plot(al_phd[i]),
                     motionst=to_plot(al_st[i]))
        lims = (g[0], r)                        # centred on this frame's root
        plot_poses(poses, parents, lims, args,
                   os.path.join(args.out_dir, f'{name}_pred_f{f:02d}.png'))

    # ── last observed RGB frame, cropped around the subject ──
    crop_around_subject(img, np.asarray(uv), args.viz_scale, args.viz_crop).save(
        os.path.join(args.out_dir, f'{name}_rgb_f{t_last + 1:02d}.png'))

    print(f'Saved {len(PLOT_FRAMES) + 1} pngs to {args.out_dir}')


if __name__ == '__main__':
    main()
