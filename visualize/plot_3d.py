"""
Test-time videos for MotionTransformerST.

Each clip is one test window rendered as an .mp4:
    left  : the real video frame, cropped square around the subject
            (box = scale * extent of the 2D joints + estimated head top / soles,
             same rule as the 3DPW feature crop in feats_3dpw.py)
    right : 3D ground truth (black) with the model output overlaid:
            past reconstruction (blue) for the first n_past frames,
            future prediction (red) afterwards, plus per-frame MPJPE.

Frames come from:
    H36M : <h36m_root>/<subject>/Videos/<action>.<cam_id>.mp4
           video frame = frame_idx * downsample (frame_idx = item 7 of Fusion)
    3DPW : <img_dir_3dpw>/<source_seq>/image_XXXXX.jpg via Fusion3DPW.frame_info()

Entry point: render_test_clips(model, fusion_dataset, args, group, out_dir, device)
"""
import os

import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)


H36M_CAM_IDS = ['54138969', '55011271', '58860488', '60457274']
H36M_VIDEO_TEMPLATE = '{root}/{subject}/Videos/{action}.{cam_id}.mp4'

# 0 hip, 1 r_hip, 2 r_knee, 3 r_ankle, 4 l_hip, 5 l_knee, 6 l_ankle, 7 spine,
# 8 thorax, 9 neck/nose, 10 head top, 11-13 l_arm, 14-16 r_arm
H36M17_PARENTS = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
SMPL24_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12,
                  13, 14, 16, 17, 18, 19, 20, 21]

COL_GT, COL_PAST, COL_FUT = 'k', 'tab:blue', 'tab:red'


# ─────────────────────────── crop (same rule as feats_3dpw.py) ───────────────

def skeleton_parents(num_joints):
    if num_joints == 17:
        return H36M17_PARENTS
    if num_joints == 24:
        return SMPL24_PARENTS
    raise ValueError(f'no skeleton for {num_joints} joints')


def extend_2d(uv, num_joints, head_ext=1.0, foot_ext=0.25):
    """Add an estimated head top and soles so the box covers the whole body.
        head_top = head  + head_ext * (head  - neck)   (SMPL only, H36M has one)
        sole     = ankle + foot_ext * (ankle - knee)
    uv: (T, J, 2) pixels. Only used to place the crop box."""
    extra = []
    if num_joints == 24:          # SMPL: head 15, neck 12, ankles 7/8, knees 4/5
        extra.append(uv[:, 15] + head_ext * (uv[:, 15] - uv[:, 12]))
        pairs = [(7, 4), (8, 5)]
    elif num_joints == 17:        # H36M: head top 10 exists, ankles 3/6, knees 2/5
        pairs = [(3, 2), (6, 5)]
    else:
        return uv
    extra += [uv[:, a] + foot_ext * (uv[:, a] - uv[:, k]) for a, k in pairs]
    return np.concatenate([uv, np.stack(extra, 1)], 1)


def square_boxes(uv, scale):
    """uv: (T, K, 2) -> (T, 3) = (cx, cy, side); side = scale * max extent."""
    mn, mx = uv.min(1), uv.max(1)
    c = 0.5 * (mn + mx)
    side = scale * np.maximum(mx[:, 0] - mn[:, 0], mx[:, 1] - mn[:, 1])
    return np.concatenate([c, np.maximum(side, 1.0)[:, None]], 1)


def crop_square(img, cx, cy, side, out=384):
    """Square crop centred at (cx, cy); outside the image is black."""
    s = out / side
    M = np.array([[s, 0, out / 2 - s * cx],
                  [0, s, out / 2 - s * cy]], np.float32)
    return cv2.warpAffine(img, M, (out, out), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def denormalize_2d(x, w, h):
    """Inverse of normalize_screen_coordinates: X / w * 2 - [1, h / w]."""
    return (x + np.array([1.0, h / w], np.float32)) * w / 2.0


# ─────────────────────────── frame sources ───────────────────────────────────

def _h36m_action_variants(action):
    out = [action]
    for a, b in (('TakingPhoto', 'Photo'), ('Photo', 'TakingPhoto'),
                 ('WalkingDog', 'WalkDog'), ('WalkDog', 'WalkingDog')):
        if a in action:
            out.append(action.replace(a, b))
    return list(dict.fromkeys(out))


def find_h36m_video(root, subject, action, cam_ind, template=H36M_VIDEO_TEMPLATE):
    tried = []
    for act in _h36m_action_variants(action):
        p = template.format(root=root.rstrip('/'), subject=subject, action=act,
                            cam_id=H36M_CAM_IDS[cam_ind], cam_ind=cam_ind)
        if os.path.exists(p):
            return p
        tried.append(p)
    raise FileNotFoundError('No H36M video found, tried:\n  ' + '\n  '.join(tried))


def read_video_frames(path, frame_ids):
    """Seek once to the first wanted frame, then decode sequentially
    (per-frame seeking is unreliable on H.264). Returns RGB frames."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f'cannot open {path}')
    wanted = sorted(set(int(i) for i in frame_ids))
    cap.set(cv2.CAP_PROP_POS_FRAMES, wanted[0])
    got, want = {}, set(wanted)
    for pos in range(wanted[0], wanted[-1] + 1):
        ok, f = cap.read()
        if not ok:
            break
        if pos in want:
            got[pos] = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    cap.release()
    missing = [i for i in wanted if i not in got]
    if missing:
        raise RuntimeError(f'could not decode frames {missing[:5]} from {path}')
    return [got[int(i)] for i in frame_ids]


def read_images(paths):
    out = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            raise FileNotFoundError(p)
        out.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return out


# ─────────────────────────── rendering ───────────────────────────────────────

def _to_plot(p):
    """camera coords (x right, y down, z forward) -> plot (x, z, -y), z up."""
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], -1)


def _draw_skel(ax, pose, parents, color, lw=2.0):
    for j, p in enumerate(parents):
        if p >= 0:
            ax.plot(*[[pose[j, d], pose[p, d]] for d in range(3)], color=color, lw=lw)
    ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], color=color, s=6)


def render_clip(out_path, frames, uv_px, gt3d, pred3d, n_past, fps,
                scale=1.2, crop=384, title='', is_real=None,
                elev=15.0, azim=-70.0, head_ext=1.0, foot_ext=0.25):
    """
    frames : list of T RGB images (full video frames)
    uv_px  : (T, J, 2) 2D GT in pixels (used only for the crop box)
    gt3d   : (T, J, 3) root-relative GT, camera coords, metres
    pred3d : (T, J, 3) past recon (first n_past) + future prediction
    """
    T, J = gt3d.shape[:2]
    parents = skeleton_parents(J)
    boxes = square_boxes(extend_2d(uv_px, J, head_ext, foot_ext), scale)

    G, P = _to_plot(gt3d), _to_plot(pred3d)
    flat = G.reshape(-1, 3)
    mid = 0.5 * (flat.max(0) + flat.min(0))
    r = max(0.5 * (flat.max(0) - flat.min(0)).max() * 1.15, 0.5)
    err_mm = np.linalg.norm(gt3d - pred3d, axis=-1).mean(-1) * 1000

    legend = [Line2D([0], [0], color=COL_GT, lw=2, label='GT'),
              Line2D([0], [0], color=COL_PAST, lw=2, label='past recon'),
              Line2D([0], [0], color=COL_FUT, lw=2, label='future pred')]

    fig = plt.figure(figsize=(10, 5.2), dpi=100)
    ax_img = fig.add_axes([0.02, 0.03, 0.46, 0.84])
    ax3 = fig.add_axes([0.50, 0.02, 0.48, 0.88], projection='3d')
    writer = None
    try:
        for t in range(T):
            ax_img.cla()
            ax_img.imshow(crop_square(frames[t], *boxes[t], out=crop))
            ax_img.axis('off')
            phase = f'past {t + 1}/{n_past}' if t < n_past else f'future +{t - n_past + 1}'
            if is_real is not None and not is_real[t]:
                phase += '  [edge pad]'
            ax_img.set_title(phase, color=COL_PAST if t < n_past else COL_FUT)

            ax3.cla()
            _draw_skel(ax3, G[t], parents, COL_GT, lw=2.5)
            _draw_skel(ax3, P[t], parents, COL_PAST if t < n_past else COL_FUT, lw=1.8)
            ax3.set_xlim(mid[0] - r, mid[0] + r)
            ax3.set_ylim(mid[1] - r, mid[1] + r)
            ax3.set_zlim(mid[2] - r, mid[2] + r)
            ax3.set_box_aspect((1, 1, 1))
            ax3.view_init(elev=elev, azim=azim)
            ax3.set_xticklabels([]); ax3.set_yticklabels([]); ax3.set_zticklabels([])
            ax3.set_title(f'MPJPE {err_mm[t]:.1f} mm')
            ax3.legend(handles=legend, loc='upper right', fontsize=8)
            fig.suptitle(title, fontsize=10)

            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            if writer is None:
                h, w = buf.shape[:2]
                writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                         float(fps), (w, h))
            writer.write(cv2.cvtColor(buf, cv2.COLOR_RGB2BGR))
    finally:
        if writer is not None:
            writer.release()
        plt.close(fig)
    return out_path


# ─────────────────────────── entry point for the test loop ───────────────────

def _np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


@torch.no_grad()
def render_test_clips(model, ds, args, group, out_dir, device, n=3):
    """
    Render `n` evenly spaced test windows of one group (H36M action / 3DPW group).
    model : the unwrapped model (model.module under DDP)
    ds    : the Fusion / Fusion3DPW test dataset of that group
    """
    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    idxs = np.unique(np.linspace(0, len(ds) - 1, max(1, n)).round().astype(int))
    paths = []

    for idx in idxs:
        idx = int(idx)
        sample = ds[idx]
        gt3d = _np(sample[1]).astype(np.float32)
        feat = torch.as_tensor(_np(sample[3])).float()[None, :args.past_length].to(device)
        past, fut = model(feat)
        pred = torch.cat([past, fut], 1)[0].float().cpu().numpy()
        if pred.shape != gt3d.shape:
            print(f'[viz] skip {group}#{idx}: pred {pred.shape} vs gt {gt3d.shape}')
            continue

        try:
            if args.dataset == '3dpw':
                fi = ds.frame_info(idx)
                if 'paths' not in fi:
                    raise ValueError('set --img_dir_3dpw to <3DPW root>/imageFiles')
                frames = read_images(fi['paths'])
                uv = np.asarray(fi['joints2d_px'], np.float32)
                is_real = fi['is_real']
                name = f"{fi['source_seq']}_a{fi['actor_id']}_img{int(fi['img_idx'][0]):05d}"
                title = f"{fi['source_seq']}  actor {fi['actor_id']}  images {int(fi['img_idx'][0])}..{int(fi['img_idx'][-1])}"
                fps = 30.0 / args.downsample
            else:
                action, subject = str(sample[4]), str(sample[5])
                cam_ind = int(_np(sample[6]))
                vid_ids = (_np(sample[7]).astype(int) * args.downsample)
                path = find_h36m_video(args.h36m_root, subject, action, cam_ind,
                                       args.h36m_video_template)
                frames = read_video_frames(path, vid_ids)
                uv = _np(sample[2]).astype(np.float32).reshape(gt3d.shape[0], -1, 2)
                if np.abs(uv).max() <= 2.0:               # normalized screen coords
                    h, w = frames[0].shape[:2]
                    uv = denormalize_2d(uv, w, h)
                is_real = None
                name = f'{subject}_{action}_c{cam_ind}_f{int(vid_ids[0]):05d}'.replace(' ', '')
                title = f'{subject}  {action}  cam {H36M_CAM_IDS[cam_ind]}  frames {int(vid_ids[0])}..{int(vid_ids[-1])}'
                fps = 50.0 / args.downsample
        except (FileNotFoundError, RuntimeError, ValueError) as e:
            print(f'[viz] skip {group}#{idx}: {e}')
            continue

        out = os.path.join(out_dir, f'{group}_{name}.mp4')
        render_clip(out, frames, uv, gt3d, pred, args.past_length, args.viz_fps or fps,
                    scale=args.viz_scale, crop=args.viz_crop, title=title,
                    is_real=is_real, elev=args.viz_elev, azim=args.viz_azim)
        print(f'[viz] saved {out}')
        paths.append(out)
    return paths