"""
Save a batch of examples returned by Fusion.__getitem__ as one tiled MP4.

Each sample renders as a pair of panels:

    [ video frame + input_2D_GT ]  [ gt_3D skeleton, camera space ]

Reads directly from the H36M .mp4 files (no extracted frames needed).
Samples whose window runs off the start/end of the sequence get edge-padded
frame indices (repeats); those are skipped automatically.

Requires matplotlib for the 3D panel.

Usage:
    python visualize_clip.py --num_samples 4 --out ./tests/batch.mp4
"""

import argparse
import math
import os

import cv2
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers '3d')

from dataset.dataloader import Human36mDataset
from dataset.losd_data_h36 import Fusion


# =========================================================================
# H36M 17-joint skeleton
# =========================================================================

SKELETON = [
    [0, 1], [1, 2], [2, 3],
    [0, 4], [4, 5], [5, 6],
    [0, 7],
    [7, 8], [8, 9], [9, 10],
    [8, 11], [11, 12], [12, 13],
    [8, 14], [14, 15], [15, 16],
]

LEFT = {(0, 4), (4, 5), (5, 6), (8, 11), (11, 12), (12, 13)}
RIGHT = {(0, 1), (1, 2), (2, 3), (8, 14), (14, 15), (15, 16)}

# BGR for OpenCV, RGB (0-1) for matplotlib
BGR_LEFT, BGR_RIGHT, BGR_MID = (255, 191, 0), (0, 0, 255), (255, 255, 255)
RGB_LEFT, RGB_RIGHT, RGB_MID = (0.0, 0.75, 1.0), (1.0, 0.0, 0.0), (0.35, 0.35, 0.35)

# Camera order used by VideoPose3D / h36m_cameras_intrinsic_params.
CAM_IDS = ['54138969', '55011271', '58860488', '60457274']

VIDEO_TEMPLATE = "{root}/{subject}/Videos/{action}.{cam_id}.mp4"


# =========================================================================
# Video reading
# =========================================================================

class VideoReader:
    """Opens an H36M video once and pulls out an arbitrary set of frames."""

    def __init__(self, root, subject, action, cam_ind):
        self.path = VIDEO_TEMPLATE.format(
            root=root.rstrip('/'),
            subject=subject,
            action=action,
            cam_id=CAM_IDS[cam_ind],
        )

        if not os.path.exists(self.path):
            raise FileNotFoundError(
                f"No video at:\n{self.path}\n\n"
                f"Check VIDEO_TEMPLATE / CAM_IDS / --frames_root."
            )

        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open {self.path}")

        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def read_frames(self, indices):
        """
        Read the given ascending absolute frame numbers.

        Seeks once to the first frame then decodes sequentially, which is far
        more reliable than seeking per frame on H.264.
        """
        indices = list(indices)
        first, last = indices[0], indices[-1]

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, first)

        wanted = set(indices)
        grabbed = {}

        for pos in range(first, last + 1):
            ok, frame = self.cap.read()
            if not ok:
                break
            if pos in wanted:
                grabbed[pos] = frame

        missing = [i for i in indices if i not in grabbed]
        if missing:
            raise RuntimeError(
                f"Could not decode frames {missing[:5]}... from {self.path} "
                f"(video reports {self.n_frames} frames)."
            )

        return [grabbed[i] for i in indices]

    def release(self):
        self.cap.release()


# =========================================================================
# 2D drawing
# =========================================================================

def denormalize_2d(kps, res_w, res_h):
    """Inverse of normalize_screen_coordinates."""
    return (kps + np.array([1.0, res_h / res_w])) * res_w / 2.0


def looks_normalized(kps):
    return np.abs(kps).max() < 2.0


def draw_skeleton_2d(frame, kps, lw=2, radius=3):
    for j1, j2 in SKELETON:
        p1 = tuple(np.round(kps[j1]).astype(int))
        p2 = tuple(np.round(kps[j2]).astype(int))

        if (j1, j2) in LEFT:
            color = BGR_LEFT
        elif (j1, j2) in RIGHT:
            color = BGR_RIGHT
        else:
            color = BGR_MID

        cv2.line(frame, p1, p2, color, lw, cv2.LINE_AA)

    for x, y in kps:
        cv2.circle(frame, (int(round(x)), int(round(y))),
                   radius, (255, 255, 255), -1, cv2.LINE_AA)

    return frame


def label(frame, text, y=28):
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


# =========================================================================
# 3D rendering
# =========================================================================

class Skeleton3DRenderer:
    """
    Renders a root-relative H36M pose to a BGR image of size (size, size).

    The figure and the Line3D objects are built once and updated per frame,
    which is roughly an order of magnitude faster than redrawing.

    H36M camera space is x right, y down, z forward, so the pose is plotted
    as (x, z, -y) to come out upright.
    """

    def __init__(self, size=384, radius=0.85, elev=12.0, azim=-70.0, dpi=100):
        self.size = size
        self.radius = radius

        self.fig = plt.figure(figsize=(size / dpi, size / dpi), dpi=dpi)
        self.fig.patch.set_facecolor('white')

        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.view_init(elev=elev, azim=azim)
        self.ax.set_xlim(-radius, radius)
        self.ax.set_ylim(-radius, radius)
        self.ax.set_zlim(-radius, radius)
        try:
            self.ax.set_box_aspect([1, 1, 1])
        except AttributeError:
            pass  # matplotlib < 3.3

        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.set_zticklabels([])
        self.ax.grid(True)

        self.lines = []
        for j1, j2 in SKELETON:
            if (j1, j2) in LEFT:
                c = RGB_LEFT
            elif (j1, j2) in RIGHT:
                c = RGB_RIGHT
            else:
                c = RGB_MID
            ln, = self.ax.plot([0, 0], [0, 0], [0, 0], color=c, lw=2.5)
            self.lines.append((ln, j1, j2))

        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    def render(self, pos):
        """pos: (17, 3) root-relative camera-space joints, in metres."""
        for ln, j1, j2 in self.lines:
            ln.set_data([pos[j1, 0], pos[j2, 0]],
                        [pos[j1, 2], pos[j2, 2]])
            ln.set_3d_properties([-pos[j1, 1], -pos[j2, 1]])

        self.fig.canvas.draw()
        rgba = np.asarray(self.fig.canvas.buffer_rgba())
        bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)

        if bgr.shape[0] != self.size or bgr.shape[1] != self.size:
            bgr = cv2.resize(bgr, (self.size, self.size))
        return bgr

    def close(self):
        plt.close(self.fig)


# =========================================================================
# Sample selection
# =========================================================================

def unpack(item):
    """Keep this in sync with the return order of Fusion.__getitem__."""
    (cam, gt_3D, input_2D, input_2D_GT, feat,
     action, subject, cam_ind, frame_idx, flip) = item
    return dict(
        cam=cam, gt_3D=gt_3D, input_2D=input_2D, input_2D_GT=input_2D_GT,
        feat=feat, action=action, subject=subject,
        cam_ind=int(cam_ind), frame_idx=np.asarray(frame_idx), flip=bool(flip),
    )


def is_clean(frame_idx):
    """True when the window was not edge-padded: strictly increasing by 1."""
    return frame_idx.size > 1 and np.all(np.diff(frame_idx) == 1)


def find_clean_samples(fusion, n_wanted, seed=0):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(fusion))

    found = []
    for idx in order:
        s = unpack(fusion[int(idx)])

        if s['flip'] or not is_clean(s['frame_idx']):
            continue

        found.append((int(idx), s))
        if len(found) == n_wanted:
            break

    if not found:
        raise RuntimeError(
            "No unclipped windows found. seq_len may exceed the sequence "
            "length, or reverse augmentation is on."
        )

    if len(found) < n_wanted:
        print(f"Only found {len(found)} clean samples (wanted {n_wanted}).")

    return found


def as_pose_sequence(gt_3D, T):
    """Normalise gt_3D to (T, 17, 3), holding the last pose if it is shorter."""
    p = np.asarray(gt_3D)
    p = p.reshape(-1, p.shape[-2], p.shape[-1])   # (?, 17, 3)

    if p.shape[0] == T:
        return p
    if p.shape[0] == 1:                            # out_all=False: centre only
        return np.repeat(p, T, axis=0)

    print(f"  gt_3D has {p.shape[0]} frames but the 2D window has {T}; "
          f"holding the last pose.")
    pad = np.repeat(p[-1:], max(0, T - p.shape[0]), axis=0)
    return np.concatenate([p, pad], axis=0)[:T]


# =========================================================================
# Main
# =========================================================================

def build_args():
    p = argparse.ArgumentParser()

    p.add_argument('--dataset_path',    type=str, default='/media/data/ferreiral1/data_3d_h36m.npz')
    p.add_argument('--dataset_2d_path', type=str, default='/media/data/ferreiral1/data_2d_h36m_gt.npz')
    p.add_argument('--features_path',   type=str, default='/media/data/ferreiral1/hmr_resnet/feats.npy')
    p.add_argument('--frames_root',     type=str, default='/home/group-cvg/datasets/h36/h36/')

    p.add_argument('--out',         type=str, default='./tests/batch.mp4')
    p.add_argument('--num_samples', type=int, default=4)
    p.add_argument('--tile',        type=int, default=384)
    p.add_argument('--fps',         type=int, default=10)
    p.add_argument('--seed',        type=int, default=0)
    p.add_argument('--elev',        type=float, default=12.0)
    p.add_argument('--azim',        type=float, default=-70.0)
    p.add_argument('--radius',      type=float, default=0.85,
                   help='half-extent of the 3D axes, in metres')

    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--downsample', type=int, default=2)
    p.add_argument('--stride',     type=int, default=2)
    p.add_argument('--data_augmentation',    type=bool, default=False)
    p.add_argument('--reverse_augmentation', type=bool, default=False)
    p.add_argument('--img_size', nargs=2, type=int, default=[192, 256])

    a = p.parse_args()

    a.past_length = 15
    a.future_length = 25
    a.seq_len = a.past_length + a.future_length
    a.frames = a.seq_len
    a.root_joint = 0
    a.num_joints = 17
    a.mode = 'test'
    a.action = 'walking'
    a.subjects_train = 'S1,S5,S6,S7,S8'
    a.subjects_val = 'S9'
    a.subjects_test = 'S11'
    return a


def main():
    args = build_args()

    dataset = Human36mDataset(args.dataset_path)
    fusion = Fusion(args, dataset, train=False, mode_t='test', action=args.action)

    samples = find_clean_samples(fusion, args.num_samples, args.seed)

    renderer = Skeleton3DRenderer(size=args.tile, radius=args.radius,
                                  elev=args.elev, azim=args.azim)

    cells = []      # one entry per sample: list of T side-by-side BGR panels

    for idx, s in samples:
        subject, action, cam_ind = s['subject'], s['action'], s['cam_ind']
        frame_idx = s['frame_idx']

        kps = np.asarray(s['input_2D_GT']).squeeze()
        if kps.ndim == 2:
            kps = kps[None]
        T = kps.shape[0]

        res_w, res_h = fusion.h_w[(subject, action, cam_ind)]
        kps_px = denormalize_2d(kps, res_w, res_h) if looks_normalized(kps) else kps

        poses_3d = as_pose_sequence(s['gt_3D'], T)

        video_frames = (frame_idx * fusion.downsample).astype(int)

        print(f"[{idx:>6}] {subject} / {action} / cam {cam_ind} | "
              f"{T} frames | video {video_frames[0]}..{video_frames[-1]} "
              f"(step {fusion.downsample})")

        reader = VideoReader(args.frames_root, subject, action, cam_ind)
        raw = reader.read_frames(video_frames)
        reader.release()

        rendered = []
        for t in range(T):
            left = cv2.resize(draw_skeleton_2d(raw[t].copy(), kps_px[t]),
                              (args.tile, args.tile))
            left = label(left, f"{subject} {action} c{cam_ind} f{video_frames[t]}")

            right = renderer.render(poses_3d[t])
            right = label(right, "gt_3D (camera space)")

            rendered.append(np.hstack([left, right]))

        cells.append(rendered)

    renderer.close()

    # -----------------------------------------------------------------
    # Tile the sample pairs into a grid and write
    # -----------------------------------------------------------------
    n = len(cells)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    T = min(len(x) for x in cells)

    cell_h, cell_w = cells[0][0].shape[:2]
    out_w, out_h = cols * cell_w, rows * cell_h

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {args.out}")

    blank = np.zeros((cell_h, cell_w, 3), np.uint8)

    for t in range(T):
        grid_rows = []
        for r in range(rows):
            row = [cells[r * cols + c][t] if r * cols + c < n else blank
                   for c in range(cols)]
            grid_rows.append(np.hstack(row))
        writer.write(np.vstack(grid_rows))

    writer.release()
    print(f"\nWrote: {args.out}  ({out_w}x{out_h}, {T} frames)")


if __name__ == '__main__':
    main()