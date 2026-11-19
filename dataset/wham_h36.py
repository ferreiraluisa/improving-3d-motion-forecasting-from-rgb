"""Convert a WHAM feature file from COCO-17 to H36M-17 with a trained adapter.

Reads the file written by the precompute script:

    feats[subject][action][cam]  ->  (T, 563) = [context (512) | coco17 kp3d (51)]

runs every camera of every action through Coco2H36mAdapter, and writes ONE file
with exactly the same nesting, keys, per-camera lengths and container types:

    feats[subject][action][cam]  ->  (T, 563) = [context (512) | h36m17 kp3d (51)]

The 512-d context is copied straight from the source array, so it is bit-exact.

    python make_h36m_feats.py \
        --in  /media/data/ferreiral1/WHAM/feats.npy \
        --out /media/data/ferreiral1/WHAM/feats_h36m.npy \
        --ckpt checkpoints/h36_adapter_best.pth
"""

import argparse
import os
import os.path as osp
import time

import numpy as np
import torch

from models.h36_adapter import Coco2H36mAdapter


# --------------------------------------------------------------------------- #
# io
# --------------------------------------------------------------------------- #
def load_feats(path):
    """Matches both branches of the precompute script's merge()."""
    if path.endswith('.npz'):
        return np.load(path, allow_pickle=True)['positions_wham'].item()
    return np.load(path, allow_pickle=True).item()


def save_feats(path, feats):
    os.makedirs(osp.dirname(osp.abspath(path)) or '.', exist_ok=True)
    tmp = f'{path}.tmp{os.getpid()}'
    if path.endswith('.npz'):
        np.savez(tmp + '.npz', positions_wham=np.array(feats, dtype=object))
        os.replace(tmp + '.npz', path)
        how = "np.load(path, allow_pickle=True)['positions_wham'].item()"
    else:
        np.save(tmp + '.npy', feats, allow_pickle=True)
        os.replace(tmp + '.npy', path)
        how = 'np.load(path, allow_pickle=True).item()'
    return how


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    saved = ckpt.get('opt', {})
    d_ctx = saved.get('d_ctx', 512)
    model = Coco2H36mAdapter(
        d_ctx=d_ctx,
        hidden=saved.get('hidden', 512),
        n_blocks=saved.get('blocks', 2),
        dropout=saved.get('dropout', 0.1),
        use_ctx=saved.get('use_ctx', False),
        temporal=not saved.get('no_temporal', False),
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f'[model] {ckpt_path}  epoch {ckpt.get("epoch", "?")}  '
          f'val MPJPE {ckpt.get("mpjpe", float("nan")):.2f} mm  '
          f'use_ctx={saved.get("use_ctx", False)}  d_ctx={d_ctx}')
    if saved.get('downsample', 2) != 1:
        print(f'[model] NOTE: trained with downsample={saved.get("downsample")}. '
              'These features are converted at their stored frame rate, which is '
              'correct — the adapter is per-frame plus a 5-tap temporal conv.')
    return model, d_ctx


@torch.no_grad()
def convert_sequence(model, arr, d_ctx, device, chunk, halo, out_dtype):
    """(T, 563) COCO -> (T, 563) H36M. Context copied verbatim from `arr`."""
    x = torch.from_numpy(np.ascontiguousarray(arr)).float()
    T = x.shape[0]
    kp_out = torch.empty(T, 17 * 3)

    # Chunk with a halo so the temporal conv sees real neighbours at the seams.
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        lo, hi = max(0, s - halo), min(T, e + halo)
        y = model.forward_joints(x[lo:hi].unsqueeze(0).to(device))[0]
        kp_out[s:e] = y[s - lo:e - lo].reshape(e - s, -1).cpu()

    out = np.empty((T, d_ctx + 51), dtype=out_dtype)
    out[:, :d_ctx] = arr[:, :d_ctx]                 # bit-exact passthrough
    out[:, d_ctx:] = kp_out.numpy().astype(out_dtype)
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='src', required=True,
                    help='old feature file, COCO-17 in the last 51 dims')
    ap.add_argument('--out', dest='dst', required=True,
                    help='new feature file, H36M-17 in the last 51 dims')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--dtype', choices=['float32', 'float16', 'same'],
                    default='same', help='output dtype (default: match input)')
    ap.add_argument('--chunk', type=int, default=8192,
                    help='frames per forward pass')
    ap.add_argument('--halo', type=int, default=16,
                    help='context frames on each side of a chunk, discarded')
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--free', action='store_true',
                    help='drop each source array after converting it — halves '
                         'peak RAM, but the input dict is destroyed in memory')
    ap.add_argument('--subjects', default='',
                    help='comma-separated subset, e.g. S9,S11 (default: all)')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    if osp.abspath(args.src) == osp.abspath(args.dst):
        raise SystemExit('--in and --out must differ')

    device = 'cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
    model, d_ctx = load_model(args.ckpt, device)

    print(f'[load] {args.src}')
    src = load_feats(args.src)
    subjects = ([s.strip() for s in args.subjects.split(',') if s.strip()]
                or sorted(src.keys()))
    missing = [s for s in subjects if s not in src]
    if missing:
        raise SystemExit(f'not in the feature file: {missing}')

    n_cams = sum(len(src[s][a]) for s in subjects for a in src[s])
    print(f'[load] {len(subjects)} subjects, '
          f'{sum(len(src[s]) for s in subjects)} actions, {n_cams} cameras')
    if args.dry_run:
        s0 = subjects[0]
        a0 = sorted(src[s0])[0]
        print(f'[dry-run] example {s0}/{a0}: '
              f'{[np.asarray(c).shape for c in src[s0][a0]]}')
        return

    dst, done, frames, t0 = {}, 0, 0, time.time()
    max_ctx_delta = 0.0

    for subject in subjects:
        dst[subject] = {}
        for action in sorted(src[subject]):
            cams = src[subject][action]
            was_array = isinstance(cams, np.ndarray) and cams.ndim == 3
            out_cams = []

            for ci in range(len(cams)):
                arr = np.asarray(cams[ci])
                if arr.ndim != 2 or arr.shape[-1] != d_ctx + 51:
                    raise ValueError(
                        f'{subject}/{action}/cam{ci}: expected (T, {d_ctx + 51}), '
                        f'got {arr.shape}')
                out_dtype = arr.dtype if args.dtype == 'same' else np.dtype(args.dtype)

                new = convert_sequence(model, arr, d_ctx, device,
                                       args.chunk, args.halo, out_dtype)
                max_ctx_delta = max(
                    max_ctx_delta,
                    float(np.abs(new[:, :d_ctx].astype(np.float64) -
                                 arr[:, :d_ctx].astype(np.float64)).max()))
                out_cams.append(new)
                frames += len(new)
                done += 1

                if args.free:
                    if was_array:
                        pass                     # can't free a slice of an array
                    else:
                        cams[ci] = None

            dst[subject][action] = (np.stack(out_cams) if was_array else out_cams)
            if args.free:
                src[subject][action] = None

            if done % 40 == 0 or done == n_cams:
                print(f'\r[convert] {done}/{n_cams} cameras  {frames} frames  '
                      f'[{time.time() - t0:.0f}s]', end='', flush=True)

    gb = frames * (d_ctx + 51) * np.dtype(
        out_dtype if args.dtype != 'same' else np.float32).itemsize / 1e9
    print(f'\n[check] context passthrough max|delta| = {max_ctx_delta:.2e} '
          '(must be 0)')
    print(f'[write] {frames} camera-frames, ~{gb:.1f} GB')

    how = save_feats(args.dst, dst)
    print(f'wrote {args.dst}')
    print(f'load it with: feats = {how}')
    print(f'              feats[subject][action][cam]  ->  (T, {d_ctx + 51}), '
          'last 51 = H36M-17')


if __name__ == '__main__':
    main()