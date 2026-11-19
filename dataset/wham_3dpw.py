"""Convert the 3DPW WHAM feature file from COCO-17 to SMPL-24 with a trained adapter.

Input layout (as printed by read.py):

    feats[split][sequence]  ->  [person0, person1, ...]
        split     in {'train', 'validation', 'test'}
        personN   (T, 563) = [context (512) | coco17 kp3d (17*3 = 51)]

Output: ONE file with exactly the same nesting, keys, key order, per-person
lengths and container types (list stays list, stacked ndarray stays ndarray):

    feats[split][sequence]  ->  [person0, ...]   each (T, 584) = [context (512) | smpl24 kp3d (24*3 = 72)]

The 512-d context is copied straight from the source array, so it is bit-exact.

    python make_3dpw_smpl_feats.py \
        --in   /media/data/ferreiral1/3DPW/feats/3dpw_wham.npy \
        --out  /media/data/ferreiral1/3DPW/feats/3dpw_wham_smpl.npy \
        --ckpt adapter.pt

If --out is omitted, it writes 3dpw_wham_smpl.npy next to the input file.
"""

import argparse
import os
import os.path as osp
import re
import time

import numpy as np
import torch

from models.smpl_adapter import Coco2SmplAdapter


# --------------------------------------------------------------------------- #
# io
# --------------------------------------------------------------------------- #
def load_feats(path):
    obj = np.load(path, allow_pickle=True)
    if path.endswith('.npz'):
        key = 'positions_wham' if 'positions_wham' in obj.files else obj.files[0]
        return obj[key].item()
    return obj.item()


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
def _extract_state_dict(ckpt):
    """Accept {'model': sd, ...}, {'state_dict': sd}, a bare state_dict, or a
    whole pickled nn.Module."""
    if isinstance(ckpt, torch.nn.Module):
        return ckpt.state_dict()
    if not isinstance(ckpt, dict):
        raise SystemExit(f'[model] unrecognised checkpoint type: {type(ckpt)}')
    for k in ('model', 'state_dict', 'model_state_dict', 'model_state', 'net', 'ema'):
        if k in ckpt and isinstance(ckpt[k], dict):
            print(f"[model] weights found under ckpt['{k}']")
            return ckpt[k]
        if k in ckpt and isinstance(ckpt[k], torch.nn.Module):
            print(f"[model] module found under ckpt['{k}']")
            return ckpt[k].state_dict()
    if ckpt and all(torch.is_tensor(v) for v in ckpt.values()):
        print('[model] checkpoint is a bare state_dict')
        return ckpt
    raise SystemExit(f'[model] no weights found; top-level keys: {list(ckpt.keys())}')


def _arch_from_state_dict(sd):
    """Read the architecture straight off the weights, so a bare state_dict
    with no saved options still builds the right model."""
    n_out, n_in = sd['A'].shape                       # (24, 17)
    hidden, kp_dim = sd['kp_in.weight'].shape         # (hidden, n_in*3)
    assert kp_dim == n_in * 3, (kp_dim, n_in)
    blocks = {int(m.group(1)) for k in sd
              if (m := re.match(r'blocks\.(\d+)\.', k))}
    arch = dict(n_in=n_in, n_out=n_out, hidden=hidden,
                n_blocks=(max(blocks) + 1) if blocks else 0,
                use_ctx='ctx_in.weight' in sd,
                temporal='temporal.weight' in sd)
    if arch['use_ctx']:
        arch['d_ctx'] = sd['ctx_in.weight'].shape[1]
    return arch


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = _extract_state_dict(ckpt)
    sd = {k[len('module.'):] if k.startswith('module.') else k: v
          for k, v in sd.items()}                     # strip DataParallel prefix

    saved = ckpt.get('opt', ckpt.get('args', {})) if isinstance(ckpt, dict) else {}
    if not isinstance(saved, dict):                   # argparse.Namespace
        saved = vars(saved)

    arch = _arch_from_state_dict(sd)
    d_ctx = arch.pop('d_ctx', saved.get('d_ctx', 512))
    model = Coco2SmplAdapter(d_ctx=d_ctx, dropout=0.0, **arch).to(device)
    model.load_state_dict(sd)
    model.eval()

    meta = ckpt if isinstance(ckpt, dict) else {}
    mpjpe = meta.get('mpjpe', float('nan'))
    print(f'[model] {ckpt_path}  epoch {meta.get("epoch", "?")}  '
          f'val MPJPE {mpjpe:.2f} mm  d_ctx={d_ctx}  {arch}')
    return model, d_ctx, arch['n_in'], arch['n_out']


@torch.no_grad()
def convert_sequence(model, arr, d_ctx, n_out, device, chunk, halo, out_dtype):
    """(T, d_ctx + 51) COCO -> (T, d_ctx + 72) SMPL. Context copied verbatim."""
    x = torch.from_numpy(np.ascontiguousarray(arr)).float()
    T = x.shape[0]
    kp_out = torch.empty(T, n_out * 3)

    # Chunk with a halo so the temporal conv sees real neighbours at the seams.
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        lo, hi = max(0, s - halo), min(T, e + halo)
        y = model.forward_joints(x[lo:hi].unsqueeze(0).to(device))[0]   # (t, 24, 3)
        kp_out[s:e] = y[s - lo:e - lo].reshape(e - s, -1).cpu()

    out = np.empty((T, d_ctx + n_out * 3), dtype=out_dtype)
    out[:, :d_ctx] = arr[:, :d_ctx]                   # bit-exact passthrough
    out[:, d_ctx:] = kp_out.numpy().astype(out_dtype)
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='src', required=True,
                    help='3DPW feature file, (T, 563) with COCO-17 in the last 51 dims')
    ap.add_argument('--out', dest='dst', default='',
                    help='output file (default: 3dpw_wham_smpl.npy next to --in)')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--dtype', choices=['float32', 'float16', 'same'],
                    default='same', help='output dtype (default: match input)')
    ap.add_argument('--chunk', type=int, default=8192,
                    help='frames per forward pass')
    ap.add_argument('--halo', type=int, default=16,
                    help='context frames on each side of a chunk, discarded')
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--splits', default='',
                    help='comma-separated subset, e.g. test (default: all)')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    if not args.dst:
        args.dst = osp.join(osp.dirname(osp.abspath(args.src)), '3dpw_wham_smpl.npy')
    if osp.abspath(args.src) == osp.abspath(args.dst):
        raise SystemExit('--in and --out must differ')

    device = 'cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
    model, d_ctx, n_in, n_out = load_model(args.ckpt, device)
    D_in, D_out = d_ctx + n_in * 3, d_ctx + n_out * 3      # 563, 584

    print(f'[load] {args.src}')
    src = load_feats(args.src)
    splits = ([s.strip() for s in args.splits.split(',') if s.strip()]
              or list(src.keys()))                    # keep original key order
    missing = [s for s in splits if s not in src]
    if missing:
        raise SystemExit(f'not in the feature file: {missing}  '
                         f'(available: {list(src.keys())})')

    n_people = sum(len(src[sp][q]) for sp in splits for q in src[sp])
    for sp in splits:
        print(f'[load] {sp}: {len(src[sp])} sequences, '
              f'{sum(len(v) for v in src[sp].values())} person-tracks')
    if args.dry_run:
        sp0 = splits[0]
        q0 = next(iter(src[sp0]))
        v0 = src[sp0][q0]
        print(f'[dry-run] example {sp0}/{q0}: type={type(v0).__name__}  '
              f'{[np.asarray(p).shape for p in v0]}  -> each becomes (T, {D_out})')
        return

    dst, done, frames, t0 = {}, 0, 0, time.time()
    max_ctx_delta = 0.0
    last_dtype = np.float32

    for split in splits:
        dst[split] = {}
        for seq in src[split]:                        # preserve original order
            people = src[split][seq]
            single_2d = isinstance(people, np.ndarray) and people.ndim == 2
            was_array = isinstance(people, np.ndarray) and people.ndim == 3
            tracks = [people] if single_2d else people
            out_people = []

            for pi in range(len(tracks)):
                arr = np.asarray(tracks[pi])
                if arr.ndim != 2 or arr.shape[-1] != D_in:
                    raise ValueError(f'{split}/{seq}/person{pi}: expected (T, {D_in}), '
                                     f'got {arr.shape}')
                out_dtype = arr.dtype if args.dtype == 'same' else np.dtype(args.dtype)
                last_dtype = out_dtype

                new = convert_sequence(model, arr, d_ctx, n_out, device,
                                       args.chunk, args.halo, out_dtype)
                assert new.shape == (len(arr), D_out), new.shape
                max_ctx_delta = max(
                    max_ctx_delta,
                    float(np.abs(new[:, :d_ctx].astype(np.float64) -
                                 arr[:, :d_ctx].astype(np.float64)).max()))
                out_people.append(new)
                frames += len(new)
                done += 1

            if single_2d:
                dst[split][seq] = out_people[0]
            elif was_array:
                dst[split][seq] = np.stack(out_people)
            else:
                dst[split][seq] = out_people

            print(f'\r[convert] {done}/{n_people} person-tracks  {frames} frames  '
                  f'[{time.time() - t0:.0f}s]', end='', flush=True)

    gb = frames * D_out * np.dtype(last_dtype).itemsize / 1e9
    print(f'\n[check] context passthrough max|delta| = {max_ctx_delta:.2e} '
          '(must be 0)')
    print(f'[write] {frames} person-frames x {D_out} dims, ~{gb:.2f} GB')

    how = save_feats(args.dst, dst)
    print(f'wrote {args.dst}')
    print(f'load it with: feats = {how}')
    print(f'              feats[split][sequence][person]  ->  (T, {D_out}), '
          f'last {n_out * 3} = SMPL-{n_out} (x, y, z per joint)')


if __name__ == '__main__':
    main()