#!/usr/bin/env python
"""
Build the `features_path` file that `losd_data_h36.Fusion` expect
Pipeline (per subject / action / camera):
    1. Load that camera's pixel-space 2D joints (17, root = joint 0).
    2. Compute ONE fixed crop box for the whole clip from the root-centered
       joints (compute_crop_size_from_centered).
    3. Stream the video frame-by-frame; crop each frame around
       (per-frame root + fixed offset) with the fixed side length, resize.
    4. Run crops through a FeatureExtractor -> (N, feat_dim).

Extractors (--extractor):
    resnet_v2       TF resnet_v2_50 / HMR      224x224  -> 2048-d
    hmr2_vit_mean   HMR2.0 ViT-H patch mean    256x256  -> 1280-d
    hmr2_crossattn  HMR2.0 decoder token       256x256  -> 1024-d
"""

import os
import sys
import time
import shutil
import argparse

import cv2
import numpy as np

from extractors import (EXTRACTORS, build_extractor,
                        compute_crop_size_from_centered, crop_one_frame)


cameras = ["54138969", "55011271", "58860488", "60457274"]

ACTION_VIDEO_ALIAS = {
    "Photo": "TakingPhoto",
    "WalkDog": "WalkingDog",
    "WalkTogether": "WalkingTogether",
}


# ─────────────────────────────────────────────────────────────────────────────
# Video resolution
# ─────────────────────────────────────────────────────────────────────────────
def action_to_video_basename(action, use_alias=True):
    if not use_alias:
        return action
    tokens = action.split(" ")
    if tokens[0] in ACTION_VIDEO_ALIAS:
        tokens[0] = ACTION_VIDEO_ALIAS[tokens[0]]
    return " ".join(tokens)


def resolve_video_path(videos_root, subject, action, cam_id, use_alias=True):
    """Try alias name then raw name. Return path or None."""
    names = {action_to_video_basename(action, use_alias), action}
    for name in names:
        p = os.path.join(videos_root, subject, "Videos", f"{name}.{cam_id}.mp4")
        if os.path.isfile(p):
            return p
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Scratch cache I/O (deleted after the single .npy is written)
# ─────────────────────────────────────────────────────────────────────────────
def shard_path(shard_dir, subject, action):
    return os.path.join(shard_dir, subject, f"{action.replace('/', '_')}.npy")


def save_shard(shard_dir, subject, action, per_cam):
    p = shard_path(shard_dir, subject, action)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    np.save(p, np.stack(per_cam, axis=0))
    return p


def load_shard(shard_dir, subject, action, n_cams, n_frames):
    p = shard_path(shard_dir, subject, action)
    if not os.path.isfile(p):
        return None
    try:
        arr = np.load(p)
    except Exception:
        return None
    if arr.ndim != 3 or arr.shape[0] != n_cams or arr.shape[1] != n_frames:
        return None
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# Sanity checks
# ─────────────────────────────────────────────────────────────────────────────
def dump_sample_crops(dbg_dir, subject, action, cam_idx, vpath, j2d, scale,
                      out_size, slice_cols=False, every=200, limit=5):
    """Write a few crops so the framing can be eyeballed. With slice_cols the
    centre 192 columns are marked, i.e. what the ViT actually sees."""
    os.makedirs(dbg_dir, exist_ok=True)
    cx, cy, side = compute_crop_size_from_centered(j2d, scale)
    cap = cv2.VideoCapture(vpath)
    written, i = 0, 0
    while written < limit and i < j2d.shape[0]:
        ret, frame = cap.read()
        if not ret:
            break
        if i % every == 0:
            crop = crop_one_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                  j2d[i, 0, :], cx, cy, side, out_size)
            vis = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            if slice_cols and out_size == 256:
                cv2.rectangle(vis, (32, 0), (223, 255), (0, 255, 255), 1)
            name = f"{subject}_{action.replace(' ', '-')}_cam{cam_idx}_{i:06d}.png"
            cv2.imwrite(os.path.join(dbg_dir, name), vis)
            written += 1
        i += 1
    cap.release()
    print(f"[debug] wrote {written} crops to {dbg_dir}  "
          f"(side={side:.1f}px, offset=({cx:.1f}, {cy:.1f}), out={out_size})")
    if slice_cols:
        print("[debug] the yellow box is the centre 192 columns the ViT sees; "
              "if limbs leave it, raise --scale")


def feature_sanity(feat):
    """Adjacent-frame vs. far-frame cosine similarity. Healthy: adjacent ~0.95+,
    far clearly lower. Both ~1.0 means the weights did not restore."""
    f = feat.astype(np.float32)
    if f.shape[0] < 50:
        return None
    n = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
    adj = float((n[:-1] * n[1:]).sum(1).mean())
    k = f.shape[0] // 2
    far = float((n[:k] * n[k:2 * k]).sum(1).mean())
    return adj, far


def print_structure(feats, per_subject=5):
    """Print the tree in the same shape as the spec, so the file can be eyeballed
    without a second script."""
    print(f"root: DICT ({len(feats)} keys)")
    for subject in feats:
        acts = feats[subject]
        print(f"  root/{subject}: DICT ({len(acts)} keys)")
        for j, action in enumerate(acts):
            if j >= per_subject:
                print(f"    ... {len(acts) - per_subject} more actions")
                break
            cams = acts[action]
            print(f"    root/{subject}/{action}: LIST (length={len(cams)})")
            a = np.asarray(cams[0])
            print(f"      root/{subject}/{action}[0]: ARRAY "
                  f"shape={a.shape}, dtype={a.dtype}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_2d_path", required=True,
                    help="Path to data_2d_h36m_gt.npz (pixel-space 2D joints).")
    ap.add_argument("--vid_path", required=True,
                    help="Root with {subject}/Videos/{action}.{camid}.mp4")
    ap.add_argument("--ckpt", required=True,
                    help="resnet_v2_50/HMR ckpt, or the HMR2.0 .ckpt "
                         "(epoch=35-step=1000000.ckpt) for the hmr2_* extractors")
    ap.add_argument("--extractor", default="resnet_v2", choices=sorted(EXTRACTORS))
    ap.add_argument("--out", default=None,
                    help="THE output file: one .npy holding "
                         "dict[subject][action] = [4 x (N, D)] float32.")
    ap.add_argument("--out_dir", default=None,
                    help="Where --out and the scratch cache live "
                         "(default: dirname of --out).")
    ap.add_argument("--subjects", nargs="+",
                    default=['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11'])
    ap.add_argument("--scale", type=float, default=1.5,
                    help="Crop side = scale * max bbox extent.")
    ap.add_argument("--out-size", type=int, default=None,
                    help="Override the extractor's native crop size. Leave "
                         "unset; the ViT extractors require 256.")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="Default: 128 for resnet_v2, 32 for the ViT extractors.")
    ap.add_argument("--device", default=None, help="cuda / cuda:1 / cpu (torch only)")
    ap.add_argument("--fp32", action="store_true",
                    help="Run the ViT in fp32 instead of fp16.")
    ap.add_argument("--hmr2_cfg", default=None,
                    help="model_config.yaml next to the HMR2 checkpoint. Supplies "
                         "IEF_ITERS, which the weights cannot reveal.")
    ap.add_argument("--ief-iters", type=int, default=1,
                    help="Decoder iterations for hmr2_crossattn. Ignored when "
                         "--hmr2_cfg is given, since the config is authoritative.")
    ap.add_argument("--store-dtype", default="float32",
                    choices=["float16", "float32"],
                    help="dtype of the arrays inside the .npy. float32 matches "
                         "the spec; float16 halves a ~16 GB file if RAM is tight.")
    ap.add_argument("--no-action-alias", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="Reuse scratch-cache entries with the right shape.")
    ap.add_argument("--keep-shards", action="store_true",
                    help="Do not delete the scratch cache after writing --out.")
    ap.add_argument("--no-assemble", action="store_true",
                    help="Stop after the scratch cache; write no .npy.")
    ap.add_argument("--allow-partial", action="store_true",
                    help="Write the .npy even if some clips are missing. Off by "
                         "default, so a half-finished run cannot masquerade as "
                         "a complete dataset.")
    ap.add_argument("--debug-crops", default=None,
                    help="Dump sample crops there and exit (no model loaded).")
    args = ap.parse_args()

    if args.out_dir is None and args.out is None:
        ap.error("one of --out or --out_dir is required")
    if args.out_dir is None:
        args.out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
    if args.out is None:
        args.out = os.path.join(args.out_dir, f"{args.extractor}.npy")
    return args


def main():
    args = parse_args()
    use_alias = not args.no_action_alias
    store_dtype = np.float16 if args.store_dtype == "float16" else np.float32

    # Scratch cache, named after the extractor so 2048-d and 1280-d runs cannot
    # contaminate each other. Removed once --out is on disk.
    shard_dir = os.path.join(args.out_dir, f".cache_{args.extractor}")
    os.makedirs(shard_dir, exist_ok=True)

    print(f"Loading 2D keypoints: {args.dataset_2d_path}")
    kp = np.load(args.dataset_2d_path, allow_pickle=True)["positions_2d"].item()

    todo = []
    for subject in args.subjects:
        if subject not in kp:
            print(f"[warn] subject {subject} not in keypoints; skipping")
            continue
        for action in kp[subject]:
            todo.append((subject, action))

    if args.debug_crops:
        native = EXTRACTORS[args.extractor].out_size
        subject, action = todo[0]
        j2d = np.asarray(kp[subject][action][0], np.float32)
        vpath = resolve_video_path(args.vid_path, subject, action,
                                   cameras[0], use_alias)
        if vpath is None:
            sys.exit(f"[ERROR] no video for {subject}/{action} cam0")
        dump_sample_crops(args.debug_crops, subject, action, 0, vpath, j2d,
                          args.scale, args.out_size or native,
                          slice_cols=args.extractor.startswith("hmr2"))
        return

    extractor = build_extractor(args.extractor, args)
    batch_size = args.batch_size or extractor.default_batch_size
    print(f"Extractor {args.extractor}: crop={extractor.out_size} "
          f"dim={extractor.feat_dim} batch={batch_size} store={args.store_dtype}")
    print(f"{len(todo)} clips -> {args.out}\n")

    t0, done, failed, reported = time.time(), 0, [], False

    for subject, action in todo:
        cam_kp = kp[subject][action]
        n_cams = len(cam_kp)
        n_frames = int(np.asarray(cam_kp[0]).shape[0])

        if args.resume and load_shard(shard_dir, subject, action,
                                      n_cams, n_frames) is not None:
            print(f"[skip] {subject}/{action} (already extracted)")
            done += 1
            continue

        per_cam, ok = [None] * n_cams, True
        for cam_idx in range(n_cams):
            j2d = np.asarray(cam_kp[cam_idx], dtype=np.float32)   # (N, J, 2)
            N = j2d.shape[0]
            if N != n_frames:
                print(f"[ERROR] {subject}/{action} cam{cam_idx}: length {N} "
                      f"!= cam0 length {n_frames}")
                ok = False
                break
            if not np.isfinite(j2d).all():
                print(f"[ERROR] {subject}/{action} cam{cam_idx}: non-finite keypoints")
                ok = False
                break

            cam_id = cameras[cam_idx] if cam_idx < len(cameras) else None
            vpath = resolve_video_path(args.vid_path, subject, action,
                                       cam_id, use_alias) if cam_id else None
            if vpath is None:
                print(f"[ERROR] video not found: {subject}/{action} "
                      f"cam{cam_idx} ({cam_id}). Check --vid_path / aliases.")
                ok = False
                break

            try:
                fc, m = extractor.extract_clip(vpath, j2d, args.scale, batch_size)
            except Exception as e:
                print(f"[ERROR] {subject}/{action} cam{cam_idx}: {e}")
                ok = False
                break

            if m == 0:
                print(f"[ERROR] {subject}/{action} cam{cam_idx}: no frames decoded")
                ok = False
                break

            # Keep len(features) == len(keypoints).
            if m < N:
                print(f"[warn] {subject}/{action} cam{cam_idx}: {m} frames vs "
                      f"{N} keypoints; padding {N - m} with the last frame.")
                fc = np.concatenate([fc, np.repeat(fc[-1:], N - m, axis=0)], axis=0)
            elif m > N:
                fc = fc[:N]

            if not reported:
                s = feature_sanity(fc)
                if s:
                    print(f"[sanity] cosine adjacent={s[0]:.4f} far={s[1]:.4f} "
                          f"(far should be clearly lower)")
                    reported = True

            per_cam[cam_idx] = fc.astype(store_dtype)

        if ok:
            save_shard(shard_dir, subject, action, per_cam)
            done += 1
            print(f"[{done}/{len(todo)}] {subject}/{action}  cams={n_cams} "
                  f"frames={per_cam[0].shape[0]} dim={per_cam[0].shape[1]}  "
                  f"({time.time() - t0:.0f}s)")
        else:
            failed.append((subject, action))
            print(f"[skip-save] {subject}/{action} incomplete; not stored")

    extractor.close()

    if failed:
        print(f"\n{len(failed)} clip(s) failed:")
        for s, a in failed:
            print(f"  {s}/{a}")

    if args.no_assemble:
        print(f"\nStopped before assembly. Scratch cache: {shard_dir}")
        return

    # ── the single output file ───────────────────────────────────────────────
    print("\nAssembling the single .npy (whole dataset held in RAM)...")
    feats, missing = {}, []
    for subject, action in todo:
        arr = load_shard(shard_dir, subject, action, len(kp[subject][action]),
                         int(np.asarray(kp[subject][action][0]).shape[0]))
        if arr is None:
            missing.append((subject, action))
            continue
        # arr[c] is a view into arr, so this does not double the memory.
        feats.setdefault(subject, {})[action] = [arr[c] for c in range(arr.shape[0])]

    if missing and not args.allow_partial:
        print(f"\n[ERROR] {len(missing)} clip(s) have no usable features:")
        for s, a in missing[:20]:
            print(f"  {s}/{a}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")
        print(f"\nNothing written. The scratch cache is intact at {shard_dir}; "
              f"fix the cause and re-run with --resume, or pass --allow-partial "
              f"to write an incomplete file on purpose.")
        sys.exit(1)
    if missing:
        print(f"[warn] writing a PARTIAL file: {len(missing)} clip(s) missing")

    np.save(args.out, feats)
    size_gb = os.path.getsize(args.out) / 1024 ** 3
    print(f"Saved {args.out}  ({size_gb:.2f} GB)\n")
    print_structure(feats)
    print("\nLoad with: np.load(path, allow_pickle=True).item()[subject][action][cam]")

    if args.keep_shards:
        print(f"Scratch cache kept at {shard_dir}")
    else:
        shutil.rmtree(shard_dir, ignore_errors=True)
        print(f"Removed scratch cache {shard_dir}; {args.out} is the only artifact.")


if __name__ == "__main__":
    main()