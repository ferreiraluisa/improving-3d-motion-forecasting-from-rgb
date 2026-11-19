"""
3DPW feature cache, same extractors as the H36M pipeline (extractors.py).

Layout expected (official 3DPW release):
    <root>/sequenceFiles/{train,validation,test}/<seq>.pkl
    <root>/imageFiles/<seq>/image_%05d.jpg

Pipeline (per split / sequence / actor):
    1. Load the .pkl (latin1 pickle). Each sequence has 1-2 annotated actors.
    2. Project that actor's SMPL joints (world) into the image with the
       per-frame cam_poses (the camera moves in 3DPW) and cam_intrinsics.
    3. A frame is USABLE for an actor when campose_valid is set AND all 24
       projected joints are inside the image and in front of the camera.
       --keep-out-of-image drops the second condition (campose_valid is
       always required).
    4. Every run of consecutive usable frames at least --min-len frames long
       becomes its own segment, named

           <seq>_<frame_start>_<frame_end>     e.g. downtown_walkUphill_00_120_455

       with image indices (image_%05d.jpg), frame_end INCLUSIVE. Segments are
       continuous by construction; frames outside them are never extracted.
    5. Per-frame square box from the 2D joints (center + side). Unlike H36M a
       single clip-level box does not work here: the subject's depth changes a
       lot, so the box side is per frame.
    6. Read each needed image ONCE, crop every actor usable in it, run the
       crops through the extractor.

Outputs:
    --out          dict[split][segment] = [person (n, D), ...]
                   n = frame_end - frame_start + 1. Usually one person; two
                   only when both actors of a sequence are usable over exactly
                   the same frames. List position k is actor_ids[k].
    <out>_meta.npy dict[split][segment] = {
        'source_seq':    str             original 3DPW sequence name
        'frame_start':   int             first image index (inclusive)
        'frame_end':     int             last image index (inclusive)
        'actor_ids':     [A]             original 3DPW actor index of each entry
        'img_frame_ids': (n,)            image file index of every row
        'valid':         [A x (n,) bool] campose_valid (all True inside segments)
        'box':           [A x (n, 3)]    crop (cx, cy, side) in pixels
        'joints2d':      [A x (n, 24, 2)] projected SMPL joints, pixels
        'joints_cam':    [A x (n, 24, 3)] SMPL joints in CAMERA coordinates (m)
        'K':             (3, 3)          full-image intrinsics
        'image_hw':      (H, W)
    }
    Use crop_intrinsics(K, box, out_size) for the intrinsics of a crop.

--extractor wham (see extractors.WHAMExtractor):
    Steps 1-4 are the same (same usable frames, same segments, same meta),
    but WHAM does NOT use the crops of steps 5-6. Exactly as in the H36M WHAM
    script it runs its own 2D detector on the full frames, builds WHAM's boxes
    from those keypoints, crops with WHAM's crop and runs the motion encoder +
    integrator. It runs once per segment (actor, continuous run), so every
    segment is its own WHAM sequence with its own SMPL init.
    The step-5 SMPL joints only pick WHICH person to detect (a tight box from
    the projected joints, matched by IoU), or with --wham-box gt become the
    ViTPose person box. meta['box'] is still the step-5 box, not WHAM's.
    Outputs:
        --out              phi_hat (integrator output), (n, D_CONTEXT)
        <out>_motion.npy   motion encoder output BEFORE the integrator
        <out>_img.npy      the HMR2 token WHAM consumed (--wham-cache-img-feats)
    all with the same [split][segment][person] layout as --out.
"""

import os
import sys
import glob
import time
import pickle
import shutil
import argparse
import itertools
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from extractors import EXTRACTORS, build_extractor, crop_one_frame


SPLIT_ALIASES = {"val": ["val", "validation"],
                 "validation": ["validation", "val"]}


# ────────────────────────────────────────────────────────────────────────────
# 3DPW I/O
# ─────────────────────────────────────────────────────────────────────────────
def find_split_dir(root, split):
    for name in SPLIT_ALIASES.get(split, [split]):
        for cand in (os.path.join(root, "sequenceFiles", name),
                     os.path.join(root, name)):
            if os.path.isdir(cand) and glob.glob(os.path.join(cand, "*.pkl")):
                return cand
    return None


def load_seq(pkl_path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def frame_image_paths(img_dir, seq, d, N):
    """Image file for every annotation row.

    The released .pkl files are already at the 30 Hz image rate (one row per
    image), so row i -> image_{i:05d}.jpg. img_frame_ids (meant to downsample
    60 Hz poses) is used only if the folder does not hold one image per row.
    """
    n_img = len(glob.glob(os.path.join(img_dir, seq, "image_*.jpg")))
    if n_img == N:
        ids = np.arange(N, dtype=np.int64)
    else:
        ids = np.asarray(d.get("img_frame_ids", np.arange(N)), np.int64)
        print(f"[warn] {seq}: {n_img} images vs {N} annotation rows; "
              f"falling back to img_frame_ids")
    paths = [os.path.join(img_dir, seq, f"image_{int(i):05d}.jpg") for i in ids]
    return ids, paths


def read_bgr(path):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise IOError(f"Cannot read image: {path}")
    return im


def read_rgb(path):
    return cv2.cvtColor(read_bgr(path), cv2.COLOR_BGR2RGB)


def prefetch_images(paths, workers, reader=read_rgb):
    """Ordered, bounded prefetch: at most 2*workers decoded frames in RAM."""
    with ThreadPoolExecutor(max_workers=workers) as ex:
        it = iter(paths)
        q = deque(ex.submit(reader, p) for p in itertools.islice(it, 2 * workers))
        while q:
            img = q.popleft().result()
            nxt = next(it, None)
            if nxt is not None:
                q.append(ex.submit(reader, nxt))
            yield img


# ─────────────────────────────────────────────────────────────────────────────
# Geometry
# ─────────────────────────────────────────────────────────────────────────────
def project_joints(d, actor):
    """World SMPL joints -> uv (N, 24, 2) pixels and xc (N, 24, 3) camera coords."""
    Xw = np.asarray(d["jointPositions"][actor], np.float64).reshape(-1, 24, 3)
    cam = np.asarray(d["cam_poses"], np.float64)            # (N, 4, 4) world -> cam
    K = np.asarray(d["cam_intrinsics"], np.float64)
    xc = np.einsum("nij,nkj->nki", cam[:, :3, :3], Xw) + cam[:, None, :3, 3]
    p = np.einsum("ij,nkj->nki", K, xc)
    z = p[..., 2:3]
    z = np.where(np.abs(z) < 1e-8, 1e-8, z)
    return (p[..., :2] / z).astype(np.float32), xc.astype(np.float32)


def boxes_from_points(uv, scale, conf=None, conf_thr=0.1, fallback=None):
    """uv (N, J, 2) -> (N, 3) square boxes (cx, cy, side)."""
    N = uv.shape[0]
    out = np.zeros((N, 3), np.float32)
    for i in range(N):
        pts = uv[i] if conf is None else uv[i][conf[i] > conf_thr]
        if len(pts) < 4:
            if fallback is None:
                raise ValueError(f"frame {i}: too few points for a box")
            out[i] = fallback[i]
            continue
        mn, mx = pts.min(0), pts.max(0)
        out[i, :2] = 0.5 * (mn + mx)
        out[i, 2] = scale * max(float((mx - mn).max()), 1.0)
    return out


def crop_intrinsics(K, box, out_size):
    """Intrinsics of a crop made by crop_one_frame (same rounding)."""
    cx, cy, side = [float(v) for v in box]
    side_i = max(1, int(round(side)))
    left = int(round(cx - side / 2.0))
    top = int(round(cy - side / 2.0))
    s = out_size / side_i
    Kc = np.array(K, np.float64).copy()
    Kc[0, 0] *= s
    Kc[1, 1] *= s
    Kc[0, 2] = (Kc[0, 2] - left) * s
    Kc[1, 2] = (Kc[1, 2] - top) * s
    return Kc


SMPL_JOINTS = ["pelvis", "l_hip", "r_hip", "spine1", "l_knee", "r_knee",
               "spine2", "l_ankle", "r_ankle", "spine3", "l_foot", "r_foot",
               "neck", "l_collar", "r_collar", "head", "l_shoulder",
               "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
               "l_hand", "r_hand"]


def joints_outside(uv, xc, H, W, margin=0.0):
    """(N, 24) bool: joint outside the image (with margin) or behind the camera."""
    return ((xc[..., 2] <= 0) |
            (uv[..., 0] < margin) | (uv[..., 0] > W - 1 - margin) |
            (uv[..., 1] < margin) | (uv[..., 1] > H - 1 - margin))


def usable_runs(usable, min_len):
    """(start, end) of runs of consecutive True frames, end EXCLUSIVE,
    keeping only runs of at least min_len frames."""
    padded = np.concatenate([[False], usable, [False]]).astype(np.int8)
    diff = np.diff(padded)
    starts = np.nonzero(diff == 1)[0]
    ends = np.nonzero(diff == -1)[0]
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e - s >= min_len]


SMPL_HEAD, SMPL_NECK = 15, 12
SMPL_ANKLE_KNEE = ((7, 4), (8, 5))          # (l_ankle, l_knee), (r_ankle, r_knee)
 
 
def extend_smpl_2d(uv, head_ext=1.0, foot_ext=0.25):
    """(N, 24, 2) -> (N, 27, 2): add an approximate head top and two soles.
 
    SMPL has neither (its highest joint is the head centre, its lowest the
    ankles / balls of the feet), so a box fitted to the raw joints cuts the
    top of the head and leaves space below the feet. Extrapolated in 2D:
        head_top = head  + head_ext * (head  - neck)
        sole     = ankle + foot_ext * (ankle - knee)
    Only used to place the crop box; nothing else sees these points.
    """
    head = uv[:, SMPL_HEAD]
    top = head + head_ext * (head - uv[:, SMPL_NECK])
    soles = [uv[:, a] + foot_ext * (uv[:, a] - uv[:, k]) for a, k in SMPL_ANKLE_KNEE]
    return np.concatenate([uv, top[:, None], soles[0][:, None], soles[1][:, None]], 1)
 
 
def hint_boxes_xyxy(uv_smpl, head_ext=1.0, foot_ext=0.25):
    """(N, 24, 2) -> (N, 4) tight xyxy person box from the projected SMPL
    joints (+ head top / soles). WHAM uses it only to pick which detection is
    this actor, or as the ViTPose box with --wham-box gt."""
    p = extend_smpl_2d(uv_smpl, head_ext, foot_ext)
    return np.concatenate([p.min(1), p.max(1)], 1).astype(np.float32)


def actor_boxes(d, actor, uv_smpl, source, scale, head_ext=1.0, foot_ext=0.25):
    box_smpl = boxes_from_points(extend_smpl_2d(uv_smpl, head_ext, foot_ext), scale)
    if source == "smpl":
        return box_smpl
    # OpenPose-style 2D detections shipped with 3DPW: (N, 3, 18) = x, y, conf.
    p2d = np.asarray(d["poses2d"][actor], np.float32).transpose(0, 2, 1)
    return boxes_from_points(p2d[..., :2], scale, conf=p2d[..., 2],
                             fallback=box_smpl)


# ─────────────────────────────────────────────────────────────────────────────
# Extraction
# ────────────────────────────────────────────────────────────────────────────
def extract_sequence(extractor, paths, boxes, masks, batch_size, workers):
    """Crop actor a at frame i only where masks[a, i]. Each image is read once,
    and only if some actor needs it.

    Returns per-actor features whose rows are the frames np.nonzero(masks[a])[0]
    in order, and the number of images read.
    """
    A = len(boxes)
    need = np.nonzero(masks.any(0))[0]
    outs = [[] for _ in range(A)]
    buf, owner = [], []

    def flush():
        f = extractor._run(buf)
        own = np.asarray(owner)
        for a in range(A):
            sel = np.nonzero(own == a)[0]
            if len(sel):
                outs[a].append(f[sel])
        buf.clear()
        owner.clear()

    n = 0
    for i, img in zip(need, prefetch_images([paths[i] for i in need], workers)):
        fill = img.reshape(-1, 3).mean(0)          # same pad as the H36M crops
        for a in range(A):
            if not masks[a, i]:
                continue
            cx, cy, side = boxes[a][i]
            buf.append(crop_one_frame(img, (cx, cy), 0.0, 0.0, side,
                                      extractor.out_size, pad_value=fill))
            owner.append(a)
        n += 1
        if len(buf) >= batch_size:
            flush()
    if buf:
        flush()
    return [np.concatenate(o, 0).astype(np.float32) if o
            else np.zeros((0, extractor.feat_dim), np.float32) for o in outs], n


def extract_sequence_wham(extractor, paths, actors, fps, workers, seq=""):
    """WHAM counterpart of extract_sequence. For each actor and each of its
    runs, the run's frames go through the whole WHAM pipeline as one clip.

    Returns per-actor phi whose rows are the frames np.nonzero(mask)[0] in
    order (same contract as extract_sequence), the number of distinct frames
    covered, and extras = {kind: [per-actor array, same rows]}.
    """
    feats = []
    extras = {k: [] for k in extractor.extra_kinds}
    for act in actors:
        per = {k: [] for k in ("phi",) + tuple(extractor.extra_kinds)}
        for st, en in act["runs"]:
            run_paths = paths[st:en]
            out = extractor.extract_frames(
                lambda: prefetch_images(run_paths, workers, reader=read_bgr),
                fps=fps, hints=act["hint"][st:en], target_len=en - st,
                tag=f"{seq} actor {act['actor']} [{st}:{en}]")
            for k in per:
                per[k].append(out[k])
        feats.append(np.concatenate(per["phi"], 0).astype(np.float32))
        for k in extras:
            extras[k].append(np.concatenate(per[k], 0).astype(np.float32))
    n = int(np.stack([a["mask"] for a in actors]).any(0).sum())
    return feats, n, extras


def build_segments(s, feats, extras=None):
    """Cut each actor's features into its runs. Segments with the same frame
    range (both actors usable over exactly the same frames) share one entry."""
    ids = s["ids"]
    K = np.asarray(s["d"]["cam_intrinsics"], np.float32)
    extras = extras or {}
    by_name = {}
    for ai, (act, f) in enumerate(zip(s["actors"], feats)):
        row = np.cumsum(act["mask"]) - 1           # frame i -> row in f
        for st, en in act["runs"]:
            name = f"{s['seq']}_{int(ids[st])}_{int(ids[en - 1])}"
            sg = by_name.get(name)
            if sg is None:
                sg = by_name[name] = dict(name=name, feats=[],
                                          extra={k: [] for k in extras},
                                          meta=dict(
                    source_seq=s["seq"],
                    frame_start=int(ids[st]),
                    frame_end=int(ids[en - 1]),
                    actor_ids=[],
                    img_frame_ids=ids[st:en],
                    valid=[], box=[], joints2d=[], joints_cam=[],
                    K=K,
                    image_hw=s["hw"],
                ))
            r0 = int(row[st])
            sg["feats"].append(f[r0:r0 + (en - st)])
            for k, per_actor in extras.items():
                sg["extra"][k].append(per_actor[ai][r0:r0 + (en - st)])
            m = sg["meta"]
            m["actor_ids"].append(act["actor"])
            m["valid"].append(act["valid"][st:en])
            m["box"].append(act["box"][st:en])
            m["joints2d"].append(act["uv"][st:en])
            m["joints_cam"].append(act["xc"][st:en])
    return list(by_name.values())


def feature_sanity(feat):
    f = feat.astype(np.float32)
    if f.shape[0] < 50:
        return None
    n = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
    adj = float((n[:-1] * n[1:]).sum(1).mean())
    k = f.shape[0] // 2
    far = float((n[:k] * n[k:2 * k]).sum(1).mean())
    return adj, far


def dump_debug_crops(dbg_dir, seq, paths, actors, out_size, vit,
                     every=100, limit=6):
    os.makedirs(dbg_dir, exist_ok=True)
    frames = np.nonzero(np.stack([a["mask"] for a in actors]).any(0))[0][::every]
    written = 0
    for i in frames[:limit]:
        img = read_rgb(paths[i])
        for act in actors:
            if not act["mask"][i]:
                continue
            b = act["box"][i]
            crop = crop_one_frame(img, b[:2], 0.0, 0.0, b[2], out_size)
            vis = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            if vit and out_size == 256:
                cv2.rectangle(vis, (32, 0), (223, 255), (0, 255, 255), 1)
            cv2.imwrite(os.path.join(dbg_dir, f"{seq}_a{act['actor']}_{i:05d}.png"),
                        vis)
        written += 1
    print(f"[debug] wrote crops for {written} frames to {dbg_dir}")
    if vit:
        print("[debug] yellow box = centre 192 columns the ViT sees; "
              "if limbs leave it, raise --scale")


# ─────────────────────────────────────────────────────────────────────────────
# Scratch cache (one shard per source sequence, holding all its segments)
# ─────────────────────────────────────────────────────────────────────────────
def shard_path(shard_dir, split, seq):
    return os.path.join(shard_dir, split, f"{seq}.npz")


def save_shard(path, segments, info):
    """segments: list of {name, feats, meta}; may be empty (nothing usable)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arrays = {f"s{i}_a{k}": f
              for i, sg in enumerate(segments) for k, f in enumerate(sg["feats"])}
    for i, sg in enumerate(segments):
        for kind, fs in sg.get("extra", {}).items():
            for k, f in enumerate(fs):
                arrays[f"s{i}_a{k}__{kind}"] = f
    index = dict(info, segments=[dict(name=sg["name"], meta=sg["meta"],
                                      extra_kinds=sorted(sg.get("extra", {})))
                                 for sg in segments])
    np.savez(path, **arrays, index=np.array(index, dtype=object))


def load_shard(path, config=None):
    """Return (info, segments), or None if missing, broken, or made with other
    settings (config mismatch), so --resume never mixes runs."""
    if not os.path.isfile(path):
        return None
    try:
        z = np.load(path, allow_pickle=True)
        index = z["index"].item()
        segments = []
        for i, sg in enumerate(index["segments"]):
            meta = sg["meta"]
            A = len(meta["actor_ids"])
            feats = [z[f"s{i}_a{k}"] for k in range(A)]
            extra = {kind: [z[f"s{i}_a{k}__{kind}"] for k in range(A)]
                     for kind in sg.get("extra_kinds", [])}
            segments.append(dict(name=sg["name"], feats=feats, extra=extra,
                                 meta=meta))
    except Exception:
        return None
    if config is not None and index.get("config") != config:
        return None
    for sg in segments:
        n = len(sg["meta"]["img_frame_ids"])
        if any(f.shape[0] != n for f in sg["feats"]):
            return None
        if any(f.shape[0] != n for fs in sg["extra"].values() for f in fs):
            return None
    info = {k: v for k, v in index.items() if k != "segments"}
    return info, segments


def prepare(args, pkl, split):
    d = load_seq(pkl)
    seq = os.path.splitext(os.path.basename(pkl))[0]
    N = np.asarray(d["cam_poses"]).shape[0]
    ids, paths = frame_image_paths(args.img_dir, seq, d, N)
    im0 = cv2.imread(paths[0], cv2.IMREAD_COLOR)
    if im0 is None:
        raise IOError(f"Cannot read image: {paths[0]}")
    H, W = im0.shape[:2]
    n_actors = len(d["poses"])
    s = dict(d=d, seq=seq, N=N, n_actors=n_actors, ids=ids, paths=paths,
             hw=(H, W), actors=[])
    for a in range(n_actors):
        valid = np.asarray(d["campose_valid"][a]).astype(bool)
        u, x = project_joints(d, a)
        out = joints_outside(u, x, H, W, args.margin)
        inside = ~out.any(1)
        usable = valid if args.keep_out_of_image else valid & inside
        runs = usable_runs(usable, args.min_len)
        mask = np.zeros(N, bool)
        for st, en in runs:
            mask[st:en] = True
        msg = (f"  {split}/{seq} actor {a}: campose_valid {valid.sum()}/{N}, "
               f"inside {inside.sum()}/{N} -> {len(runs)} segment(s), "
               f"{mask.sum()} frames kept")
        if not args.keep_out_of_image and (valid & ~inside).any():
            worst = int(out[valid].sum(0).argmax())
            msg += f" (joint most often outside: {SMPL_JOINTS[worst]})"
        print(msg)
        if not runs:
            continue
        s["actors"].append(dict(
            actor=a, uv=u, xc=x, valid=valid, runs=runs, mask=mask,
            box=actor_boxes(d, a, u, args.box_source, args.scale),
            hint=hint_boxes_xyxy(u)))
    return s



def process_todo(args, todo, config, shard_dir, device=None, tag=""):
    """Extract every (split, pkl) in todo into its shard. Runs in the main
    process, or in one worker process per GPU with --gpus."""
    if device is not None:
        args.device = device
    store_dtype = np.float16 if args.store_dtype == "float16" else np.float32
    extractor = build_extractor(args.extractor, args)
    batch_size = args.batch_size or extractor.default_batch_size
    print(f"Extractor {args.extractor}: crop={extractor.out_size} "
          f"dim={extractor.feat_dim} batch={batch_size} box={args.box_source} "
          f"min_len={args.min_len}\n")

    t0, failed, reported = time.time(), [], False
    for k, (split, pkl) in enumerate(todo, 1):
        seq = os.path.splitext(os.path.basename(pkl))[0]
        sp = shard_path(shard_dir, split, seq)
        if args.resume and load_shard(sp, config) is not None:
            print(f"{tag}[skip] {split}/{seq}")
            continue
        try:
            s = prepare(args, pkl, split)
        except Exception as e:
            print(f"{tag}[ERROR] {pkl}: {e}")
            failed.append((split, seq))
            continue
        N, paths, actors = s["N"], s["paths"], s["actors"]
        info = dict(config=config, seq=seq, n_actors=s["n_actors"], n_frames=N)

        if not actors:
            save_shard(sp, [], info)           # recorded as processed, nothing kept
            print(f"{tag}[{k}/{len(todo)}] {split}/{seq}  no usable segment")
            continue

        masks = np.stack([a["mask"] for a in actors])
        need = np.nonzero(masks.any(0))[0]
        missing = [paths[i] for i in need if not os.path.isfile(paths[i])]
        if missing:
            print(f"{tag}[ERROR] {split}/{seq}: {len(missing)} images missing "
                  f"(first: {missing[0]})")
            failed.append((split, seq))
            continue

        try:
            if getattr(extractor, "sequence_level", False):
                feats, n, extras = extract_sequence_wham(
                    extractor, paths, actors, args.wham_fps, args.workers, seq)
            else:
                feats, n = extract_sequence(extractor, paths,
                                            [a["box"] for a in actors], masks,
                                            batch_size, args.workers)
                extras = {}
        except Exception as e:
            print(f"{tag}[ERROR] {split}/{seq}: {e}")
            failed.append((split, seq))
            continue
        if n != len(need):
            print(f"{tag}[ERROR] {split}/{seq}: read {n} frames, expected {len(need)}")
            failed.append((split, seq))
            continue

        if not reported:
            san = feature_sanity(feats[0])
            if san:
                print(f"[sanity] cosine adjacent={san[0]:.4f} far={san[1]:.4f} "
                      f"(far should be clearly lower)")
                reported = True

        segments = build_segments(s, feats, extras)
        for sg in segments:
            sg["feats"] = [f.astype(store_dtype) for f in sg["feats"]]
            sg["extra"] = {k: [f.astype(store_dtype) for f in fs]
                           for k, fs in sg["extra"].items()}
        save_shard(sp, segments, info)
        print(f"{tag}[{k}/{len(todo)}] {split}/{seq}  {len(segments)} segment(s) "
              f"from actor(s) {[a['actor'] for a in actors]}, "
              f"{n}/{N} images read  ({time.time() - t0:.0f}s)")

    extractor.close()
    sys.stdout.flush()


def _worker(rank, gpu, args, todo, config, shard_dir):
    # Only this GPU is visible to the worker, so it is cuda:0 inside.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    process_todo(args, todo, config, shard_dir, device="cuda:0",
                 tag=f"[gpu{gpu}] ")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="3DPW root folder")
    ap.add_argument("--img-dir", default=None,
                    help="Image folder (default: <root>/imageFiles)")
    ap.add_argument("--splits", nargs="+",
                    default=["train", "validation", "test"])
    ap.add_argument("--ckpt", default=None,
                    help="HMR2.0 .ckpt (or resnet_v2 ckpt for --extractor "
                         "resnet_v2). Required except for --extractor wham, "
                         "where it is the HMR2 ckpt inside WHAM "
                         "(default: models.wham.HMR2A_CKPT).")
    ap.add_argument("--extractor", default="hmr2_vit_mean",
                    choices=sorted(EXTRACTORS))
    ap.add_argument("--out", required=True,
                    help="Output .npy; metadata goes to <out>_meta.npy")
    ap.add_argument("--box-source", default="smpl", choices=["smpl", "openpose"],
                    help="smpl: projected GT joints (clean, every frame). "
                         "openpose: the 2D detections in the .pkl, falling back "
                         "to smpl where too few joints are detected.")
    ap.add_argument("--scale", type=float, default=1.5,
                    help="Box side = scale * max extent of the 2D joints. SMPL "
                         "joints have no head top / soles, so keep a margin.")
    ap.add_argument("--margin", type=float, default=0.0,
                    help="Pixels from the image border that also count as "
                         "outside when deciding if a frame is usable.")
    ap.add_argument("--min-len", type=int, default=30,
                    help="Shortest segment kept, in frames. Set it to at least "
                         "the span of your input+prediction window.")
    ap.add_argument("--keep-out-of-image", action="store_true",
                    help="Do not require joints inside the image; segments are "
                         "then cut on campose_valid only.")
    ap.add_argument("--out-size", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8,
                    help="JPEG decoding threads")
    ap.add_argument("--device", default=None)
    ap.add_argument("--gpus", default=None,
                    help="comma-separated GPU ids, e.g. 6,7: one worker per GPU, "
                         "sequences split between them, ONE output with all "
                         "splits. Overrides --device.")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--hmr2_cfg", default=None)
    ap.add_argument("--ief-iters", type=int, default=1)
    ap.add_argument("--store-dtype", default="float32",
                    choices=["float16", "float32"])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--keep-shards", action="store_true")
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--debug-crops", default=None,
                    help="Dump sample crops of the first sequence and exit.")

    w = ap.add_argument_group("WHAM (--extractor wham)")
    w.add_argument("--wham-ckpt", default=None,
                   help="default: models.wham.WHAM_CKPT")
    w.add_argument("--wham-detector", choices=["vitpose", "yolo"],
                   default="vitpose")
    w.add_argument("--wham-root", default=None,
                   help="WHAM checkout, needed for --wham-detector vitpose")
    w.add_argument("--yolo-pose-ckpt", default="yolov8x-pose.pt")
    w.add_argument("--wham-init", choices=["smpl", "zeros"], default="smpl")
    w.add_argument("--smpl-dir", default=None,
                   help="SMPL model folder for the init (default: the WHAM one)")
    w.add_argument("--wham-box", choices=["det", "gt"], default="det",
                   help="det: YOLO box matched by IoU to the projected SMPL "
                        "box (GT only gives identity). gt: the projected SMPL "
                        "box is the ViTPose person box (vitpose only).")
    w.add_argument("--wham-fps", type=float, default=30.0)
    w.add_argument("--det-batch", type=int, default=16)
    w.add_argument("--wham-cache-img-feats", action="store_true",
                   help="also write <out>_img.npy (HMR2 tokens, large)")
    args = ap.parse_args()
    if args.ckpt is None and args.extractor != "wham":
        ap.error("--ckpt is required for --extractor " + args.extractor)
    if args.extractor == "wham":
        if args.wham_detector == "vitpose" and not args.wham_root:
            ap.error("--wham-detector vitpose needs --wham-root")
        if args.wham_box == "gt" and args.wham_detector != "vitpose":
            ap.error("--wham-box gt only applies to --wham-detector vitpose")
    if args.img_dir is None:
        args.img_dir = os.path.join(args.root, "imageFiles")
    return args


def main():
    args = parse_args()
    store_dtype = np.float16 if args.store_dtype == "float16" else np.float32
    out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
    meta_out = os.path.splitext(args.out)[0] + "_meta.npy"
    shard_dir = os.path.join(out_dir, f".cache_3dpw_{args.extractor}")
    # Settings that change what gets saved; a cached sequence made with other
    # values is redone instead of reused.
    config = dict(extractor=args.extractor, scale=args.scale,
                  box_source=args.box_source, margin=args.margin,
                  min_len=args.min_len,
                  keep_out_of_image=args.keep_out_of_image,
                  out_size=args.out_size, store_dtype=args.store_dtype)
    is_wham = args.extractor == "wham"
    if is_wham:
        config.update(wham_ckpt=args.wham_ckpt, ckpt=args.ckpt,
                      wham_detector=args.wham_detector, wham_box=args.wham_box,
                      wham_init=args.wham_init, wham_fps=args.wham_fps,
                      wham_cache_img_feats=args.wham_cache_img_feats)

    todo = []
    for split in args.splits:
        d = find_split_dir(args.root, split)
        if d is None:
            print(f"[warn] no .pkl files for split '{split}' under {args.root}")
            continue
        todo += [(split, p) for p in sorted(glob.glob(os.path.join(d, "*.pkl")))]
    if not todo:
        sys.exit("[ERROR] no sequences found")

    if args.debug_crops:
        for split, pkl in todo:
            s = prepare(args, pkl, split)
            if s["actors"]:
                break
        else:
            sys.exit("[ERROR] no usable segment in any sequence")
        native = EXTRACTORS[args.extractor].out_size
        dump_debug_crops(args.debug_crops, s["seq"], s["paths"], s["actors"],
                         args.out_size or native,
                         vit=args.extractor.startswith("hmr2"))
        if is_wham:
            print("[debug] wham: these are the identity boxes only; WHAM "
                  "crops from its own detections")
        return

    gpus = [g.strip() for g in (args.gpus or "").split(",") if g.strip()]
    if len(gpus) > 1:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        procs = []
        for r, g in enumerate(gpus):
            p = ctx.Process(target=_worker,
                            args=(r, g, args, todo[r::len(gpus)], config, shard_dir))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
        bad = [g for g, p in zip(gpus, procs) if p.exitcode != 0]
        if bad:
            print(f"[warn] worker(s) on GPU {bad} exited non-zero; their "
                  f"sequences show up as missing below. Re-run with --resume.")
    elif len(gpus) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus[0]
        process_todo(args, todo, config, shard_dir, device="cuda:0")
    else:
        process_todo(args, todo, config, shard_dir)

    # ── assemble ────────────────────────────────────────────────────────────
    print("\nAssembling...")
    feats_all, meta_all, missing = {}, {}, []
    extra_all = {}
    stats = {}
    for split, pkl in todo:
        seq = os.path.splitext(os.path.basename(pkl))[0]
        got = load_shard(shard_path(shard_dir, split, seq), config)
        if got is None:
            missing.append((split, seq))
            continue
        info, segments = got
        st = stats.setdefault(split, dict(seqs=0, seqs_used=0, segs=0,
                                          frames=0, kept=0))
        st["seqs"] += 1
        st["seqs_used"] += bool(segments)
        st["segs"] += len(segments)
        st["frames"] += info["n_frames"] * info["n_actors"]
        for sg in segments:
            st["kept"] += sum(f.shape[0] for f in sg["feats"])
            feats_all.setdefault(split, {})[sg["name"]] = sg["feats"]
            meta_all.setdefault(split, {})[sg["name"]] = sg["meta"]
            for kind, fs in sg["extra"].items():
                extra_all.setdefault(kind, {}).setdefault(split, {})[sg["name"]] = fs

    if missing and not args.allow_partial:
        print(f"[ERROR] {len(missing)} sequence(s) missing:")
        for s_, q in missing[:20]:
            print(f"  {s_}/{q}")
        print(f"Nothing written; cache intact at {shard_dir}. Fix and re-run "
              f"with --resume, or pass --allow-partial.")
        sys.exit(1)

    np.save(args.out, feats_all)
    np.save(meta_out, meta_all)
    print(f"Saved {args.out} ({os.path.getsize(args.out) / 1024 ** 3:.2f} GB)")
    print(f"Saved {meta_out}")
    for kind, d_ in sorted(extra_all.items()):
        p = os.path.splitext(args.out)[0] + f"_{kind}.npy"
        np.save(p, d_)
        print(f"Saved {p}  (same layout as {args.out})")
    for split, st in stats.items():
        frac = st["kept"] / max(st["frames"], 1)
        print(f"  {split}: {st['segs']} segments from {st['seqs_used']}/"
              f"{st['seqs']} sequences, {st['kept']}/{st['frames']} actor-frames "
              f"kept ({frac:.0%})")
    print("\nLoad with: np.load(path, allow_pickle=True).item()[split][segment][actor]")

    if not args.keep_shards:
        shutil.rmtree(shard_dir, ignore_errors=True)


if __name__ == "__main__":
    main()