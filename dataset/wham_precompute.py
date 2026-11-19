from __future__ import absolute_import, division, print_function

import argparse
import json
import multiprocessing as mp
import os
import os.path as osp
import pickle
import shutil
import sys
import time
from collections import defaultdict

import cv2
import numpy as np
import scipy.signal as signal
import torch

from models.wham import (WHAMFeatureNet, build_wham_feature_net, HMR2A_CKPT,
                  WHAM_CKPT, KEYPOINTS_THR, D_CONTEXT, D_FEAT_VIT)
from models.smpl import SMPL

# =========================================================================== #
#  VERBATIM WHAM 
# =========================================================================== #


FLDR = f'/home/ferreiral1/WHAM/dataset/body_models/smpl/'

def build_body_model(device, batch_size=1, gender='neutral', **kwargs):
    body_model = SMPL(
        model_path=FLDR,
        gender=gender,
        batch_size=batch_size,
        create_transl=False).to(device)
    return body_model

IMG_NORM_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMG_NORM_STD = np.array([0.229, 0.224, 0.225], np.float32)

VIS_THRESH = 0.3        # detector.py
BBOX_S_FACTOR = 1.2     # detector.compute_bboxes_from_keypoints
KP_PATCH = 224          # dataset_custom: normalizer called with 224, 224
CROP_SIZE = 256         # demo.py: extractor.run(patch_h=256, patch_w=256)

MIN_BBOX_SCALE = 0.05   # NOT WHAM: guard against degenerate (all-zero) frames


def get_transform(center, scale, res, rot=0):
    """imutils.get_transform / backbone.utils.get_transform (identical)."""
    crop_aspect_ratio = res[0] / float(res[1])
    h = 200 * scale
    w = h / crop_aspect_ratio
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / w
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / w + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if rot != 0:
        raise NotImplementedError('rot is never non-zero on the demo path')
    return t


def transform(pt, center, scale, res, invert=0, rot=0):
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return np.array([round(new_pt[0]), round(new_pt[1])], dtype=int) + 1


def crop(img, center, scale, res):
    """backbone/utils.py crop(). Zero-filled outside the image, not mean-filled."""
    ul = np.array(transform([1, 1], center, scale, res, invert=1)) - 1
    br = np.array(transform([res[1] + 1, res[0] + 1], center, scale, res, invert=1)) - 1

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape, dtype=np.float32)

    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = \
        img[old_y[0]:old_y[1], old_x[0]:old_x[1]]

    return cv2.resize(new_img, (res[1], res[0])), ul, br


def process_image(orig_img_rgb, center, scale, crop_height=256, crop_width=256):
    """backbone/utils.py process_image, blur=False, do_crop=True."""
    img, _, _ = crop(orig_img_rgb, center, scale, (crop_height, crop_width))
    img = img / 255.
    norm_img = (img - IMG_NORM_MEAN) / IMG_NORM_STD
    return np.transpose(norm_img, (2, 0, 1)).astype(np.float32)


def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height,
                            dst_width, dst_height, scale, rot, inv=False):
    """imutils.gen_trans_from_patch_cv, scale=1.0 and rot=0 on this path."""
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], np.float32)

    rot_rad = np.pi * rot / 180
    src_downdir = _rotate_2d(np.array([0., src_h * 0.5], np.float32), rot_rad)
    src_rightdir = _rotate_2d(np.array([src_w * 0.5, 0.], np.float32), rot_rad)

    dst_w, dst_h = dst_width, dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], np.float32)
    dst_downdir = np.array([0., dst_h * 0.5], np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0.], np.float32)

    src = np.zeros((3, 2), np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    if inv:
        return cv2.getAffineTransform(np.float32(dst), np.float32(src))
    return cv2.getAffineTransform(np.float32(src), np.float32(dst))


def _rotate_2d(pt_2d, rot_rad):
    x, y = pt_2d
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    return np.array([x * cs - y * sn, x * sn + y * cs], dtype=np.float32)


def trans_point2d(pt_2d, trans):
    src_pt = np.array([pt_2d[0], pt_2d[1], 1.]).T
    dst_pt = np.dot(trans, src_pt)
    return dst_pt[0:2]


def transform_keypoints(kp_2d, bbox, patch_width, patch_height):
    """imutils.transform_keypoints. Note scale=1.0, rot=0 (hardcoded there)."""
    center_x, center_y, scale = bbox[:3]
    width = height = scale * 200
    trans = gen_trans_from_patch_cv(center_x, center_y, width, height,
                                    patch_width, patch_height, 1.0, 0, inv=False)
    kp_2d = kp_2d.copy()
    for n in range(kp_2d.shape[0]):
        kp_2d[n] = trans_point2d(kp_2d[n], trans)
    return kp_2d, trans


def normalize_keypoints_to_patch(kp_2d, crop_size=224):
    return 2.0 * kp_2d * (1.0 / crop_size) - 1.0


def normalize_keypoints_to_image(x, res):
    """normalizer.normalize_keypoints_to_image, x (N, 1, 2), res (2,)."""
    scale = res.max(-1)[0].reshape(-1)
    mean = torch.stack([res[..., 0] / scale, res[..., 1] / scale], dim=-1)
    return (2 * x / scale.reshape(*[1 for _ in x.shape[1:]])
            - mean.reshape(*[1 for _ in x.shape[1:-1]], -1))


def bbox_normalization(kp_2d, bbox, res, patch_width=224, patch_height=224):
    """normalizer.Normalizer.bbox_normalization -> (N, 37).

    kp_2d (N, 17, 2) pixels, bbox (N, 3) as [cx, cy, scale] with scale in
    units of 200 px, res (2,) = [width, height].
    """
    out = np.zeros_like(kp_2d)
    for i in range(len(out)):
        out[i] = transform_keypoints(kp_2d[i], bbox[i][:3],
                                     patch_width, patch_height)[0]
        out[i] = normalize_keypoints_to_patch(out[i], patch_width)

    out = torch.from_numpy(out).float()
    bb = torch.from_numpy(bbox).float()

    centers = normalize_keypoints_to_image(bb[:, :2].unsqueeze(1), res).squeeze(1)
    scale = bb[:, 2:] * 200 / res.max()
    location = torch.cat((centers, scale), dim=-1)

    out = out.reshape(out.shape[0], -1)
    return torch.cat((out, location), dim=-1)          # (N, 37)


def compute_bboxes_from_keypoints(kp, fps, s_factor=BBOX_S_FACTOR, smooth=True):
    """detector.compute_bboxes_from_keypoints + the median filter from
    detector.process(). kp (N, 17, 3) -> bbox (N, 3) = [cx, cy, scale]."""
    mask = kp[..., -1] > VIS_THRESH
    bbox = np.zeros((len(kp), 3))
    for i, (k, m) in enumerate(zip(kp, mask)):
        if m.sum() == 0:
            m = np.ones(len(k), bool)      # degenerate frame: use everything
        bb = [k[m, 0].min(), k[m, 1].min(), k[m, 0].max(), k[m, 1].max()]
        cx, cy = (bb[2] + bb[0]) / 2, (bb[3] + bb[1]) / 2
        s = max(bb[2] - bb[0], bb[3] - bb[1])
        bbox[i] = (cx, cy, s)
    bbox[:, 2] = bbox[:, 2] * s_factor / 200.0

    # FIX: a frame with no usable keypoints gives s = 0, which makes
    # get_transform divide by h = 200 * 0 and makes crop() build a zero-size
    # array that cv2.resize rejects. Fall back to the sequence median.
    bad = bbox[:, 2] < MIN_BBOX_SCALE
    if bad.any():
        good = ~bad
        fill = np.median(bbox[good, 2]) if good.any() else 1.0
        bbox[bad, 2] = max(fill, MIN_BBOX_SCALE)
        if good.any():
            for j in np.where(bad)[0]:
                k = np.argmin(np.abs(np.where(good)[0] - j))
                bbox[j, :2] = bbox[np.where(good)[0][k], :2]

    if smooth:
        kernel = int(int(fps / 2) / 2) * 2 + 1
        bbox = np.array([signal.medfilt(p, kernel) for p in bbox.T]).T
    return bbox


def root_centering_coco(x):
    """kp_utils.root_centering for joint_type='coco': COCO hips are 11/12."""
    return x - x[..., [12, 11], :].mean(-2, keepdims=True)


# =========================================================================== #
# =========================================================================== #
#  Frame sources (mp4 for H36M, a list of JPGs for 3DPW)
# =========================================================================== #

def iter_frames(source):
    """Yield BGR frames from a video path, or from a list of image paths."""
    if isinstance(source, (list, tuple)):
        for p in source:
            img = cv2.imread(p)
            if img is None:
                raise IOError(f'cannot read {p}')
            yield img
        return
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise IOError(f'cannot open {source}')
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def source_tag(source):
    if isinstance(source, (list, tuple)):
        return osp.basename(osp.dirname(source[0])) if source else '<empty>'
    return osp.basename(source)


# =========================================================================== #
#  Detectors
# =========================================================================== #

MATCH_IOU = 0.3   # 3DPW: min IoU between a detection and the GT actor box


def box_iou(a, b):
    """a (M, 4) xyxy, b (4,) xyxy -> (M,)"""
    x0 = np.maximum(a[:, 0], b[0])
    y0 = np.maximum(a[:, 1], b[1])
    x1 = np.minimum(a[:, 2], b[2])
    y1 = np.minimum(a[:, 3], b[3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


class YoloPoseDetector:
    """ultralytics yolov8x-pose -> COCO-17 with confidences. One dependency.

    Not what WHAM was trained on. Keypoint noise here becomes pose noise the
    LSTM integrates over time, so treat this as the "get it running" path.

    hints: optional per-frame xyxy box (or None). When given, the detection
    with the highest IoU against it is kept instead of the most confident one
    (3DPW has two actors in some sequences; this keeps identity).
    """
    name = 'yolo'

    def __init__(self, ckpt='yolov8x-pose.pt', device='cuda', conf=0.5):
        from ultralytics import YOLO
        self.model = YOLO(ckpt)
        self.device = device
        self.conf = conf

    def __call__(self, frames_bgr, hints=None):
        if hints is None:
            hints = [None] * len(frames_bgr)
        res = self.model.predict(frames_bgr, device=self.device, conf=self.conf,
                                 verbose=False, save=False)
        out = []
        for r, hint in zip(res, hints):
            if r.keypoints is None or len(r.boxes) == 0:
                out.append(None)
                continue
            if hint is not None:
                ious = box_iou(r.boxes.xyxy.cpu().numpy(), hint)
                j = int(ious.argmax())
                if ious[j] < MATCH_IOU:
                    out.append(None)
                    continue
            else:
                # single-person: highest box confidence
                j = int(r.boxes.conf.argmax())
            xy = r.keypoints.xy[j].cpu().numpy()                    # (17, 2)
            c = (r.keypoints.conf[j].cpu().numpy()
                 if r.keypoints.conf is not None else np.ones(17, np.float32))
            out.append(np.concatenate([xy, c[:, None]], -1).astype(np.float32))
        return out


class ViTPoseDetector:
    """YOLOv8x boxes + ViTPose-H through mmpose. What WHAM's demo uses.

    Needs WHAM's third-party/ViTPose checkout and its mmpose fork; the config
    and checkpoint names below are the ones detector.py hardcodes.

    hints: optional per-frame xyxy box (or None). With hint_as_box=False the
    YOLO box with the highest IoU against the hint is used; with
    hint_as_box=True the hint itself is the ViTPose person box (no YOLO).
    """
    name = 'vitpose'

    def __init__(self, wham_root, device='cuda', bbox_conf=0.5, hint_as_box=False):
        from ultralytics import YOLO
        from mmpose.apis import inference_top_down_pose_model, init_pose_model
        self._infer = inference_top_down_pose_model

        cfg = osp.join(wham_root, 'third-party/ViTPose/configs/body/'
                       '2d_kpt_sview_rgb_img/topdown_heatmap/coco/'
                       'ViTPose_huge_coco_256x192.py')
        ckpt = osp.join(wham_root, 'checkpoints', 'vitpose-h-multi-coco.pth')
        for p in (cfg, ckpt):
            if not osp.exists(p):
                raise FileNotFoundError(p)

        self.pose_model = init_pose_model(cfg, ckpt, device=device.lower())
        self.bbox_model = YOLO(osp.join(wham_root, 'checkpoints', 'yolov8x.pt'))
        self.device = device
        self.bbox_conf = bbox_conf
        self.hint_as_box = hint_as_box

    def __call__(self, frames_bgr, hints=None):
        if hints is None:
            hints = [None] * len(frames_bgr)
        out = []
        for img, hint in zip(frames_bgr, hints):
            if hint is not None and self.hint_as_box:
                box = np.asarray(hint, np.float32)
            else:
                boxes = self.bbox_model.predict(
                    img, device=self.device, classes=0, conf=self.bbox_conf,
                    save=False, verbose=False)[0].boxes
                if len(boxes) == 0:
                    out.append(None)
                    continue
                xyxy = boxes.xyxy.detach().cpu().numpy()
                if hint is not None:
                    ious = box_iou(xyxy, hint)
                    j = int(ious.argmax())
                    if ious[j] < MATCH_IOU:
                        out.append(None)
                        continue
                else:
                    # single-person H36M: largest box, no tracker
                    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
                    j = int(areas.argmax())
                box = xyxy[j]
            person = [{'bbox': box}]

            pose_results, _ = self._infer(self.pose_model, img,
                                          person_results=person, format='xyxy',
                                          return_heatmap=False, outputs=None)
            out.append(None if not pose_results
                       else pose_results[0]['keypoints'].astype(np.float32))
        return out


# =========================================================================== #
#  Per-video pipeline
# =========================================================================== #

def _hint_at(hints, idx):
    if hints is None or idx >= len(hints):
        return None
    h = hints[idx]
    return None if np.isnan(h).any() else h


@torch.no_grad()
def detect_video(source, detector, frame_stride=1, batch=16, max_frames=None,
                 hints=None):
    """-> kp (N, 17, 3) COCO pixels+conf, res (w, h), n_read.

    hints: optional (N_src, 4) xyxy array indexed by SOURCE frame (before
    stride), NaN where unknown.
    """
    kps, buf, hbuf, idx, last, res = [], [], [], 0, None, None
    for frame in iter_frames(source):
        if res is None:
            res = (frame.shape[1], frame.shape[0])
        if idx % frame_stride == 0:
            buf.append(frame)
            hbuf.append(_hint_at(hints, idx))
            if len(buf) == batch:
                kps += detector(buf, hbuf)
                buf, hbuf = [], []
            if max_frames and len(kps) >= max_frames:
                break
        idx += 1
    if buf:
        kps += detector(buf, hbuf)
    if res is None:
        res = (0, 0)

    # Frames with no detection: hold the previous pose with zero confidence,
    # so the mask marks them and mask_embedding takes over. Dropping them
    # instead would desynchronise the cache from your 2D/3D arrays.
    out = []
    for k in kps:
        if k is None:
            k = (last.copy() if last is not None else np.zeros((17, 3), np.float32))
            k[:, 2] = 0.0
        else:
            last = k
        out.append(k)
    if not out:
        return np.zeros((0, 17, 3), np.float32), res, idx
    return np.stack(out).astype(np.float32), res, idx


@torch.no_grad()
def encode_video_images(source, bbox, image_encoder, device,
                        frame_stride=1, batch=32, n_expected=None):
    """WHAM's crop -> HMR2 crossattn token. -> (N, 1024), first_crop.

    Also hands back the frame-0 crop, so the SMPL init no longer needs its own
    decode pass over the video.
    """
    feats, buf, idx, kept = [], [], 0, 0
    first_crop = None

    def flush():
        if not buf:
            return
        t = torch.from_numpy(np.stack(buf)).to(device)
        feats.append(image_encoder(t, tap='crossattn').float().cpu())

    for frame in iter_frames(source):
        if idx % frame_stride == 0:
            if n_expected is not None and kept >= n_expected:
                break
            cx, cy, scale = bbox[kept]
            crop_img = process_image(frame[..., ::-1], [cx, cy], scale,
                                     CROP_SIZE, CROP_SIZE)
            if kept == 0:
                first_crop = crop_img
            buf.append(crop_img)
            kept += 1
            if len(buf) == batch:
                flush()
                buf = []
        idx += 1
    flush()
    out = (torch.cat(feats, 0).numpy().astype(np.float32) if feats else
           np.zeros((0, D_FEAT_VIT), np.float32))
    return out, first_crop


def build_init_kp(norm_kp2d0, init_kp3d):
    """dataset_custom: cat(init_kp3d[51], norm_kp2d[0][37]) -> (1, 88)."""
    out = torch.cat([init_kp3d.reshape(1, -1),
                     norm_kp2d0.reshape(1, -1)], dim=-1)      # (1, 88)
    return out.unsqueeze(0)                                    # (1, 1, 88)


import torch.nn.functional as F

def rot6d_to_rotmat_hmr2(x):
    """hmr2/utils/geometry.py rot6d_to_rotmat.
    """
    x = x.reshape(-1, 2, 3).permute(0, 2, 1).contiguous()
    a1, a2 = x[:, :, 0], x[:, :, 1]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


@torch.no_grad()
def smpl_init_kp3d(image_encoder, first_crop, smpl, device):
    """extractor.predict_init + SMPL forward -> (1, 51). None se indisponível."""
    if smpl is None:
        return None
    head = getattr(image_encoder, 'smpl_head', None)
    if head is None:
        return None
    if head.input_is_mean_shape or head.ief_iters != 1:
        raise RuntimeError(
            f'este atalho assume o caminho do token zero com ief_iters=1; '
            f'got input_is_mean_shape={head.input_is_mean_shape}, '
            f'ief_iters={head.ief_iters}. Com mais iterações o loop acumula '
            'decpose e você precisa replicá-lo.')

    x = torch.from_numpy(first_crop).unsqueeze(0).to(device)
    token = image_encoder(x, tap='crossattn')               # (1, 1024)

    pose6d = head.decpose(token) + head.init_body_pose      # (1, 144)
    betas = head.decshape(token) + head.init_betas          # (1, 10)

    rotmat = rot6d_to_rotmat_hmr2(pose6d).view(1, 24, 3, 3)
    go, bp = rotmat[:, :1], rotmat[:, 1:]                   # 1 global + 23 body

    out = smpl.get_output(global_orient=go, body_pose=bp, betas=betas,
                          pose2rot=False, return_full_pose=True)
    return root_centering_coco(out.joints[:, :17]).reshape(1, -1).cpu()


# =========================================================================== #
#  Dataset layouts
#
#  Every dataset is described the same way:
#     layout[group][name] = [unit, unit, ...]
#     unit = {'unit': id, 'source': video path | [image paths] | None,
#             'gt_len': int | None, 'hint': (pkl, actor) | None}
#
#  H36M : group=subject, name=action, unit=camera id
#  3DPW : group=split,   name=sequence, unit='actor<k>'
#
#  and the merged output is out[group][name] = [unit0, unit1, ...], each a
#  (T, D_CONTEXT) array in unit order.
# =========================================================================== #

ACTION_ALIASES = {
    'Photo': ['Photo', 'TakingPhoto'],
    'WalkDog': ['WalkDog', 'WalkingDog', 'Walking Dog'],
    'WalkTogether': ['WalkTogether', 'WalkingTogether', 'Walking Together'],
    'SittingDown': ['SittingDown', 'Sitting Down'],
    'Walking': ['Walking'],
}
CAM_IDS = ['54138969', '55011271', '58860488', '60457274']


def resolve_video(h36m_root, subject, action, cam_id):
    """H36M video filenames are inconsistent across extractions, so try the
    known spellings. Returns None if nothing matches."""
    base = action.split(' ')[0]
    suffix = action[len(base):]
    cands = []
    for n in ACTION_ALIASES.get(base, [base]):
        stem = n + suffix
        cands += [stem, stem.replace(' ', '_'), stem.replace(' ', '')]
    seen = set()
    for a in cands:
        if a in seen:
            continue
        seen.add(a)
        p = osp.join(h36m_root, subject, 'Videos', f'{a}.{cam_id}.mp4')
        if osp.exists(p):
            return p
    return None


def build_h36m_layout(h36m_root, keys):
    layout = {}
    for subject in sorted(keys):
        for action in sorted(keys[subject]):
            arrs = keys[subject][action]
            units = []
            for ci, cam_id in enumerate(CAM_IDS):
                units.append({
                    'unit': cam_id,
                    'source': resolve_video(h36m_root, subject, action, cam_id),
                    'gt_len': len(arrs[ci]) if ci < len(arrs) else None,
                    'hint': None,
                })
            layout.setdefault(subject, {})[action] = units
    return layout


PW3D_SPLITS = ['train', 'validation', 'test']


def load_pw3d_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f, encoding='latin1')   # py2 pickles


def build_pw3d_layout(pw3d_root, splits):
    """<root>/sequenceFiles/<split>/<seq>.pkl + <root>/imageFiles/<seq>/*.jpg.

    One unit per actor. The frame list comes from img_frame_ids, so features
    line up 1:1 with poses2d / jointPositions / campose_valid in the pkl.
    """
    layout = {}
    for split in splits:
        d = osp.join(pw3d_root, 'sequenceFiles', split)
        if not osp.isdir(d):
            raise FileNotFoundError(d)
        for fn in sorted(os.listdir(d)):
            if not fn.endswith('.pkl'):
                continue
            seq = fn[:-4]
            pkl = osp.join(d, fn)
            data = load_pw3d_pkl(pkl)
            img_dir = osp.join(pw3d_root, 'imageFiles', seq)
            if 'img_frame_ids' in data:
                ids = np.asarray(data['img_frame_ids']).astype(int).tolist()
            else:
                ids = list(range(len(data['poses2d'][0])))
            frames = [osp.join(img_dir, f'image_{i:05d}.jpg') for i in ids]
            ok = bool(frames) and osp.exists(frames[0]) and osp.exists(frames[-1])
            units = []
            for a, p2d in enumerate(data['poses2d']):
                units.append({
                    'unit': f'actor{a}',
                    'source': frames if ok else None,
                    'gt_len': len(p2d),
                    'hint': (pkl, a),
                })
            layout.setdefault(split, {})[seq] = units
    return layout


def pw3d_hint_boxes(pkl, actor, pad=0.1):
    """GT OpenPose-18 2D of one actor -> (N, 4) xyxy boxes, padded by `pad`
    of the box size on each side. Used only to tell the actors apart (or, with
    --pw3d-box gt, as the ViTPose person box). Frames without GT 2D are filled
    from the nearest frame that has it."""
    p = np.asarray(load_pw3d_pkl(pkl)['poses2d'][actor], np.float32)
    if p.shape[-1] != 3:                     # stored as (N, 3, 18)
        p = p.transpose(0, 2, 1)             # -> (N, 18, 3)
    boxes = np.full((len(p), 4), np.nan, np.float32)
    for i, k in enumerate(p):
        m = k[:, 2] > 0
        if m.sum() < 2:
            continue
        x0, y0 = k[m, :2].min(0)
        x1, y1 = k[m, :2].max(0)
        w, h = x1 - x0, y1 - y0
        boxes[i] = [x0 - pad * w, y0 - pad * h, x1 + pad * w, y1 + pad * h]

    good = np.where(~np.isnan(boxes[:, 0]))[0]
    if len(good) and len(good) < len(boxes):
        for j in np.where(np.isnan(boxes[:, 0]))[0]:
            boxes[j] = boxes[good[np.argmin(np.abs(good - j))]]
    return boxes


def limit_layout(layout, n):
    """Keep the first n (group, name) entries, with all their units."""
    trimmed, k = {}, 0
    for g in sorted(layout):
        for a in sorted(layout[g]):
            if k >= n:
                return trimmed
            trimmed.setdefault(g, {})[a] = layout[g][a]
            k += 1
    return trimmed


class Spec:
    """One dataset to process: its layout and where its outputs go."""

    def __init__(self, name, layout, out, motion_out, cache_dir, fps, levels):
        self.name, self.layout = name, layout
        self.out, self.motion_out, self.cache_dir = out, motion_out, cache_dir
        self.fps, self.levels = fps, levels      # levels: ('subject','action','cam')


# =========================================================================== #
#  Cache layout
# =========================================================================== #

def safe(name):
    return name.replace(' ', '_').replace('/', '-')


def cam_cache_path(cache_dir, group, name, unit, kind='phi'):
    return osp.join(cache_dir, safe(group), safe(name), f'{unit}.{kind}.npy')


def atomic_save(path, arr):
    os.makedirs(osp.dirname(path), exist_ok=True)
    tmp = f'{path}.tmp{os.getpid()}.npy'
    np.save(tmp, arr, allow_pickle=False)
    os.replace(tmp, path)          # rename is atomic: no half-written caches


def target_length(gt_len, stride):
    """Frames kept are 0, stride, 2*stride, ... so this is how many of them
    cover a ground-truth sequence of gt_len frames."""
    return (gt_len + stride - 1) // stride


def fit_length(arr, T, tag=''):
    """Trim, or pad by repeating the last frame. H36M mp4s usually have a few
    MORE frames than the mocap; padding means something upstream is wrong."""
    n = len(arr)
    if n == T:
        return arr
    if n > T:
        return arr[:T]
    pad = np.repeat(arr[-1:], T - n, axis=0)
    print(f'  [warn] {tag}: only {n} frames for a target of {T}; padding '
          f'{T - n} by repeat. Check the video against the 2D array.')
    return np.concatenate([arr, pad], 0)


# =========================================================================== #
#  Per-video work
# =========================================================================== #

@torch.no_grad()
def process_one(source, target_len, detector, net, smpl, device, args,
                fps, hints=None):
    """One camera / actor ->
    (phi (T, D_CONTEXT), img_feat (T, D_FEAT_VIT), motion_feat (T, D_CONTEXT)).

    motion_feat is WHAMFeatureNet.motion_encoder's output — the motion
    context BEFORE it is fused with the image feature by net.integrator to
    produce phi. Same width as phi (D_CONTEXT) but not the same tensor.

    The whole thing runs under no_grad: without it the LSTM rollout over a
    multi-thousand-frame H36M sequence builds an autograd graph and OOMs.
    """
    kp, res_wh, _ = detect_video(source, detector,
                                 frame_stride=args.frame_stride,
                                 batch=args.det_batch, hints=hints)
    n = len(kp)
    if n == 0:
        raise RuntimeError('decoded zero frames')

    eff_fps = fps / args.frame_stride
    bbox = compute_bboxes_from_keypoints(kp, eff_fps)
    res = torch.tensor(res_wh).float()

    norm_kp2d = bbox_normalization(kp[..., :2].copy(), bbox, res,
                                   KP_PATCH, KP_PATCH)          # (N, 37)
    mask = torch.from_numpy(kp[..., -1] < KEYPOINTS_THR)        # (N, 17)

    img_feat, first_crop = encode_video_images(
        source, bbox, net.image_encoder, device,
        frame_stride=args.frame_stride, batch=args.vit_batch, n_expected=n)

    n = min(n, len(img_feat))
    norm_kp2d, mask, img_feat = norm_kp2d[:n], mask[:n], img_feat[:n]

    init_kp3d = None
    if smpl is not None and first_crop is not None:
        init_kp3d = smpl_init_kp3d(net.image_encoder, first_crop, smpl, device)
    if init_kp3d is None:
        init_kp3d = torch.zeros(1, 17 * 3)
    init_kp = build_init_kp(norm_kp2d[0], init_kp3d.cpu())

    phi, _pred_kp3d, motion_feat = net(
        norm_kp2d.unsqueeze(0).to(device),
        init_kp.to(device),
        # FIX: this used to stay on the CPU while everything else was on
        # the GPU, which throws as soon as the net touches it.
        features=torch.from_numpy(img_feat).unsqueeze(0).to(device),
        mask=mask.unsqueeze(0).to(device),
        return_extras=True)
    phi = phi[0].float().cpu().numpy()
    motion_feat = motion_feat[0].float().cpu().numpy()

    if target_len is not None:
        tag = source_tag(source)
        phi = fit_length(phi, target_len, tag)
        img_feat = fit_length(img_feat, target_len, tag)
        motion_feat = fit_length(motion_feat, target_len, tag)
    return phi, img_feat, motion_feat


def run_worker(rank, device, tasks, args):
    """One process, one GPU, its own slice of the task list."""
    if device.startswith('cuda'):
        torch.cuda.set_device(device)
    tag = f'[{device}]'
    print(f'{tag} {len(tasks)} videos', flush=True)

    if args.detector == 'vitpose':
        detector = ViTPoseDetector(args.wham_root, device=device,
                                   hint_as_box=(args.pw3d_box == 'gt'))
    else:
        detector = YoloPoseDetector(args.yolo_pose_ckpt, device=device)

    net = build_wham_feature_net(wham_ckpt=args.wham_ckpt,
                                 hmr2_ckpt=args.hmr2_ckpt, device=device)
    net.eval()

    smpl = None
    if args.init == 'smpl':
        try:
            # FIX: was built on 'cpu' and then fed CUDA tensors from the
            # image encoder.
            smpl = build_body_model(device)
        except Exception as e:
            print(f'{tag} [warn] SMPL unavailable ({e}); zero init_kp3d, so '
                  'the first frames of each clip will be off.', flush=True)

    dt = np.float16 if args.dtype == 'float16' else np.float32
    errors, t0 = defaultdict(list), time.time()

    for i, t in enumerate(tasks):
        cd, g, a, u = t['cache_dir'], t['group'], t['name'], t['unit']
        label = f"{t['dataset']}:{g}/{a}.{u}"
        try:
            hints = pw3d_hint_boxes(*t['hint']) if t['hint'] else None
            phi, img_feat, motion_feat = process_one(
                t['source'], t['tlen'], detector, net, smpl, device, args,
                fps=t['fps'], hints=hints)
            atomic_save(cam_cache_path(cd, g, a, u, 'phi'), phi.astype(dt))
            if args.cache_img_feats:
                atomic_save(cam_cache_path(cd, g, a, u, 'img'), img_feat.astype(dt))
            atomic_save(cam_cache_path(cd, g, a, u, 'motion'), motion_feat.astype(dt))
            print(f'{tag} {i+1}/{len(tasks)}  {label}  '
                  f'{len(phi)} frames  [{time.time()-t0:.0f}s]', flush=True)
        except Exception as e:
            errors[cd].append({'dataset': t['dataset'], 'group': g, 'name': a,
                               'unit': u, 'source': source_tag(t['source']),
                               'error': repr(e)})
            print(f'{tag} FAIL {label}: {e!r}', flush=True)

    for cd, errs in errors.items():
        with open(osp.join(cd, f'errors.{rank}.json'), 'w') as f:
            json.dump(errs, f, indent=1)

    close = getattr(detector, 'close', None)
    if close:
        close()


# =========================================================================== #
#  Task list / merge
# =========================================================================== #

def build_tasks(args, spec):
    """Task dicts for everything in spec not yet cached. Missing videos are
    collected, not silently skipped."""
    tasks, missing, done = [], [], 0
    for g in sorted(spec.layout):
        for a in sorted(spec.layout[g]):
            for u in spec.layout[g][a]:
                uid = u['unit']
                if u['source'] is None:
                    missing.append(f'{g}/{a}.{uid}')
                    continue
                cp = lambda k: cam_cache_path(spec.cache_dir, g, a, uid, k)
                need_img = args.cache_img_feats and not osp.exists(cp('img'))
                need_motion = not osp.exists(cp('motion'))
                if (osp.exists(cp('phi')) and not need_img and not need_motion
                        and not args.overwrite):
                    done += 1
                    continue
                tlen = (target_length(u['gt_len'], args.frame_stride)
                        if u['gt_len'] else None)
                tasks.append({'dataset': spec.name, 'cache_dir': spec.cache_dir,
                              'group': g, 'name': a, 'unit': uid,
                              'source': u['source'], 'tlen': tlen,
                              'fps': spec.fps, 'hint': u['hint']})
    return tasks, missing, done


def _assemble_kind(args, spec, kind):
    """Per-unit '<cache>/<group>/<name>/<unit>.<kind>.npy' caches -> one
    in-memory dict, shared by the 'phi' and 'motion' outputs.

        feats[group][name] = [unit0, unit1, ...]

    with each unit a (T_unit, D_CONTEXT) array. A list rather than a stacked
    array, so units keep their own lengths and line up 1:1 with the GT
    (positions_2d[subject][action][cam] for H36M, poses2d[actor] in the
    sequence pkl for 3DPW).
    """
    dt = np.float16 if args.dtype == 'float16' else np.float32
    feats, incomplete, mismatched, total_frames = {}, [], [], 0

    for g in sorted(spec.layout):
        for a in sorted(spec.layout[g]):
            units = spec.layout[g][a]
            paths = [cam_cache_path(spec.cache_dir, g, a, u['unit'], kind)
                     for u in units]
            if not all(osp.exists(p) for p in paths):
                incomplete.append(f'{g}/{a}')
                continue

            per_unit = [np.load(p).astype(dt) for p in paths]
            for u, arr in zip(units, per_unit):
                if u['gt_len'] is None:
                    continue
                expect = target_length(u['gt_len'], args.frame_stride)
                if len(arr) != expect:
                    mismatched.append(f"{g}/{a}.{u['unit']}: {len(arr)} vs {expect}")
            feats.setdefault(g, {})[a] = per_unit
            total_frames += sum(len(x) for x in per_unit)

    return feats, incomplete, mismatched, total_frames


def _write_feats(args, spec, feats, total_frames, out_path, array_key, label):
    dt = np.float16 if args.dtype == 'float16' else np.float32
    n_items = sum(len(v) for v in feats.values())
    gb = total_frames * D_CONTEXT * np.dtype(dt).itemsize / 1e9
    lg, ln, lu = spec.levels
    print(f'\n[{spec.name} {label}] assembled {n_items} {ln}s / {len(feats)} '
          f'{lg}s, {total_frames} {lu}-frames, {gb:.1f} GB in memory')

    os.makedirs(osp.dirname(osp.abspath(out_path)) or '.', exist_ok=True)
    if out_path.endswith('.npz'):
        # np.savez pickles the dict into a single object array
        np.savez(out_path, **{array_key: np.array(feats, dtype=object)})
        how = f"np.load(path, allow_pickle=True)['{array_key}'].item()"
    else:
        np.save(out_path, feats, allow_pickle=True)
        how = 'np.load(path, allow_pickle=True).item()'
    print(f'wrote {out_path}')
    print(f'load it with: feats = {how}')
    print(f'              feats[{lg}][{ln}][{lu}]  ->  (T, {D_CONTEXT})')
    return n_items


def merge(args, spec):
    """Assemble both outputs of one dataset from its per-unit cache:

      - spec.out        : phi_hat, the integrator's output
      - spec.motion_out : motion_context, the motion encoder's output BEFORE
                          the integrator fuses in the image feature
    """
    results = {}
    for kind, path in (('phi', spec.out), ('motion', spec.motion_out)):
        feats, incomplete, mismatched, total = _assemble_kind(args, spec, kind)
        if incomplete:
            print(f'[warn] {len(incomplete)} entries SKIPPED in {path} '
                  f'(missing units), first few: {incomplete[:5]}')
        if mismatched:
            print(f'[warn] {len(mismatched)} units in {path} disagree with '
                  'the GT length. Your features will not line up frame-for-frame:')
            for m in mismatched[:10]:
                print('   ', m)
        results[kind] = (feats, incomplete, mismatched, total)

    f, inc, _, tot = results['phi']
    n_items = _write_feats(args, spec, f, tot, spec.out, 'positions_wham', 'phi')
    f, _, _, tot = results['motion']
    _write_feats(args, spec, f, tot, spec.motion_out, 'motion_feats', 'motion')

    clean = all(not r[1] and not r[2] for r in results.values())
    if args.rm_cache and clean:
        shutil.rmtree(spec.cache_dir, ignore_errors=True)
        print(f'removed {spec.cache_dir}')
    elif args.rm_cache:
        print(f'[warn] keeping {spec.cache_dir}: there were warnings above')

    return n_items, inc


# =========================================================================== #
#  Main
# =========================================================================== #

def _default_cache(out):
    base = out[:-4] if out.endswith(('.npy', '.npz')) else out
    return base + '_cache'


def main():
    ap = argparse.ArgumentParser()

    h = ap.add_argument_group('Human3.6M')
    h.add_argument('--h36m-root', default=None)
    h.add_argument('--out', default=None,
                   help='H36M output FILE, e.g. feats.npy (or .npz). '
                        'feats[subject][action][cam] -> (T, D_CONTEXT). This '
                        'is phi_hat, the integrator output.')
    h.add_argument('--motion-out', default=None,
                   help='second output FILE holding the motion encoder '
                        'features from BEFORE the integrator fuses in the '
                        'image feature (same [subject][action][cam] -> '
                        '(T, D_CONTEXT) layout). Default: motion_feats.npy '
                        '(or .npz to match --out) next to --out.')
    h.add_argument('--cache-dir', default=None,
                   help='per-camera scratch cache (default: <out>_cache). '
                        'Written as the run goes so a crash is resumable; '
                        'this is NOT the deliverable.')
    h.add_argument('--dataset-2d-path', default=None,
                   help='data_2d_h36m_gt.npz — supplies the subject/action '
                        'keys AND the per-camera frame counts')
    h.add_argument('--fps', type=float, default=50.0, help='H36M source fps')

    p = ap.add_argument_group('3DPW')
    p.add_argument('--pw3d-root', default=None,
                   help='3DPW root with sequenceFiles/ and imageFiles/')
    p.add_argument('--pw3d-out', default=None,
                   help='3DPW output FILE. feats[split][seq][actor] -> '
                        '(T, D_CONTEXT), actor order as in the pkl.')
    p.add_argument('--pw3d-motion-out', default=None,
                   help='default: <pw3d-out>_motion.npy (or .npz)')
    p.add_argument('--pw3d-cache-dir', default=None,
                   help='default: <pw3d-out>_cache')
    p.add_argument('--pw3d-splits', default='test',
                   help="comma-separated subset of train,validation,test")
    p.add_argument('--pw3d-fps', type=float, default=30.0)
    p.add_argument('--pw3d-box', choices=['det', 'gt'], default='det',
                   help='det: YOLO box matched to the GT actor by IoU (only '
                        'identity comes from GT). gt: the GT-2D-derived box '
                        'is the ViTPose person box (vitpose detector only).')

    ap.add_argument('--detector', choices=['vitpose', 'yolo'], default='vitpose')
    ap.add_argument('--wham-root', default=None,
                    help='WHAM checkout, needed for --detector vitpose')
    ap.add_argument('--yolo-pose-ckpt', default='yolov8x-pose.pt')
    ap.add_argument('--wham-ckpt', default=WHAM_CKPT)
    ap.add_argument('--hmr2-ckpt', default=HMR2A_CKPT)
    ap.add_argument('--frame-stride', type=int, default=1,
                    help='decimation applied HERE. If your loader also applies '
                         'opt.downsample you will downsample twice — pick one '
                         'side and leave this at 1 on the other.')
    ap.add_argument('--init', choices=['smpl', 'zeros'], default='smpl')
    ap.add_argument('--cache-img-feats', action='store_true',
                    help='also keep the raw ViT features (large)')
    ap.add_argument('--dtype', choices=['float32', 'float16'], default='float32')
    ap.add_argument('--det-batch', type=int, default=16)
    ap.add_argument('--vit-batch', type=int, default=32)
    ap.add_argument('--gpus', default='0',
                    help="comma-separated GPU ids, e.g. '0,1,2,3'. One worker "
                         "process per id, videos split round-robin.")
    ap.add_argument('--cpu', action='store_true', help='ignore --gpus, run on CPU')
    ap.add_argument('--limit', type=int, default=None,
                    help='per dataset, stop after N H36M actions (4 videos '
                         'each) / N 3DPW sequences (all actors). --limit 1 is '
                         'a complete smoke test.')
    ap.add_argument('--overwrite', action='store_true',
                    help='ignore the cache and redo everything')
    ap.add_argument('--dry-run', action='store_true',
                    help='resolve videos and report what is missing, no GPU work')
    ap.add_argument('--merge-only', action='store_true')
    ap.add_argument('--rm-cache', action='store_true',
                    help='delete the scratch cache after a clean merge')
    args = ap.parse_args()

    if not args.h36m_root and not args.pw3d_root:
        ap.error('give --h36m-root and/or --pw3d-root')

    specs = []
    if args.h36m_root:
        if not args.out or not args.dataset_2d_path:
            ap.error('--h36m-root needs --out and --dataset-2d-path')
        if args.cache_dir is None:
            args.cache_dir = _default_cache(args.out)
        if args.motion_out is None:
            ext = '.npz' if args.out.endswith('.npz') else '.npy'
            args.motion_out = osp.join(osp.dirname(osp.abspath(args.out)),
                                       f'motion_feats{ext}')
        keys = np.load(args.dataset_2d_path, allow_pickle=True)['positions_2d'].item()
        layout = build_h36m_layout(args.h36m_root, keys)
        if args.limit:                   # keep whole actions, all 4 cameras
            layout = limit_layout(layout, args.limit)
        specs.append(Spec('h36m', layout, args.out, args.motion_out,
                          args.cache_dir, args.fps, ('subject', 'action', 'cam')))

    if args.pw3d_root:
        if not args.pw3d_out:
            ap.error('--pw3d-root needs --pw3d-out')
        if args.pw3d_box == 'gt' and args.detector != 'vitpose':
            ap.error('--pw3d-box gt only applies to --detector vitpose')
        splits = [s.strip() for s in args.pw3d_splits.split(',') if s.strip()]
        bad = [s for s in splits if s not in PW3D_SPLITS]
        if bad:
            ap.error(f'unknown 3DPW split(s) {bad}; choose from {PW3D_SPLITS}')
        if args.pw3d_cache_dir is None:
            args.pw3d_cache_dir = _default_cache(args.pw3d_out)
        if args.pw3d_motion_out is None:
            base, ext = osp.splitext(args.pw3d_out)
            args.pw3d_motion_out = base + '_motion' + (ext if ext in ('.npy', '.npz') else '.npy')
        layout = build_pw3d_layout(args.pw3d_root, splits)
        if args.limit:
            layout = limit_layout(layout, args.limit)
        specs.append(Spec('3dpw', layout, args.pw3d_out, args.pw3d_motion_out,
                          args.pw3d_cache_dir, args.pw3d_fps,
                          ('split', 'seq', 'actor')))

    for s in specs:
        for path in (s.out, s.motion_out):
            os.makedirs(osp.dirname(osp.abspath(path)) or '.', exist_ok=True)
        os.makedirs(s.cache_dir, exist_ok=True)

    if args.merge_only:
        for s in specs:
            merge(args, s)
        return

    tasks = []
    for s in specs:
        t, missing, done = build_tasks(args, s)
        print(f'[{s.name}] {len(t)} videos to do, {done} already cached, '
              f'{len(missing)} not found on disk')
        if missing:
            print('missing (first 10):')
            for m in missing[:10]:
                print('   ', m)
            print('   -> these will be reported as incomplete at merge')
        tasks += t

    if args.dry_run or not tasks:
        if not args.dry_run:
            for s in specs:
                merge(args, s)
        return

    if args.detector == 'vitpose' and not args.wham_root:
        ap.error('--detector vitpose needs --wham-root')

    # stale failure logs from an earlier run would otherwise be reported again
    for s in specs:
        for f in os.listdir(s.cache_dir):
            if f.startswith('errors.') and f.endswith('.json'):
                os.remove(osp.join(s.cache_dir, f))

    devices = ['cpu'] if args.cpu else [f'cuda:{g.strip()}'
                                        for g in args.gpus.split(',') if g.strip()]
    shards = [tasks[i::len(devices)] for i in range(len(devices))]

    if len(devices) == 1:
        run_worker(0, devices[0], shards[0], args)
    else:
        ctx = mp.get_context('spawn')
        procs = []
        for r, (dev, shard) in enumerate(zip(devices, shards)):
            p = ctx.Process(target=run_worker, args=(r, dev, shard, args))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
        bad = [p.exitcode for p in procs if p.exitcode != 0]
        if bad:
            print(f'[warn] {len(bad)} worker(s) exited non-zero: {bad}')

    for s in specs:
        errs = [f for f in os.listdir(s.cache_dir) if f.startswith('errors.')]
        if errs:
            print(f'[warn] [{s.name}] per-video failures logged in {s.cache_dir}/'
                  f'{{{",".join(sorted(errs))}}} — rerun the same command to '
                  'retry just those (finished videos are skipped)')

    for s in specs:
        merge(args, s)


if __name__ == '__main__':
    main()