#!/usr/bin/env python
"""
Feature extractors for the H36M feature-cache pipeline.

Available extractors:
    resnet_v2       TF-slim resnet_v2_50 / HMR  -> 2048-d, 224x224, [-1,1]
    hmr2_vit_mean   HMR2.0 ViT-H patch mean     -> 1280-d, 256x256, ImageNet
    hmr2_crossattn  HMR2.0 decoder token        -> 1024-d, 256x256, ImageNet
    wham            WHAM phi_hat (integrator)   -> D_CONTEXT, sequence-level:
                    own detector + WHAM crops, see WHAMExtractor
"""

import os

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Cropping
# ─────────────────────────────────────────────────────────────────────────────
def compute_crop_size_from_centered(joints2d_np, scale=1.6):
    """Center each frame's joints around root (joint 0), then compute a single
    bounding box from all centered joints across the whole sequence.
    Returns (cx_offset, cy_offset, side): offset from root to crop center (px)
    and the crop side length. Fixed for the whole clip."""
    root = joints2d_np[:, 0:1, :]            # (N, 1, 2)
    centered = joints2d_np - root            # (N, J, 2) — translation removed
    pts = centered.reshape(-1, 2)
    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
    cx_offset = 0.5 * (x_min + x_max)
    cy_offset = 0.5 * (y_min + y_max)
    side = scale * max(float(x_max - x_min), float(y_max - y_min), 1.0)
    return cx_offset, cy_offset, side


def crop_one_frame(frame_rgb, root_xy, cx_offset, cy_offset, side, out_size,
                   pad_value=None):
    H, W, _ = frame_rgb.shape
    half = side / 2.0
    cx = float(root_xy[0]) + cx_offset
    cy = float(root_xy[1]) + cy_offset

    side_i = max(1, int(round(side)))
    left = int(round(cx - half))
    top = int(round(cy - half))

    src_l, src_t = max(left, 0), max(top, 0)
    src_r, src_b = min(left + side_i, W), min(top + side_i, H)

    fill = (np.asarray(frame_rgb.reshape(-1, 3).mean(0), np.float32)
            if pad_value is None else np.asarray(pad_value, np.float32))
    canvas = np.empty((side_i, side_i, 3), np.uint8)
    canvas[:] = fill.round().astype(np.uint8)

    if src_r > src_l and src_b > src_t:
        canvas[src_t - top:src_b - top, src_l - left:src_r - left, :] = \
            frame_rgb[src_t:src_b, src_l:src_r, :]

    interp = cv2.INTER_AREA if side_i > out_size else cv2.INTER_LINEAR
    return cv2.resize(canvas, (out_size, out_size), interpolation=interp)


# ─────────────────────────────────────────────────────────────────────────────
# Base extractor
# ─────────────────────────────────────────────────────────────────────────────
class FeatureExtractor:
    out_size = None
    feat_dim = None
    default_batch_size = 128

    def preprocess(self, crops_uint8_nhwc):
        """(B, H, W, 3) uint8 RGB -> model input."""
        raise NotImplementedError

    def forward(self, model_input):
        """model input -> (B, feat_dim) np.float32."""
        raise NotImplementedError

    def close(self):
        pass

    def _run(self, crops_list):
        arr = np.stack(crops_list)                 # (B, H, W, 3) uint8 RGB
        return self.forward(self.preprocess(arr))  # (B, feat_dim)

    def extract_clip(self, video_path, j2d, scale, batch_size):
        N = j2d.shape[0]
        cx_off, cy_off, side = compute_crop_size_from_centered(j2d, scale)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        chunks, buf, i = [], [], 0
        while i < N:
            ret, frame = cap.read()
            if not ret:
                break  # video shorter than keypoints
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            crop = crop_one_frame(frame, j2d[i, 0, :],
                                  cx_off, cy_off, side, self.out_size)
            buf.append(crop)
            i += 1
            if len(buf) == batch_size:
                chunks.append(self._run(buf))
                buf = []
        cap.release()

        if buf:
            chunks.append(self._run(buf))
        if not chunks:
            return np.empty((0, self.feat_dim), np.float32), 0
        return np.concatenate(chunks, axis=0).astype(np.float32), i


# ─────────────────────────────────────────────────────────────────────────────
# ResNet v2 / HMR (TensorFlow)
# ─────────────────────────────────────────────────────────────────────────────
class ResNetV2Extractor(FeatureExtractor):
    """TF-slim resnet_v2_50 -> 2048-d global-pool features.

    Normalization: inception preprocessing (scale to [-1, 1]). This is what the
    resnet_v2 checkpoints expect. Do NOT add ImageNet mean/std here — that is
    resnet_v1 / VGG preprocessing and would corrupt v2 features.
    """

    feat_dim = 2048
    out_size = 224
    default_batch_size = 128

    def __init__(self, ckpt, out_size=224, weight_decay=0.0):
        self.out_size = out_size
        import tensorflow as tf
        self.tf = tf
        tf.compat.v1.disable_eager_execution()
        slim = tf.contrib.slim
        from tensorflow.contrib.slim.python.slim.nets import resnet_v2

        self.x_ph = tf.compat.v1.placeholder(
            tf.float32, shape=[None, out_size, out_size, 3], name="input_nhwc")
        with slim.arg_scope(resnet_v2.resnet_arg_scope(weight_decay=weight_decay)):
            net, _ = resnet_v2.resnet_v2_50(
                self.x_ph, num_classes=None, is_training=False,
                reuse=False, scope="resnet_v2_50")
            self.feat_op = tf.squeeze(net, axis=[1, 2], name="feat_2048")

        resnet_vars = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope="resnet_v2_50")
        saver = tf.compat.v1.train.Saver(var_list=resnet_vars)

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=config)
        self.sess.run(tf.compat.v1.global_variables_initializer())
        saver.restore(self.sess, ckpt)
        print(f"Restored resnet_v2_50 vars: {len(resnet_vars)} from {ckpt}")

    def preprocess(self, crops_uint8_nhwc):
        x = crops_uint8_nhwc.astype(np.float32) / 255.0
        return x * 2.0 - 1.0

    def forward(self, model_input):
        return self.sess.run(self.feat_op, feed_dict={self.x_ph: model_input})

    def close(self):
        self.sess.close()


# ─────────────────────────────────────────────────────────────────────────────
# HMR2.0 ViT-H (PyTorch)
# ─────────────────────────────────────────────────────────────────────────────
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


class _HMR2Base(FeatureExtractor):
    """Shared plumbing for the two HMR2 taps.
    """

    out_size = 256
    default_batch_size = 32
    tap = None

    def __init__(self, ckpt, device=None, fp16=True, ief_iters=1, cfg=None):
        import torch
        from models.hmr2 import load_hmr2
        self.torch = torch

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        # autocast rather than model.half(): it keeps LayerNorm and the attention
        # softmax in fp32, which matters over 32 blocks. Costs ~1.2GB more VRAM
        # than half weights. Pass fp16=False for a pure fp32 run.
        self.amp = bool(fp16) and self.device.type == 'cuda'

        needs_head = (self.tap == 'crossattn')
        self.model = load_hmr2(ckpt, with_head=needs_head, ief_iters=ief_iters,
                               device=self.device, cfg_path=cfg)
        print(f"HMR2 ready: tap={self.tap} dim={self.feat_dim} "
              f"device={self.device} autocast_fp16={self.amp}")

    def preprocess(self, crops_uint8_nhwc):
        x = crops_uint8_nhwc.astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        t = self.torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2)))
        return t.to(self.device, non_blocking=True)

    def forward(self, model_input):
        torch = self.torch
        with torch.no_grad():
            if self.amp:
                with torch.autocast('cuda', dtype=torch.float16):
                    out = self.model(model_input, tap=self.tap)
            else:
                out = self.model(model_input, tap=self.tap)
        return out.float().cpu().numpy()

    def close(self):
        del self.model
        if self.device.type == 'cuda':
            self.torch.cuda.empty_cache()


class HMR2ViTMeanExtractor(_HMR2Base):
    """HMR2.0 ViT-H backbone, mean-pooled over the 16x12 = 192 patch tokens.

    Only the backbone is needed, so this also works with a backbone-only
    checkpoint (e.g. vitpose_backbone.pth).
    """
    feat_dim = 1280
    tap = "mean"


class HMR2CrossAttnExtractor(_HMR2Base):
    """HMR2.0 cross-attention decoder token.

    This is the vector the SMPL heads read from: a single learned query token
    that cross-attends over the 192 backbone tokens through 6 decoder layers.
    Compared to the mean-pool tap it is pose-specific rather than a generic
    appearance summary, since it was trained to be linearly decodable into
    pose, shape and camera. Requires the full HMR2 checkpoint (backbone + head).
    """
    feat_dim = 1024
    tap = "crossattn"


# ─────────────────────────────────────────────────────────────────────────────
# WHAM (PyTorch) — verbatim port of the H36M WHAM precompute pipeline
#
# Unlike the extractors above this one is SEQUENCE-level: it does not take the
# caller's crops. Exactly as in the original script it
#   1. runs its own 2D detector (ViTPose-H on YOLOv8x boxes, or yolov8x-pose)
#      on the FULL frames,
#   2. builds WHAM's boxes from those keypoints (+ median filter over time),
#   3. crops with WHAM's zero-filled crop at 256x256, HMR2 crossattn token,
#   4. runs WHAMFeatureNet's motion encoder + integrator over the sequence.
# Output per frame: phi (integrator, D_CONTEXT), plus 'motion' (motion
# encoder before the integrator, D_CONTEXT) and optionally 'img' (ViT token).
#
# torch / scipy / models.* are imported lazily so the TF extractor does not
# need them.
# ─────────────────────────────────────────────────────────────────────────────
WHAM_SMPL_DIR = '/home/ferreiral1/WHAM/dataset/body_models/smpl/'


def build_body_model(device, batch_size=1, gender='neutral',
                     model_path=WHAM_SMPL_DIR, **kwargs):
    from models.smpl import SMPL
    body_model = SMPL(
        model_path=model_path,
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
    import torch
    scale = res.max(-1)[0].reshape(-1)
    mean = torch.stack([res[..., 0] / scale, res[..., 1] / scale], dim=-1)
    return (2 * x / scale.reshape(*[1 for _ in x.shape[1:]])
            - mean.reshape(*[1 for _ in x.shape[1:-1]], -1))


def bbox_normalization(kp_2d, bbox, res, patch_width=224, patch_height=224):
    """normalizer.Normalizer.bbox_normalization -> (N, 37).

    kp_2d (N, 17, 2) pixels, bbox (N, 3) as [cx, cy, scale] with scale in
    units of 200 px, res (2,) = [width, height].
    """
    import torch
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
    import scipy.signal as signal
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


# ── WHAM detectors ──────────────────────────────────────────────────────────
WHAM_MATCH_IOU = 0.3   # min IoU between a detection and the caller's hint box


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
                if ious[j] < WHAM_MATCH_IOU:
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

        cfg = os.path.join(wham_root, 'third-party/ViTPose/configs/body/'
                           '2d_kpt_sview_rgb_img/topdown_heatmap/coco/'
                           'ViTPose_huge_coco_256x192.py')
        ckpt = os.path.join(wham_root, 'checkpoints', 'vitpose-h-multi-coco.pth')
        for p in (cfg, ckpt):
            if not os.path.exists(p):
                raise FileNotFoundError(p)

        self.pose_model = init_pose_model(cfg, ckpt, device=str(device).lower())
        self.bbox_model = YOLO(os.path.join(wham_root, 'checkpoints', 'yolov8x.pt'))
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
                    if ious[j] < WHAM_MATCH_IOU:
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


# ── WHAM per-sequence pipeline ──────────────────────────────────────────────
def _hint_at(hints, idx):
    if hints is None or idx >= len(hints):
        return None
    h = hints[idx]
    return None if np.isnan(h).any() else h


def wham_detect_frames(frames_bgr, detector, frame_stride=1, batch=16,
                       max_frames=None, hints=None):
    """-> kp (N, 17, 3) COCO pixels+conf, res (w, h), n_read.

    frames_bgr: iterable of BGR frames. hints: optional (N_src, 4) xyxy,
    indexed by SOURCE frame (before stride), NaN where unknown.
    """
    kps, buf, hbuf, idx, last, res = [], [], [], 0, None, None
    for frame in frames_bgr:
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


def wham_encode_frames(frames_bgr, bbox, image_encoder, device, d_feat,
                       frame_stride=1, batch=32, n_expected=None):
    """WHAM's crop -> HMR2 crossattn token. -> (N, 1024), first_crop."""
    import torch
    feats, buf, idx, kept = [], [], 0, 0
    first_crop = None

    def flush():
        if not buf:
            return
        t = torch.from_numpy(np.stack(buf)).to(device)
        feats.append(image_encoder(t, tap='crossattn').float().cpu())

    for frame in frames_bgr:
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
           np.zeros((0, d_feat), np.float32))
    return out, first_crop


def build_init_kp(norm_kp2d0, init_kp3d):
    """dataset_custom: cat(init_kp3d[51], norm_kp2d[0][37]) -> (1, 88)."""
    import torch
    out = torch.cat([init_kp3d.reshape(1, -1),
                     norm_kp2d0.reshape(1, -1)], dim=-1)      # (1, 88)
    return out.unsqueeze(0)                                    # (1, 1, 88)


def rot6d_to_rotmat_hmr2(x):
    """hmr2/utils/geometry.py rot6d_to_rotmat."""
    import torch
    import torch.nn.functional as F
    x = x.reshape(-1, 2, 3).permute(0, 2, 1).contiguous()
    a1, a2 = x[:, :, 0], x[:, :, 1]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def smpl_init_kp3d(image_encoder, first_crop, smpl, device):
    """extractor.predict_init + SMPL forward -> (1, 51). None se indisponível."""
    import torch
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


def fit_length(arr, T, tag=''):
    """Trim, or pad by repeating the last frame. Padding means something
    upstream is wrong (fewer frames decoded than annotated)."""
    n = len(arr)
    if n == T:
        return arr
    if n > T:
        return arr[:T]
    pad = np.repeat(arr[-1:], T - n, axis=0)
    print(f'  [warn] {tag}: only {n} frames for a target of {T}; padding '
          f'{T - n} by repeat.')
    return np.concatenate([arr, pad], 0)


class WHAMExtractor(FeatureExtractor):
    """WHAM phi_hat (integrator output), D_CONTEXT per frame.

    Sequence-level: call extract_frames(), not _run(). Also returns 'motion'
    (motion encoder output before the integrator) and, with cache_img_feats,
    'img' (the HMR2 crossattn token WHAM fed in).
    """
    sequence_level = True
    out_size = CROP_SIZE
    default_batch_size = 32           # ViT batch
    feat_dim = None                   # D_CONTEXT, known after import

    def __init__(self, hmr2_ckpt=None, wham_ckpt=None, detector='vitpose',
                 wham_root=None, yolo_pose_ckpt='yolov8x-pose.pt', device=None,
                 init='smpl', smpl_dir=WHAM_SMPL_DIR, hint_as_box=False,
                 det_batch=16, vit_batch=32, cache_img_feats=False):
        import torch
        from models.wham import (build_wham_feature_net, HMR2A_CKPT, WHAM_CKPT,
                                 KEYPOINTS_THR, D_CONTEXT, D_FEAT_VIT)
        self.torch = torch
        self.KEYPOINTS_THR, self.D_FEAT_VIT = KEYPOINTS_THR, D_FEAT_VIT
        self.feat_dim = D_CONTEXT
        self.extra_kinds = ('motion', 'img') if cache_img_feats else ('motion',)

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = str(device)
        if device == 'cuda':
            device = 'cuda:0'          # set_device needs an index
        if device.startswith('cuda'):
            torch.cuda.set_device(device)
        self.device = device
        self.det_batch, self.vit_batch = det_batch, vit_batch

        if detector == 'vitpose':
            if not wham_root:
                raise ValueError('WHAM with detector=vitpose needs wham_root')
            self.detector = ViTPoseDetector(wham_root, device=device,
                                            hint_as_box=hint_as_box)
        else:
            self.detector = YoloPoseDetector(yolo_pose_ckpt, device=device)

        self.net = build_wham_feature_net(wham_ckpt=wham_ckpt or WHAM_CKPT,
                                          hmr2_ckpt=hmr2_ckpt or HMR2A_CKPT,
                                          device=device)
        self.net.eval()

        self.smpl = None
        if init == 'smpl':
            try:
                self.smpl = build_body_model(device, model_path=smpl_dir)
            except Exception as e:
                print(f'[warn] SMPL unavailable ({e}); zero init_kp3d, so '
                      'the first frames of each clip will be off.', flush=True)
        print(f"WHAM ready: detector={detector} dim={self.feat_dim} "
              f"device={device} init={'smpl' if self.smpl else 'zeros'}")

    def _run(self, crops_list):
        raise RuntimeError('WHAMExtractor is sequence-level: use extract_frames()')

    def extract_frames(self, frames_fn, fps, hints=None, target_len=None,
                       frame_stride=1, tag=''):
        """One continuous clip of one person ->
        {'phi': (T, D_CONTEXT), 'motion': (T, D_CONTEXT), 'img': (T, 1024)}.

        frames_fn: zero-arg callable returning a fresh iterator of BGR frames
        (called twice: detection pass, then crop pass, as in the original).
        hints: optional (T, 4) xyxy per frame to pick the right person.
        """
        torch, net, device = self.torch, self.net, self.device
        with torch.no_grad():
            kp, res_wh, _ = wham_detect_frames(frames_fn(), self.detector,
                                               frame_stride=frame_stride,
                                               batch=self.det_batch, hints=hints)
            n = len(kp)
            if n == 0:
                raise RuntimeError('decoded zero frames')

            eff_fps = fps / frame_stride
            bbox = compute_bboxes_from_keypoints(kp, eff_fps)
            res = torch.tensor(res_wh).float()

            norm_kp2d = bbox_normalization(kp[..., :2].copy(), bbox, res,
                                           KP_PATCH, KP_PATCH)          # (N, 37)
            mask = torch.from_numpy(kp[..., -1] < self.KEYPOINTS_THR)   # (N, 17)

            img_feat, first_crop = wham_encode_frames(
                frames_fn(), bbox, net.image_encoder, device, self.D_FEAT_VIT,
                frame_stride=frame_stride, batch=self.vit_batch, n_expected=n)

            n = min(n, len(img_feat))
            norm_kp2d, mask, img_feat = norm_kp2d[:n], mask[:n], img_feat[:n]

            init_kp3d = None
            if self.smpl is not None and first_crop is not None:
                init_kp3d = smpl_init_kp3d(net.image_encoder, first_crop,
                                           self.smpl, device)
            if init_kp3d is None:
                init_kp3d = torch.zeros(1, 17 * 3)
            init_kp = build_init_kp(norm_kp2d[0], init_kp3d.cpu())

            phi, _pred_kp3d, motion_feat = net(
                norm_kp2d.unsqueeze(0).to(device),
                init_kp.to(device),
                features=torch.from_numpy(img_feat).unsqueeze(0).to(device),
                mask=mask.unsqueeze(0).to(device),
                return_extras=True)
            phi = phi[0].float().cpu().numpy()
            motion_feat = motion_feat[0].float().cpu().numpy()

        if target_len is not None:
            phi = fit_length(phi, target_len, tag)
            img_feat = fit_length(img_feat, target_len, tag)
            motion_feat = fit_length(motion_feat, target_len, tag)
        out = {'phi': phi, 'motion': motion_feat}
        if 'img' in self.extra_kinds:
            out['img'] = img_feat
        return out

    def close(self):
        close = getattr(self.detector, 'close', None)
        if close:
            close()
        del self.net
        if self.device.startswith('cuda'):
            self.torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────
EXTRACTORS = {
    "resnet_v2": ResNetV2Extractor,
    "hmr2_vit_mean": HMR2ViTMeanExtractor,
    "hmr2_crossattn": HMR2CrossAttnExtractor,
    "wham": WHAMExtractor,
}


def build_extractor(name, args):
    """Factory: map --extractor name to an extractor instance."""
    if name == "resnet_v2":
        return ResNetV2Extractor(
            ckpt=args.ckpt,
            out_size=getattr(args, "out_size", None) or 224)
    if name in ("hmr2_vit_mean", "hmr2_crossattn"):
        cls = EXTRACTORS[name]
        ex = cls(ckpt=args.ckpt,
                 device=getattr(args, "device", None),
                 fp16=not getattr(args, "fp32", False),
                 ief_iters=getattr(args, "ief_iters", 1),
                 cfg=getattr(args, "hmr2_cfg", None))
        override = getattr(args, "out_size", None)
        if override and override != ex.out_size:
            raise ValueError(
                f"{name} requires out_size=256 (the ViT expects 256x192 after "
                f"the centre slice); got {override}")
        return ex
    if name == "wham":
        override = getattr(args, "out_size", None)
        if override and override != CROP_SIZE:
            raise ValueError(f"wham crops at {CROP_SIZE}x{CROP_SIZE}; got {override}")
        return WHAMExtractor(
            hmr2_ckpt=getattr(args, "ckpt", None),
            wham_ckpt=getattr(args, "wham_ckpt", None),
            detector=getattr(args, "wham_detector", "vitpose"),
            wham_root=getattr(args, "wham_root", None),
            yolo_pose_ckpt=getattr(args, "yolo_pose_ckpt", "yolov8x-pose.pt"),
            device=getattr(args, "device", None),
            init=getattr(args, "wham_init", "smpl"),
            smpl_dir=getattr(args, "smpl_dir", None) or WHAM_SMPL_DIR,
            hint_as_box=getattr(args, "wham_box", "det") == "gt",
            det_batch=getattr(args, "det_batch", 16),
            vit_batch=getattr(args, "batch_size", None) or 32,
            cache_img_feats=getattr(args, "wham_cache_img_feats", False))
    raise ValueError(f"Unknown extractor: {name}. Choices: {sorted(EXTRACTORS)}")