

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
from torch import nn
import sys
import os

from models.hmr2 import load_hmr2

# Checkpoints from pretrained WHAM ViT-H model. The HMR2 checkpoint is used to build the image encoder.
HMR2A_CKPT = '/media/data/ferreiral1/models/WHAM/checkpoints/hmr2a.ckpt'
WHAM_CKPT = '/media/data/ferreiral1/models/WHAM/checkpoints/wham_vit_bedlam_w_3dpw.pth.tar'



N_JOINTS = 17          # _C.KEYPOINTS.NUM_JOINTS
D_EMBED = 512          # model_base.yaml
N_LAYERS = 3           # model_base.yaml
POSE_DR = 0.15         # model_base.yaml
RNN_TYPE = 'LSTM'      # Network default
D_FEAT_VIT = 1024      # _C.IMG_FEAT_DIM['vit'] — the cross-attention token
D_FEAT_RESNET = 2048   # _C.IMG_FEAT_DIM['resnet']

IN_DIM = N_JOINTS * 2 + 3          # 37
D_CONTEXT = D_EMBED + N_JOINTS * 3  # 563

KEYPOINTS_THR = 0.3    # CustomDataset: kp confidence below this is masked

CROP_H = CROP_W = 256  # WHAM's FeatureExtractor.run(patch_h=256, patch_w=256)
KP_PATCH = 224         # Normalizer patch size — deliberately NOT 256


#  VERBATIM WHAM MODULES — lib/models/layers/modules.py

# https://github.com/yohanshin/WHAM/blob/2b54f7797391c94876848b905ed875b154c4a295/lib/models/layers/modules.py#L13
class Regressor(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dims, init_dim, layer='LSTM', n_layers=2, n_iters=1):
        super().__init__()
        self.n_outs = len(out_dims)

        self.rnn = getattr(nn, layer.upper())(
            in_dim + init_dim, hid_dim, n_layers,
            bidirectional=False, batch_first=True, dropout=0.3)

        for i, out_dim in enumerate(out_dims):
            setattr(self, 'declayer%d' % i, nn.Linear(hid_dim, out_dim))
            nn.init.xavier_uniform_(getattr(self, 'declayer%d' % i).weight, gain=0.01)

    def forward(self, x, inits, h0):
        xc = torch.cat([x, *inits], dim=-1)
        xc, h0 = self.rnn(xc, h0)

        preds = []
        for j in range(self.n_outs):
            out = getattr(self, 'declayer%d' % j)(xc)
            preds.append(out)

        return preds, xc, h0

# https://github.com/yohanshin/WHAM/blob/2b54f7797391c94876848b905ed875b154c4a295/lib/models/layers/modules.py#L38
class NeuralInitialization(nn.Module):
    def __init__(self, in_dim, hid_dim, layer, n_layers):
        super().__init__()

        out_dim = hid_dim
        self.n_layers = n_layers
        self.num_inits = int(layer.upper() == 'LSTM') + 1
        out_dim *= self.num_inits * n_layers

        self.linear1 = nn.Linear(in_dim, hid_dim)
        self.linear2 = nn.Linear(hid_dim, hid_dim * self.n_layers)
        self.linear3 = nn.Linear(hid_dim * self.n_layers, out_dim)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x):
        b = x.shape[0]

        out = self.linear3(self.relu2(self.linear2(self.relu1(self.linear1(x)))))
        out = out.view(b, self.num_inits, self.n_layers, -1).permute(1, 2, 0, 3).contiguous()

        if self.num_inits == 2:
            return tuple([_ for _ in out])
        return out[0]

# https://github.com/yohanshin/WHAM/blob/2b54f7797391c94876848b905ed875b154c4a295/lib/models/layers/modules.py#L64
class Integrator(nn.Module):
    """Fuses motion context (563) with the image feature (1024) -> phi_hat (563)."""

    def __init__(self, in_channel, out_channel, hid_channel=1024):
        super().__init__()

        self.layer1 = nn.Linear(in_channel, hid_channel)
        self.relu1 = nn.ReLU()
        self.dr1 = nn.Dropout(0.1)

        self.layer2 = nn.Linear(hid_channel, hid_channel)
        self.relu2 = nn.ReLU()
        self.dr2 = nn.Dropout(0.1)

        self.layer3 = nn.Linear(hid_channel, out_channel)

    def forward(self, x, feat):
        res = x
        # Per-sequence flag: a batch element whose features are all-zero
        # (i.e. no image feature supplied) skips the residual.
        mask = (feat != 0).all(dim=-1).all(dim=-1)

        out = torch.cat((x, feat), dim=-1)
        out = self.layer1(out)
        out = self.relu1(out)
        out = self.dr1(out)

        out = self.layer2(out)
        out = self.relu2(out)
        out = self.dr2(out)

        out = self.layer3(out)
        out[mask] = out[mask] + res[mask]

        return out

# https://github.com/yohanshin/WHAM/blob/2b54f7797391c94876848b905ed875b154c4a295/lib/models/layers/modules.py#L98
class MotionEncoder(nn.Module):
    """Recurrent 2D->3D lifter. Returns (pred_kp3d, motion_context[512+51])."""

    def __init__(self, in_dim, d_embed, pose_dr, rnn_type, n_layers, n_joints):
        super().__init__()

        self.n_joints = n_joints

        self.embed_layer = nn.Linear(in_dim, d_embed)
        self.pos_drop = nn.Dropout(pose_dr)

        # Keypoints initializer
        self.neural_init = NeuralInitialization(n_joints * 3 + in_dim, d_embed, rnn_type, n_layers)

        # 3d keypoints regressor
        self.regressor = Regressor(
            d_embed, d_embed, [n_joints * 3], n_joints * 3, rnn_type, n_layers)

    def forward(self, x, init):
        """ Forward pass of motion encoder.
        """

        self.b, self.f = x.shape[:2]
        x = self.embed_layer(x.reshape(self.b, self.f, -1))
        x = self.pos_drop(x)

        h0 = self.neural_init(init)
        pred_list = [init[..., :self.n_joints * 3]]
        motion_context_list = []

        for i in range(self.f):
            (pred_kp3d, ), motion_context, h0 = self.regressor(x[:, [i]], pred_list[-1:], h0)
            motion_context_list.append(motion_context)
            pred_list.append(pred_kp3d)

        pred_kp3d = torch.cat(pred_list[1:], dim=1).view(self.b, self.f, -1, 3)
        motion_context = torch.cat(motion_context_list, dim=1)

        # Merge 3D keypoints with motion context
        motion_context = torch.cat((motion_context, pred_kp3d.reshape(self.b, self.f, -1)), dim=-1)
        return pred_kp3d, motion_context



class WHAMFeatureNet(nn.Module):
    """image encoder -> motion encoder -> feature integrator -> phi_hat.
    """

    def __init__(self,
                 hmr2_ckpt=HMR2A_CKPT,
                 n_joints=N_JOINTS,
                 d_embed=D_EMBED,
                 n_layers=N_LAYERS,
                 pose_dr=POSE_DR,
                 rnn_type=RNN_TYPE,
                 d_feat=D_FEAT_VIT,
                 ief_iters=1,
                 hmr2_device='cuda'):
        super().__init__()

        self.n_joints = n_joints
        self.in_dim = n_joints * 2 + 3
        self.d_context = d_embed + n_joints * 3
        self.d_feat = d_feat

        if d_feat != D_FEAT_VIT:
            raise ValueError(
                f'the cross-attention token is {D_FEAT_VIT}-d, but d_feat={d_feat}. '
                'The HMR2 image encoder only makes sense with the vit config.')

        # Learned embedding substituted for low-confidence keypoints
        self.mask_embedding = nn.Parameter(torch.zeros(1, 1, n_joints, 2))

        # Module 1. Motion Encoder
        self.motion_encoder = MotionEncoder(in_dim=self.in_dim,
                                            d_embed=d_embed,
                                            pose_dr=pose_dr,
                                            rnn_type=rnn_type,
                                            n_layers=n_layers,
                                            n_joints=n_joints)

        # Module 2. Image Encoder — HMR2Features with hmr2a.ckpt, tapped at
        # the cross-attention decoder token. load_hmr2 already eval()s it and
        # sets requires_grad=False.
        self.image_encoder = load_hmr2(hmr2_ckpt,
                                       with_head=True,   # crossattn tap needs it
                                       ief_iters=ief_iters,
                                       device=hmr2_device,
                                       strict_report=True)

        # Module 3. Feature Integrator
        self.integrator = Integrator(in_channel=d_feat + self.d_context,
                                     out_channel=self.d_context)

    def train(self, mode=True):
        """Keep the ViT in eval mode always.

        vit_huge() is built with drop_path_rate=0.55. If anything ever calls
        net.train(), stochastic depth switches on and the image features become
        noise — silently, with correct shapes. This is a frozen extractor, so
        the ViT has no business being in train mode.
        """
        super().train(mode)
        self.image_encoder.eval()
        return self

    @property
    def feature_dim(self):
        """Width of phi_hat. 563 for the standard WHAM config."""
        return self.d_context

    # ----------------------------------------------------------------- #
    #  Image encoding
    # ----------------------------------------------------------------- #

    @torch.no_grad()
    def encode_images(self, images, chunk=32):
        """(B, T, 3, 256, 256) ImageNet-normalised crops -> (B, T, 1024). """
        B, T = images.shape[:2]
        flat = images.reshape(B * T, *images.shape[2:])
        device = next(self.image_encoder.parameters()).device

        outs = []
        for i in range(0, flat.shape[0], chunk):
            batch = flat[i:i + chunk].to(device, non_blocking=True).float()
            outs.append(self.image_encoder(batch, tap='crossattn'))
        feats = torch.cat(outs, dim=0)

        if feats.shape[-1] != self.d_feat:
            raise RuntimeError(
                f'image encoder returned {feats.shape[-1]}-d, expected {self.d_feat}. '
                "Check that tap='crossattn' and that the head was built.")

        return feats.reshape(B, T, self.d_feat)


    def preprocess(self, x, mask):
        b, f = x.shape[:2]
        x = x.clone()

        mask_embedding = mask.unsqueeze(-1) * self.mask_embedding
        _mask = mask.unsqueeze(-1).repeat(1, 1, 1, 2).reshape(b, f, -1)
        _mask = torch.cat((_mask, torch.zeros_like(_mask[..., :3])), dim=-1)
        _mask_embedding = mask_embedding.reshape(b, f, -1)
        _mask_embedding = torch.cat((_mask_embedding, torch.zeros_like(_mask_embedding[..., :3])), dim=-1)
        x[_mask] = 0.0
        x = x + _mask_embedding
        return x

    def forward(self, x, init_kp, images=None, features=None, mask=None,
                return_extras=False, chunk=32):
        if (features is None) == (images is None):
            raise ValueError('give exactly one of `images` or `features`.')

        if features is None:
            features = self.encode_images(images, chunk=chunk)

        features = features.to(x.device)

        if mask is None:
            mask = torch.zeros(*x.shape[:2], self.n_joints, dtype=torch.bool, device=x.device)

        if x.shape[-1] != self.in_dim:
            raise ValueError(f'x must be (B, T, {self.in_dim}), got {tuple(x.shape)}. '
                             'Did you forget to append [cx, cy, scale]?')
        if features.shape[-1] != self.d_feat:
            raise ValueError(f'features must be (B, T, {self.d_feat}), got {tuple(features.shape)}.')
        if features.shape[:2] != x.shape[:2]:
            raise ValueError(f'features {tuple(features.shape[:2])} and x {tuple(x.shape[:2])} '
                             'disagree on (B, T) — frame misalignment.')

        x = self.preprocess(x, mask)

        # Stage 1. Encode motion
        pred_kp3d, motion_context = self.motion_encoder(x, init_kp)

        # Stage 3. Integrate features  (stages 2/4/5 are decoders — dropped)
        phi_hat = self.integrator(motion_context, features)

        if return_extras:
            return phi_hat, pred_kp3d, motion_context
        return phi_hat


# =========================================================================== #
#  CHECKPOINT LOADING
# =========================================================================== #

_KEEP_PREFIXES = ('motion_encoder.', 'integrator.')
_KEEP_EXACT = ('mask_embedding',)


def load_wham_feature_weights(model, ckpt_path=WHAM_CKPT, verbose=True):
    """Fill motion_encoder.* / integrator.* / mask_embedding from WHAM's checkpoint.

    image_encoder.* is deliberately out of scope — it was already filled from
    hmr2a.ckpt in __init__. The completeness check below only inspects
    WHAM-owned parameters; a naive `missing` check over the whole state_dict
    would report every ViT-H tensor as absent from the WHAM checkpoint.
    """
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    except TypeError:          # torch < 1.13 has no weights_only
        ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt

    own = model.state_dict()
    wanted = [k for k in own if k.startswith(_KEEP_PREFIXES) or k in _KEEP_EXACT]

    filtered = {k: state[k] for k in wanted if k in state}
    missing = [k for k in wanted if k not in state]
    shape_bad = [k for k, v in filtered.items() if own[k].shape != v.shape]

    if not filtered:
        raise RuntimeError(
            f'No motion_encoder/integrator weights found in {ckpt_path}. '
            f'Top-level keys were: {list(state.keys())[:10]}')
    if shape_bad:
        raise RuntimeError(
            f'{len(shape_bad)} tensor(s) matched by name but not by shape, e.g. '
            f'{[(k, tuple(own[k].shape), tuple(filtered[k].shape)) for k in shape_bad[:3]]}. '
            'Usually a backbone mismatch (vit -> d_feat=1024, resnet -> 2048).')
    if missing:
        raise RuntimeError(
            f'{len(missing)} WHAM parameter(s) got no weights and are still randomly '
            f'initialised: {missing[:8]}{" ..." if len(missing) > 8 else ""}.')

    model.load_state_dict(filtered, strict=False)

    if verbose:
        print(f'[wham_feature_net] loaded {len(filtered)}/{len(wanted)} WHAM tensors '
              f'from {ckpt_path} ({len(state) - len(filtered)} decoder/SMPL tensors dropped)')
    return model


def build_wham_feature_net(wham_ckpt=WHAM_CKPT,
                           hmr2_ckpt=HMR2A_CKPT,
                           device='cuda',
                           ief_iters=1):
    """Construct, load both checkpoints, freeze, move to device.

    The ViT is loaded on CPU first and moved once at the end, so it is not
    resident on the GPU twice during construction.

    If load_hmr2 raises about missing init_body_pose / init_betas / init_cam
    and nothing else, those three buffers are unused on the zero-token path.
    """
    net = WHAMFeatureNet(hmr2_ckpt=hmr2_ckpt, ief_iters=ief_iters, hmr2_device='cpu')
    load_wham_feature_weights(net, wham_ckpt)

    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net.to(device)