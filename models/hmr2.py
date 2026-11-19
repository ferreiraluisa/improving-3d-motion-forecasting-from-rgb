"""
HMR2.0 ViT-H backbone + cross-attention SMPL decoder,
stripped to the parts needed for feature extraction.

Two types of features are available:
    * backbone tokens  -> (B, 1280, 16, 12), mean-pooled to 1280-d
    * decoder token    -> (B, 1024), the output of the cross-attention
                          transformer, i.e. the vector the SMPL heads read from

Weights: the released HMR2.0b checkpoint, inside
    https://www.cs.utexas.edu/~pavlakos/4dhumans/hmr2_data.tar.gz
at
    logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt

Module and parameter names below deliberately mirror 4D-Humans
(hmr2/models/backbones/vit.py, hmr2/models/components/pose_transformer.py,
hmr2/models/heads/smpl_head.py) so the checkpoint loads by key without any
remapping. Do not rename anything here.
"""

import inspect
from functools import partial

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.vision_transformer import Block
from timm.models.layers import trunc_normal_, to_2tuple


# ─── Patch embedding ─────────────────────────────────────────────────────────

class PatchEmbed(nn.Module):
    """Image to Patch Embedding with optional stride ratio.

    NOTE: the conv geometry here is not the obvious one. The official HMR2 /
    ViTPose patch embed uses

        kernel_size = patch_size
        stride      = patch_size // ratio
        padding     = 4 + 2 * (ratio // 2 - 1)

    which for ratio=1 gives kernel 16, stride 16, **padding 2**. A padding-0
    version produces the same 16x12 grid, so the checkpoint still loads and the
    shapes still line up -- but every patch samples a 2px-shifted window, so the
    features are quietly not the ones the model was trained to produce. That is
    the kind of bug that costs you a couple of mm MPJPE and no error message.
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, ratio=1):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = ((img_size[1] // patch_size[1])
                            * (img_size[0] // patch_size[0]) * (ratio ** 2))
        self.patch_shape = (int(img_size[0] // patch_size[0] * ratio),
                            int(img_size[1] // patch_size[1] * ratio))
        self.origin_patch_shape = (int(img_size[0] // patch_size[0]),
                                   int(img_size[1] // patch_size[1]))
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size,
                              stride=(patch_size[0] // ratio),
                              padding=4 + 2 * (ratio // 2 - 1))

    def forward(self, x, **kwargs):
        x = self.proj(x)                            # (B, embed_dim, Hp, Wp)
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)            # (B, N, embed_dim)
        return x, (Hp, Wp)


# ─── ViT backbone ────────────────────────────────────────────────────────────

class ViT(nn.Module):

    def __init__(self,
                 img_size=224, patch_size=16, in_chans=3, num_classes=80, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None,
                 norm_layer=None, use_checkpoint=False, frozen_stages=-1, ratio=1,
                 last_norm=True, patch_padding='pad', freeze_attn=False, freeze_ffn=False):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.frozen_stages = frozen_stages
        self.use_checkpoint = use_checkpoint
        self.patch_padding = patch_padding
        self.freeze_attn = freeze_attn
        self.freeze_ffn = freeze_ffn
        self.depth = depth

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_chans=in_chans, embed_dim=embed_dim, ratio=ratio)
        num_patches = self.patch_embed.num_patches

        # pos_embed keeps a slot for a class token (index 0) that is never used.
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # timm's Block matches HMR2's own Block key-for-key
        # (norm1, attn.qkv, attn.proj, norm2, mlp.fc1, mlp.fc2). LayerScale and
        # q_norm/k_norm are Identity by default, so they add no keys.
        _block_kw = inspect.signature(Block.__init__).parameters
        _drop_kw = 'proj_drop' if 'proj_drop' in _block_kw else 'drop'

        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, attn_drop=attn_drop_rate,
                  drop_path=dpr[i], norm_layer=norm_layer,
                  **{_drop_kw: drop_rate})
            for i in range(depth)])

        self.last_norm = norm_layer(embed_dim) if last_norm else nn.Identity()

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for p in self.patch_embed.parameters():
                p.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            m = self.blocks[i]
            m.eval()
            for p in m.parameters():
                p.requires_grad = False
        if self.freeze_attn:
            for i in range(self.depth):
                m = self.blocks[i]
                m.attn.eval(); m.norm1.eval()
                for p in list(m.attn.parameters()) + list(m.norm1.parameters()):
                    p.requires_grad = False
        if self.freeze_ffn:
            self.pos_embed.requires_grad = False
            self.patch_embed.eval()
            for p in self.patch_embed.parameters():
                p.requires_grad = False
            for i in range(self.depth):
                m = self.blocks[i]
                m.mlp.eval(); m.norm2.eval()
                for p in list(m.mlp.parameters()) + list(m.norm2.parameters()):
                    p.requires_grad = False

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x):
        B = x.shape[0]
        x, (Hp, Wp) = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed[:, 1:] + self.pos_embed[:, :1]
        for blk in self.blocks:
            x = checkpoint.checkpoint(blk, x) if self.use_checkpoint else blk(x)
        x = self.last_norm(x)
        return x.permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous()

    def forward(self, x):
        return self.forward_features(x)

    def train(self, mode=True):
        # nn.Module.eval() is `return self.train(False)`, so an override that
        # returns None makes `vit.eval()` evaluate to None. The upstream HMR2
        # and ViTPose code has this bug; returning self here avoids it.
        super().train(mode)
        self._freeze_stages()
        return self


def vit_huge():
    """ViT-H as configured in HMR2.0."""
    return ViT(img_size=(256, 192), patch_size=16, embed_dim=1280, depth=32,
               num_heads=16, ratio=1, use_checkpoint=False, mlp_ratio=4,
               qkv_bias=True, drop_path_rate=0.55)


# ─── Cross-attention decoder ────────────────────────────────────────────────
# Layout mirrors hmr2/models/components/pose_transformer.py exactly.

def _default(val, d):
    return val if val is not None else d


class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm="layer", norm_cond_dim=-1):
        super().__init__()
        if norm != "layer":
            raise NotImplementedError(f"only norm='layer' is supported, got {norm}")
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


def _heads(t, h):
    b, n, _ = t.shape
    return t.reshape(b, n, h, -1).permute(0, 2, 1, 3)


def _unheads(t):
    b, h, n, d = t.shape
    return t.permute(0, 2, 1, 3).reshape(b, n, h * d)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        q, k, v = [_heads(t, self.heads) for t in self.to_qkv(x).chunk(3, dim=-1)]
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        out = torch.matmul(self.dropout(self.attend(dots)), v)
        return self.to_out(_unheads(out))


class CrossAttention(nn.Module):
    def __init__(self, dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        context_dim = _default(context_dim, dim)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, context=None):
        context = _default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q, k, v = [_heads(t, self.heads) for t in (self.to_q(x), k, v)]
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        out = torch.matmul(self.dropout(self.attend(dots)), v)
        return self.to_out(_unheads(out))


class TransformerCrossAttn(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0,
                 norm="layer", norm_cond_dim=-1, context_dim=None):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            sa = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
            ca = CrossAttention(dim, context_dim=context_dim, heads=heads,
                                dim_head=dim_head, dropout=dropout)
            ff = FeedForward(dim, mlp_dim, dropout=dropout)
            self.layers.append(nn.ModuleList([
                PreNorm(dim, sa, norm=norm, norm_cond_dim=norm_cond_dim),
                PreNorm(dim, ca, norm=norm, norm_cond_dim=norm_cond_dim),
                PreNorm(dim, ff, norm=norm, norm_cond_dim=norm_cond_dim),
            ]))

    def forward(self, x, context=None):
        for self_attn, cross_attn, ff in self.layers:
            x = self_attn(x) + x
            x = cross_attn(x, context=context) + x
            x = ff(x) + x
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, num_tokens, token_dim, dim, depth, heads, mlp_dim,
                 dim_head=64, dropout=0.0, emb_dropout=0.0, emb_dropout_type='drop',
                 norm="layer", norm_cond_dim=-1, context_dim=None,
                 skip_token_embedding=False):
        super().__init__()
        self.to_token_embedding = (nn.Identity() if skip_token_embedding
                                   else nn.Linear(token_dim, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, dim))
        # All three dropout variants are parameter-free; at eval they are no-ops
        # and contribute no state_dict keys either way.
        self.dropout = nn.Identity()
        self.transformer = TransformerCrossAttn(
            dim, depth, heads, dim_head, mlp_dim, dropout, norm=norm,
            norm_cond_dim=norm_cond_dim, context_dim=context_dim)

    def forward(self, inp, context=None):
        x = self.to_token_embedding(inp)
        n = x.shape[1]
        x = self.dropout(x)
        x = x + self.pos_embedding[:, :n]
        return self.transformer(x, context=context)


class SMPLCrossAttnHead(nn.Module):
    """The HMR2 SMPL head, truncated at the decoder token.

    Everything after `token_out` (rot6d -> rotmat, SMPL forward) is dropped;
    decpose/decshape/deccam are only instantiated because they are needed to
    run more than one IEF iteration when the token carries the current estimate.
    """

    def __init__(self, token_dim=1, dim=1024, depth=6, heads=8, mlp_dim=1024,
                 dim_head=64, context_dim=1280, npose=144, ief_iters=1):
        super().__init__()
        self.token_dim = token_dim
        self.input_is_mean_shape = (token_dim != 1)
        self.ief_iters = ief_iters
        self.npose = npose
        self.transformer = TransformerDecoder(
            num_tokens=1, token_dim=token_dim, dim=dim, depth=depth, heads=heads,
            mlp_dim=mlp_dim, dim_head=dim_head, dropout=0.0, emb_dropout=0.0,
            norm="layer", context_dim=context_dim)
        self.decpose = nn.Linear(dim, npose)
        self.decshape = nn.Linear(dim, 10)
        self.deccam = nn.Linear(dim, 3)
        self.register_buffer('init_body_pose', torch.zeros(1, npose))
        self.register_buffer('init_betas', torch.zeros(1, 10))
        self.register_buffer('init_cam', torch.zeros(1, 3))

    def forward(self, x):
        """x: (B, C, Hp, Wp) backbone map -> token_out (B, dim)."""
        B = x.shape[0]
        x = x.flatten(2).transpose(1, 2)          # b c h w -> b (h w) c

        if not self.input_is_mean_shape:
            # Zero token: every IEF iteration is identical, so one pass is exact.
            token = torch.zeros(B, 1, 1, device=x.device, dtype=x.dtype)
            return self.transformer(token, context=x).squeeze(1)

        pose = self.init_body_pose.expand(B, -1)
        betas = self.init_betas.expand(B, -1)
        cam = self.init_cam.expand(B, -1)
        token_out = None
        for _ in range(self.ief_iters):
            token = torch.cat([pose, betas, cam], dim=1)[:, None, :]
            token_out = self.transformer(token, context=x).squeeze(1)
            pose = self.decpose(token_out) + pose
            betas = self.decshape(token_out) + betas
            cam = self.deccam(token_out) + cam
        return token_out


# ─── Assembled model + checkpoint loading ───────────────────────────────────

class HMR2Features(nn.Module):
    """ViT-H backbone, optionally with the cross-attention head attached.

    Input is (B, 3, 256, 256), ImageNet-normalised. The backbone sees the
    centre 192 columns (256x192), exactly as in HMR2's own forward step, so the
    subject stays centred as long as the incoming crop is centred.
    """

    def __init__(self, with_head=True, head_kwargs=None):
        super().__init__()
        self.backbone = vit_huge()
        self.smpl_head = SMPLCrossAttnHead(**(head_kwargs or {})) if with_head else None

    def tokens(self, x):
        if x.shape[-1] == 256 and x.shape[-2] == 256:
            x = x[:, :, :, 32:-32]                # centred 256x192
        elif x.shape[-2:] != (256, 192):
            raise ValueError(f"expected (B,3,256,256) or (B,3,256,192), got {tuple(x.shape)}")
        return self.backbone(x)                   # (B, 1280, 16, 12)

    def forward(self, x, tap="mean"):
        f = self.tokens(x)
        if tap == "mean":
            return f.flatten(2).mean(dim=2)       # (B, 1280)
        if tap == "crossattn":
            if self.smpl_head is None:
                raise RuntimeError("head not built; construct with with_head=True")
            return self.smpl_head(f)              # (B, 1024)
        raise ValueError(f"unknown tap: {tap}")


def read_hmr2_config(cfg_path):
    """Pull the two head settings that the checkpoint tensors cannot reveal.

    TRANSFORMER_INPUT is inferable from the token-embedding shape, but
    IEF_ITERS is not: the decoder weights are identical whether the head was
    run once or three times. It only lives in the config.
    """
    import yaml
    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)
    head = (cfg.get('MODEL', {}) or {}).get('SMPL_HEAD', {}) or {}
    return {
        'transformer_input': head.get('TRANSFORMER_INPUT', 'zero'),
        'ief_iters': int(head.get('IEF_ITERS', 1)),
    }


def _strip(sd):
    """Lightning checkpoints wrap the weights and sometimes prefix them."""
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    out = {}
    for k, v in sd.items():
        for p in ('model.', 'module.'):
            if k.startswith(p):
                k = k[len(p):]
        out[k] = v
    return out


def _align_prefix(sd, own):
    """Match the checkpoint's key layout to the model's.

    The full HMR2 checkpoint stores `backbone.*` / `smpl_head.*`. The standalone
    ViTPose backbone (vitpose_backbone.pth) stores bare ViT keys with no prefix
    at all, so it needs one added before anything will match.
    """
    if any(k in own for k in sd):
        return sd
    prefixed = {f'backbone.{k}': v for k, v in sd.items()}
    if any(k in own for k in prefixed):
        print("Checkpoint has bare backbone keys; adding the 'backbone.' prefix")
        return prefixed
    return sd


def read_checkpoint(ckpt_path):
    """Load the raw tensors once. The HMR2 checkpoint is ~2.5GB, so callers
    should pass the result around rather than re-reading it."""
    try:
        obj = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    except TypeError:          # torch < 1.13 has no weights_only
        obj = torch.load(ckpt_path, map_location='cpu')
    return _strip(obj)


def inspect_checkpoint(ckpt_path_or_sd):
    """Read the head geometry out of the checkpoint instead of assuming it."""
    sd = (ckpt_path_or_sd if isinstance(ckpt_path_or_sd, dict)
          else read_checkpoint(ckpt_path_or_sd))
    info = {'has_head': any(k.startswith('smpl_head.') for k in sd)}
    w = sd.get('smpl_head.transformer.to_token_embedding.weight')
    if w is not None:
        info['dim'] = int(w.shape[0])
        info['token_dim'] = int(w.shape[1])
    dp = sd.get('smpl_head.decpose.weight')
    if dp is not None:
        info['npose'] = int(dp.shape[0])
    prefix = 'smpl_head.transformer.transformer.layers.'
    depth = {int(k[len(prefix):].split('.')[0]) for k in sd if k.startswith(prefix)}
    if depth:
        info['depth'] = max(depth) + 1
    kv = sd.get('smpl_head.transformer.transformer.layers.0.1.fn.to_kv.weight')
    if kv is not None:
        info['context_dim'] = int(kv.shape[1])
    return info


def load_hmr2(ckpt_path, with_head=True, ief_iters=1, device='cuda',
              cfg_path=None, strict_report=True):
    """Build the model and load the released HMR2 weights.

    Pass cfg_path (the model_config.yaml sitting next to the checkpoint) to pick
    up IEF_ITERS, which cannot be recovered from the weights.

    Raises if the checkpoint does not populate the model, rather than silently
    returning a randomly initialised network -- which is exactly what
    `load_state_dict(..., strict=False)` would do here.
    """
    sd = read_checkpoint(ckpt_path)
    info = inspect_checkpoint(sd)

    if cfg_path:
        cfg = read_hmr2_config(cfg_path)
        ief_iters = cfg['ief_iters']
        print(f"Config: TRANSFORMER_INPUT={cfg['transformer_input']} "
              f"IEF_ITERS={cfg['ief_iters']}")
        # The config and the weights must agree about the token layout.
        cfg_token_dim_is_1 = (cfg['transformer_input'] != 'mean_shape')
        ckpt_token_dim = info.get('token_dim')
        if ckpt_token_dim is not None and cfg_token_dim_is_1 != (ckpt_token_dim == 1):
            raise ValueError(
                f"config says TRANSFORMER_INPUT={cfg['transformer_input']} but the "
                f"checkpoint's token embedding takes {ckpt_token_dim} inputs. "
                f"The config and checkpoint are from different runs.")
        if not cfg_token_dim_is_1 and ief_iters == 1:
            print("[warn] TRANSFORMER_INPUT=mean_shape with IEF_ITERS=1; check "
                  "that this matches how the model was trained")

    head_kwargs = {}
    if with_head:
        if not info.get('has_head'):
            raise ValueError(f"{ckpt_path} has no smpl_head.* weights; "
                             f"the cross-attention tap needs the full HMR2 checkpoint")
        head_kwargs = dict(
            token_dim=info.get('token_dim', 1),
            dim=info.get('dim', 1024),
            depth=info.get('depth', 6),
            context_dim=info.get('context_dim', 1280),
            npose=info.get('npose', 144),
            ief_iters=ief_iters,
        )
        print(f"Head geometry from checkpoint: {head_kwargs}")

    model = HMR2Features(with_head=with_head, head_kwargs=head_kwargs)
    own = model.state_dict()
    sd = _align_prefix(sd, own)

    usable = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
    missing = [k for k in own if k not in usable]
    shape_bad = [k for k, v in sd.items() if k in own and own[k].shape != v.shape]

    model.load_state_dict(usable, strict=False)

    if strict_report:
        print(f"Loaded {len(usable)}/{len(own)} tensors from {ckpt_path}")
        if shape_bad:
            raise RuntimeError(
                f"{len(shape_bad)} tensor(s) matched by name but not by shape, "
                f"e.g. {shape_bad[:3]}. The architecture here does not match the "
                f"checkpoint.")
        if missing:
            raise RuntimeError(
                f"{len(missing)} parameter(s) got no weights, e.g. {missing[:5]}. "
                f"Refusing to return a partly random model.")

    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    return model