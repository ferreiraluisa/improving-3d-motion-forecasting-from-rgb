import torch
import torch.nn as nn

# SMPL-24 joint order
SMPL_NAMES = ['pelvis', 'l_hip', 'r_hip', 'spine1', 'l_knee', 'r_knee', 'spine2',
              'l_ank', 'r_ank', 'spine3', 'l_foot', 'r_foot', 'neck', 'l_collar',
              'r_collar', 'head', 'l_sho', 'r_sho', 'l_elb', 'r_elb', 'l_wri',
              'r_wri', 'l_hand', 'r_hand']

# SMPL kinematic tree (parent of each joint) -> bones for a bone-length loss
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14,
                16, 17, 18, 19, 20, 21]
SMPL_BONES = [(p, c) for c, p in enumerate(SMPL_PARENTS) if p >= 0]

# COCO-17: 0 nose, 1 l_eye, 2 r_eye, 3 l_ear, 4 r_ear, 5 l_sho, 6 r_sho,
#          7 l_elb, 8 r_elb, 9 l_wri, 10 r_wri, 11 l_hip, 12 r_hip,
#          13 l_knee, 14 r_knee, 15 l_ank, 16 r_ank


def coco2smpl_init():
    """ linear map from COCO-17 to SMPL-24 (A is 24x17, y_i = sum_j A[i,j] x_j)
    15 SMPL joints have a direct COCO counterpart (hips, knees, ankles, shoulders,
    elbows, wrists) or are simple averages (pelvis, head).
    the rest (spine1-3, neck, collars, feet, hands) are interpolated / extrapolated
    rough guesses so the MLP starts from something sensible.
    every row sums to 1 -> the map is translation-equivariant.
    """
    A = torch.zeros(24, 17)
    hip_l, hip_r, sho_l, sho_r = 11, 12, 5, 6

    # pelvis
    A[0, hip_l] = A[0, hip_r] = 0.5
    # legs
    A[1, 11] = A[4, 13] = A[7, 15] = 1.0            # left  hip / knee / ankle
    A[2, 12] = A[5, 14] = A[8, 16] = 1.0            # right hip / knee / ankle
    A[10, 15] = A[11, 16] = 1.0                     # feet ~ ankles (rough)

    # spine: interpolate pelvis -> thorax (mid-shoulders) at 1/4, 1/2, 3/4
    for row, t in [(3, 0.25), (6, 0.5), (9, 0.75)]:
        A[row, hip_l] = A[row, hip_r] = (1 - t) / 2
        A[row, sho_l] = A[row, sho_r] = t / 2

    # neck: mostly mid-shoulders, pulled up a bit towards the nose
    A[12, sho_l] = A[12, sho_r] = 0.4
    A[12, 0] = 0.2
    # collars: halfway between neck and each shoulder
    A[13, sho_l], A[13, sho_r], A[13, 0] = 0.7, 0.2, 0.1
    A[14, sho_r], A[14, sho_l], A[14, 0] = 0.7, 0.2, 0.1
    # head: centre of the skull ~ mid-ears
    A[15, 3] = A[15, 4] = 0.5

    # arms
    A[16, 5] = A[18, 7] = A[20, 9] = 1.0            # left  shoulder / elbow / wrist
    A[17, 6] = A[19, 8] = A[21, 10] = 1.0           # right shoulder / elbow / wrist
    # hands: extrapolate forearm past the wrist, 1.2*wri - 0.2*elb
    A[22, 9], A[22, 7] = 1.2, -0.2
    A[23, 10], A[23, 8] = 1.2, -0.2
    return A


class ResBlock(nn.Module):
    def __init__(self, d, p):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(d, d), nn.Linear(d, d)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.act, self.do = nn.GELU(), nn.Dropout(p)

    def forward(self, x):
        h = self.do(self.act(self.n1(self.fc1(x))))
        h = self.do(self.act(self.n2(self.fc2(h))))
        return x + h


class Coco2SmplAdapter(nn.Module):
    """(B, T, 563) -> (B, T, 584): ctx (512) kept, COCO-17 (51) -> SMPL-24 (72)."""

    def __init__(self, d_ctx=512, n_in=17, n_out=24, hidden=512, n_blocks=2,
                 dropout=0.1, use_ctx=False, temporal=True):
        super().__init__()
        self.d_ctx, self.n_in, self.n_out = d_ctx, n_in, n_out
        self.use_ctx = use_ctx
        self.d_in = d_ctx + n_in * 3        # 563
        self.d_out = d_ctx + n_out * 3      # 584

        self.A = nn.Parameter(coco2smpl_init())          # (24, 17)

        self.kp_in = nn.Linear(n_in * 3, hidden)
        self.ctx_in = nn.Linear(d_ctx, hidden) if use_ctx else None
        self.temporal = (nn.Conv1d(hidden, hidden, 5, padding=2, groups=hidden)
                         if temporal else None)
        self.blocks = nn.Sequential(*[ResBlock(hidden, dropout)
                                      for _ in range(n_blocks)])
        self.out = nn.Linear(hidden, n_out * 3)

        # Zero-init the residual head so the model starts exactly at A.
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def split(self, x):
        kp = x[..., self.d_ctx:]
        return x[..., :self.d_ctx], kp.reshape(*kp.shape[:-1], self.n_in, 3)

    def forward_joints(self, x):
        """Return only the SMPL joints, shape (..., 24, 3)."""
        ctx, kp = self.split(x)
        # Work on kps centred on their mean and add the centre back: the output
        # is then exactly translation-equivariant whatever A becomes during
        # training (rows need not sum to 1). Matters in 3DPW, where camera-space
        # depth is 2-10 m and any row-sum drift would turn into a huge offset.
        c = kp.mean(-2, keepdim=True)
        kpc = kp - c
        base = torch.einsum('ij,...jc->...ic', self.A, kpc) + c   # (..., 24, 3)

        h = self.kp_in(kpc.flatten(-2))
        if self.use_ctx:
            h = h + self.ctx_in(ctx)
        if self.temporal is not None and h.dim() == 3:
            h = h + self.temporal(h.transpose(1, 2)).transpose(1, 2)
        h = self.blocks(h)

        return base + self.out(h).reshape(*base.shape)

    def forward(self, x):
        ctx, _ = self.split(x)
        kp = self.forward_joints(x)
        return torch.cat([ctx, kp.flatten(-2)], dim=-1)      # (..., 584)

    @torch.no_grad()
    def set_A(self, A):
        self.A.copy_(torch.as_tensor(A, dtype=self.A.dtype, device=self.A.device))