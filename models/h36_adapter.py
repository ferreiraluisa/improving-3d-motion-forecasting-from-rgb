import torch
import torch.nn as nn

def coco2h36m_init():
    """ linear map from COCO-17 to H36M-17 
    coco has 12 joints that are in h36m, and 5 joints that are not in h36m (nose, eyes, ears)
    h36m has 5 joints that are not in coco (pelvis, spine, thorax, neck, head)
    do this linear mapping hard coded so that mlp have an idea of what to learn, instead of learning from scratch
    """
    A = torch.zeros(17, 17)
    A[0, 11] = A[0, 12] = 0.5                       # pelvis
    A[1, 12] = A[2, 14] = A[3, 16] = 1.0            # right leg
    A[4, 11] = A[5, 13] = A[6, 15] = 1.0            # left leg
    A[7, 11] = A[7, 12] = A[7, 5] = A[7, 6] = 0.25  # spine
    A[8, 5]  = A[8, 6]  = 0.5                       # thorax
    A[9, 0]  = 1.0                                  # neck/nose
    A[10, 1] = A[10, 2] = 0.5                       # head top (rough)
    A[11, 5] = A[12, 7] = A[13, 9] = 1.0            # left arm
    A[14, 6] = A[15, 8] = A[16, 10] = 1.0           # right arm
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


class Coco2H36mAdapter(nn.Module):
    """(B, T, 563) -> (B, T, 563), last 51 dims re-expressed in H36M-17 order."""
 
    def __init__(self, d_ctx=512, n_j=17, hidden=512, n_blocks=2,
                 dropout=0.1, use_ctx=False, temporal=True):
        super().__init__()
        self.d_ctx, self.n_j = d_ctx, n_j
        self.use_ctx, self.d_in = use_ctx, d_ctx + n_j * 3
 
        self.A = nn.Parameter(coco2h36m_init())
 
        self.kp_in = nn.Linear(n_j * 3, hidden)
        self.ctx_in = nn.Linear(d_ctx, hidden) if use_ctx else None
        self.temporal = (nn.Conv1d(hidden, hidden, 5, padding=2, groups=hidden)
                         if temporal else None)
        self.blocks = nn.Sequential(*[ResBlock(hidden, dropout)
                                      for _ in range(n_blocks)])
        self.out = nn.Linear(hidden, n_j * 3)
 
        # Zero-init the residual head so the model starts exactly at A.
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
 
    def split(self, x):
        kp = x[..., self.d_ctx:]
        return x[..., :self.d_ctx], kp.reshape(*kp.shape[:-1], self.n_j, 3)
 
    def forward_joints(self, x):
        """Return only the H36M joints, shape (..., 17, 3)."""
        ctx, kp = self.split(x)
        base = torch.einsum('ij,...jc->...ic', self.A, kp)
 
        h = self.kp_in(kp.flatten(-2))
        if self.use_ctx:
            h = h + self.ctx_in(ctx)
        if self.temporal is not None and h.dim() == 3:
            h = h + self.temporal(h.transpose(1, 2)).transpose(1, 2)
        h = self.blocks(h)
 
        return base + self.out(h).reshape(*base.shape)
 
    def forward(self, x):
        ctx, _ = self.split(x)
        kp = self.forward_joints(x)
        return torch.cat([ctx, kp.flatten(-2)], dim=-1)
 
    @torch.no_grad()
    def set_A(self, A):
        self.A.copy_(torch.as_tensor(A, dtype=self.A.dtype, device=self.A.device))