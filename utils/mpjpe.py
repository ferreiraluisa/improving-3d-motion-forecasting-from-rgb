import torch

def mpjpe_3d(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1))



def loss_velocity(predicted, target):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape
    if predicted.shape[1]<=1:
        return torch.FloatTensor(1).fill_(0.)[0].to(predicted.device)
    velocity_predicted = predicted[:,1:] - predicted[:,:-1]
    velocity_target = target[:,1:] - target[:,:-1]
    return torch.mean(torch.norm(velocity_predicted - velocity_target, dim=-1))


def n_mpjpe(predicted, target):
    """
    Normalized MPJPE (scale only), adapted from:
    https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning/blob/master/losses/poses.py
    """
    assert predicted.shape == target.shape
    norm_predicted = torch.mean(torch.sum(predicted**2, dim=3, keepdim=True), dim=2, keepdim=True)
    norm_target = torch.mean(torch.sum(target*predicted, dim=3, keepdim=True), dim=2, keepdim=True)
    scale = norm_target / norm_predicted
    return mpjpe_3d(scale * predicted, target)



H36M_EDGES = [
    (0,1),(1,2),(2,3),
    (0,4),(4,5),(5,6),
    (0,7),(7,8),(8,9),(9,10),
    (8,11),(11,12),(12,13),
    (8,14),(14,15),(15,16),
]

# Pre-build edge index tensors once at module load time (moved to device inside each loss)
_EDGE_SRC = torch.tensor([e[0] for e in H36M_EDGES], dtype=torch.long)  # (E,)
_EDGE_DST = torch.tensor([e[1] for e in H36M_EDGES], dtype=torch.long)  # (E,)

# Bone length loss: encourage predicted bone lengths to match GT at every timestep.
# pred, gt: (B,T,J,3)
def bone_length_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    src = _EDGE_SRC.to(pred.device)
    dst = _EDGE_DST.to(pred.device)
    pred_bones = pred[:, :, dst] - pred[:, :, src]   # (B,T,E,3)
    gt_bones   = gt[:, :, dst]   - gt[:, :, src]     # (B,T,E,3)

    return torch.mean(torch.norm(pred_bones - gt_bones, dim=-1))  # mean L2 bone length error