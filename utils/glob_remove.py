import torch
from visualize.visualize import helper_plot_3d_2im, helper_plot_2d_2im



def remove_rotation_and_translation_batch(poses_3d, j_root=0, j_left=4, j_right=1):
    """
    Normalize skeleton poses by centering at hip midpoint and aligning hips along x-axis.

    Args:
        poses_3d: torch.Tensor of shape [batch_size, frame_length, num_joints, 3]
        j_root: int, index of the mid-hip joint (will be redefined)
        j_left: int, index of the left hip joint
        j_right: int, index of the right hip joint

    Returns:
        poses_3d_modified: torch.Tensor of same shape as input
                          Normalized poses where:
                          - mid-hip is at [0, 0, 0]
                          - right hip is at [-1, 0, 0]
                          - left hip is at [1, 0, 0]
        scale: torch.Tensor of shape [batch_size, frame_length]
    """
    poses_3d_modified = poses_3d.clone()  # [batch, len, j, 3]

    # --- Step 1: Center at mid-hip ---
    right_hip = poses_3d_modified[:, :, j_right, :]  # [b, len, 3]
    left_hip  = poses_3d_modified[:, :, j_left,  :]  # [b, len, 3]
    new_mid_hip = (right_hip + left_hip) / 2.0        # [b, len, 3]
    poses_3d_modified = poses_3d_modified - new_mid_hip.unsqueeze(2)  # [b, len, j, 3]

    # Recompute hip positions after centering
    right_hip  = poses_3d_modified[:, :, j_right, :]  # [b, len, 3]
    left_hip   = poses_3d_modified[:, :, j_left,  :]  # [b, len, 3]
    hip_vector = left_hip - right_hip                  # [b, len, 3]

    # --- Step 2: Build Ry — rotate hip in XZ plane to align with +X (zero out Z) ---
    # H36M camera coordinate system: X=left-right, Y=DOWN (vertical), Z=forward (depth)
    # When person turns, hip vector rotates in XZ plane (ground plane)
    # Ry rotates around Y-axis → only affects X and Z → Y (vertical) is NEVER touched
    # atan2(z, x) gives the angle of hip vector in XZ plane
    # No negation needed for Ry under v @ R^T convention
    batch_size, frame_length, num_joints, _ = poses_3d_modified.shape
    angle_y = torch.atan2(hip_vector[..., 2], hip_vector[..., 0])  # [b, len]

    cos_y  = torch.cos(angle_y)
    sin_y  = torch.sin(angle_y)
    zeros  = torch.zeros_like(cos_y)
    ones   = torch.ones_like(cos_y)

    Ry = torch.stack([
        torch.stack([ cos_y, zeros,  sin_y], dim=2),
        torch.stack([ zeros,  ones,  zeros], dim=2),
        torch.stack([-sin_y, zeros,  cos_y], dim=2),
    ], dim=2)  # [b, len, 3, 3]

    # --- Step 3: Build Rz — rotate residual to align hip fully with +X (zero out Y tilt) ---
    # After Ry, hip vector lies in XY plane: [d_x, d_y, ~0]
    # atan2(y, x) gives the remaining tilt angle
    # NEGATION needed for Rz under v @ R^T convention (Rz has asymmetric sign pattern)
    hip_flat = hip_vector.view(batch_size * frame_length, 1, 3)
    Ry_flat  = Ry.view(batch_size * frame_length, 3, 3)
    hip_after_Ry = torch.matmul(hip_flat, Ry_flat.transpose(1, 2))  # [b*len, 1, 3]
    hip_after_Ry = hip_after_Ry.view(batch_size, frame_length, 3)   # [b, len, 3]

    angle_z = -torch.atan2(hip_after_Ry[..., 1], hip_after_Ry[..., 0])  # [b, len]

    cos_z = torch.cos(angle_z)
    sin_z = torch.sin(angle_z)

    Rz = torch.stack([
        torch.stack([ cos_z, -sin_z, zeros], dim=2),
        torch.stack([ sin_z,  cos_z, zeros], dim=2),
        torch.stack([ zeros,  zeros,  ones], dim=2),
    ], dim=2)  # [b, len, 3, 3]

    # --- Step 4: Compute scale ---
    # Both Ry and Rz are pure rotations → ||hip|| is invariant → safe to use full norm
    hip_distance = torch.sqrt(
        hip_vector[..., 0] ** 2 +
        hip_vector[..., 1] ** 2 +
        hip_vector[..., 2] ** 2
    )
    target_hip_distance = 2.0
    scale = target_hip_distance / (hip_distance + 1e-8)  # [b, len]

    # --- Step 5: Combined rotation R = Rz @ Ry (apply Ry first, then Rz) ---
    Ry_flat = Ry.view(batch_size * frame_length, 3, 3)
    Rz_flat = Rz.view(batch_size * frame_length, 3, 3)
    rotation_flat = torch.matmul(Rz_flat, Ry_flat)  # [b*len, 3, 3]

    # --- Step 6: Apply scale then rotation ---
    scale_expanded = scale.unsqueeze(2).unsqueeze(3)  # [b, len, 1, 1]
    poses_3d_modified = poses_3d_modified * scale_expanded  # [b, len, j, 3]

    poses_flat   = poses_3d_modified.view(batch_size * frame_length, num_joints, 3)
    poses_rotated = torch.matmul(poses_flat, rotation_flat.transpose(1, 2))
    poses_3d_modified = poses_rotated.view(batch_size, frame_length, num_joints, 3)

    # --- Step 7: Conditional Y-flip to ensure consistent upright orientation ---
    # H36M camera coords: Y points DOWN → head has smaller Y than knees.
    # After rotation, the spine/thorax should still have smaller Y than knees.
    #
    # H36M 17-joint indices:
    #   spine=7, thorax=8  (upper body — should have SMALLER Y, i.e. higher)
    #   right_knee=2, left_knee=5  (lower body — should have LARGER Y, i.e. lower)
    j_spine, j_thorax = 7, 8
    j_right_knee, j_left_knee = 2, 5

    upper_y = (poses_3d_modified[:, :, j_spine, 1] +
               poses_3d_modified[:, :, j_thorax, 1]) / 2.0       # [b, len]
    lower_y = (poses_3d_modified[:, :, j_right_knee, 1] +
               poses_3d_modified[:, :, j_left_knee, 1]) / 2.0    # [b, len]

    # Y points down → upright means upper_y < lower_y
    # If upper_y > lower_y → skeleton is inverted → flip Y
    needs_flip = upper_y > lower_y                                # [b, len]  bool
    flip = torch.where(needs_flip,
                       torch.tensor(-1.0, device=poses_3d_modified.device),
                       torch.tensor(1.0, device=poses_3d_modified.device))  # [b, len]
    poses_3d_modified[..., 1] = poses_3d_modified[..., 1] * flip.unsqueeze(2)

    # --- Optional visualization ---
    visualize = False
    if visualize:
        i = 5
        from visualize.visualize import helper_plot_3d_2im
        helper_plot_3d_2im(poses_3d_modified[i], poses_3d_modified[i], 'Rotation', 17)
        # helper_plot_3d_2im(poses_3d[i],           poses_3d[i],          'pose_3d_',     17, axis_limit=1)

    return poses_3d_modified, scale



    """
    Normalize skeleton poses by centering at hip midpoint and aligning hips along x-axis.
    Uses angular unwrapping for temporal continuity.

    Args:
        poses_3d: torch.Tensor of shape [batch_size, frame_length, num_joints, 3]
        j_root: int, index of the mid-hip joint
        j_left: int, index of the left hip joint
        j_right: int, index of the right hip joint

    Returns:
        poses_3d_modified: torch.Tensor of same shape as input
        scale: torch.Tensor of shape [batch_size, frame_length]
    """
    poses_3d_modified = poses_3d.clone()

    # --- Step 1: Center at mid-hip ---
    right_hip = poses_3d_modified[:, :, j_right, :]
    left_hip  = poses_3d_modified[:, :, j_left,  :]
    new_mid_hip = (right_hip + left_hip) / 2.0
    poses_3d_modified = poses_3d_modified - new_mid_hip.unsqueeze(2)

    right_hip  = poses_3d_modified[:, :, j_right, :]
    left_hip   = poses_3d_modified[:, :, j_left,  :]
    hip_vector = left_hip - right_hip

    # --- Step 2: Compute raw angle_z ---
    raw_angle_z = -torch.atan2(hip_vector[..., 1], hip_vector[..., 0])  # [b, len], range [-pi, pi]

    # --- Step 2b: Unwrap angle_z for temporal continuity ---
    # Compute frame-to-frame differences
    diff = raw_angle_z[:, 1:] - raw_angle_z[:, :-1]  # [b, len-1]
    # Wrap differences to [-pi, pi]
    diff = torch.remainder(diff + torch.pi, 2 * torch.pi) - torch.pi
    # Cumulative sum of wrapped differences + initial angle = unwrapped angle
    angle_z = torch.cat([
        raw_angle_z[:, :1],  # keep frame 0 as-is
        raw_angle_z[:, :1] + torch.cumsum(diff, dim=1)  # accumulate smooth deltas
    ], dim=1)  # [b, len]

    # --- Step 2c: Correct frame 0 so right hip is on negative X ---
    batch_size, frame_length, num_joints, _ = poses_3d_modified.shape
    zeros = torch.zeros_like(angle_z)
    ones  = torch.ones_like(angle_z)

    # Build Rz just for frame 0 to test
    cos_z0 = torch.cos(angle_z[:, 0])
    sin_z0 = torch.sin(angle_z[:, 0])
    Rz0 = torch.stack([
        torch.stack([cos_z0, -sin_z0, torch.zeros_like(cos_z0)], dim=1),
        torch.stack([sin_z0,  cos_z0, torch.zeros_like(cos_z0)], dim=1),
        torch.stack([torch.zeros_like(cos_z0), torch.zeros_like(cos_z0), torch.ones_like(cos_z0)], dim=1),
    ], dim=1)  # [b, 3, 3]

    right_hip_0 = right_hip[:, 0, :]  # [b, 3]
    right_hip_after = torch.matmul(right_hip_0.unsqueeze(1), Rz0.transpose(1, 2)).squeeze(1)

    needs_correction = right_hip_after[:, 0] > 0  # [b]
    correction = torch.where(needs_correction,
                             torch.tensor(torch.pi, device=angle_z.device),
                             torch.tensor(0.0, device=angle_z.device))  # [b]
    # Apply correction to ALL frames (shifts the entire unwrapped sequence)
    angle_z = angle_z + correction.unsqueeze(1)  # [b, len]

    # --- Step 2d: Build Rz from unwrapped + corrected angles ---
    cos_z = torch.cos(angle_z)
    sin_z = torch.sin(angle_z)
    zeros = torch.zeros_like(cos_z)
    ones  = torch.ones_like(cos_z)

    Rz = torch.stack([
        torch.stack([ cos_z, -sin_z, zeros], dim=2),
        torch.stack([ sin_z,  cos_z, zeros], dim=2),
        torch.stack([ zeros,  zeros,  ones], dim=2),
    ], dim=2)

    # --- Step 3: Build Ry ---
    hip_flat = hip_vector.view(batch_size * frame_length, 1, 3)
    Rz_flat  = Rz.view(batch_size * frame_length, 3, 3)
    hip_after_Rz = torch.matmul(hip_flat, Rz_flat.transpose(1, 2))
    hip_after_Rz = hip_after_Rz.view(batch_size, frame_length, 3)

    angle_y = torch.atan2(hip_after_Rz[..., 2], hip_after_Rz[..., 0])

    cos_y = torch.cos(angle_y)
    sin_y = torch.sin(angle_y)

    Ry = torch.stack([
        torch.stack([ cos_y, zeros,  sin_y], dim=2),
        torch.stack([ zeros,  ones,  zeros], dim=2),
        torch.stack([-sin_y, zeros,  cos_y], dim=2),
    ], dim=2)

    # --- Step 4: Compute scale ---
    hip_distance = torch.sqrt(
        hip_vector[..., 0] ** 2 +
        hip_vector[..., 1] ** 2 +
        hip_vector[..., 2] ** 2
    )
    target_hip_distance = 2.0
    scale = target_hip_distance / (hip_distance + 1e-8)

    # --- Step 5: Combined rotation R = Ry @ Rz ---
    Rz_flat = Rz.view(batch_size * frame_length, 3, 3)
    Ry_flat = Ry.view(batch_size * frame_length, 3, 3)
    rotation_flat = torch.matmul(Ry_flat, Rz_flat)

    # --- Step 6: Apply scale then rotation ---
    scale_expanded = scale.unsqueeze(2).unsqueeze(3)
    poses_3d_modified = poses_3d_modified * scale_expanded

    poses_flat = poses_3d_modified.view(batch_size * frame_length, num_joints, 3)
    poses_rotated = torch.matmul(poses_flat, rotation_flat.transpose(1, 2))
    poses_3d_modified = poses_rotated.view(batch_size, frame_length, num_joints, 3)

    # --- Step 7: Y-flip based on frame 0 ---
    j_spine, j_thorax = 7, 8
    j_right_knee, j_left_knee = 2, 5

    upper_y_0 = (poses_3d_modified[:, 0, j_spine, 1] +
                 poses_3d_modified[:, 0, j_thorax, 1]) / 2.0
    lower_y_0 = (poses_3d_modified[:, 0, j_right_knee, 1] +
                 poses_3d_modified[:, 0, j_left_knee, 1]) / 2.0

    needs_flip = upper_y_0 < lower_y_0
    flip = torch.where(needs_flip,
                       torch.tensor(-1.0, device=poses_3d_modified.device),
                       torch.tensor(1.0, device=poses_3d_modified.device))
    poses_3d_modified[..., 1] = poses_3d_modified[..., 1] * flip.unsqueeze(1).unsqueeze(2)

    visualize = True
    if visualize:
        i = 5
        from visualize.visualize import helper_plot_3d_2im
        helper_plot_3d_2im(poses_3d_modified[i], poses_3d_modified[i], 'Rotation', 17)
        # helper_plot_3d_2im(poses_3d[i],           poses_3d[i],          'pose_3d_',     17, axis_limit=1)

    return poses_3d_modified, scale

