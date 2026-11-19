import os
import sys
import argparse
from tqdm import tqdm
from datetime import datetime
import numpy as np
import random
import wandb

# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist

# model
from models.phd_model import PHDFor3DJoints

# dataset
from dataset.losd_data_h36 import Fusion as FusionH36M
from dataset.losd_data_3dpw import Fusion3DPW
from dataset.dataloader import Human36mDataset, ThreeDPWDataset

from visualize.visualize import helper_plot_3d_2im

# utils
from utils.mpjpe import mpjpe_3d, bone_length_loss, loss_velocity
from utils.config_tools import get_config as get_config_aux
from utils.ddp_utils import reduce_tensor


SPLITS_3DPW = {'train': 'train', 'val': 'validation', 'test': 'test'}

H36M_ACTIONS = [
    "walking", "eating", "smoking", "discussion", "directions",
    "greeting", "phoning", "posing", "purchases", "sitting",
    "sittingdown", "takingphoto", "waiting", "walkingdog", "walkingtogether",
]

# CLI args that the yaml config must NOT override (the yaml is merged on top of
# the CLI below, and an H36M yaml could otherwise silently reset the dataset).
DATASET_ARGS = ('dataset', 'dataset_3dpw_path', 'dataset_3dpw_feats', 'img_dir_3dpw')


# ─────────────────────────── dataset selection ───────────────────────────────

def build_raw_dataset(args):
    """Returns (Fusion class, raw dataset object)."""
    if args.dataset == 'h36m':
        return FusionH36M, Human36mDataset(args.dataset_path)
    if args.dataset == '3dpw':
        return Fusion3DPW, ThreeDPWDataset(args.dataset_3dpw_path)
    raise ValueError(args.dataset)


def eval_groups(args, dataset, mode_t):
    """
    List of (name, extra kwargs for the Fusion constructor) used for val/test.
      H36M : one group per action       -> action=<act>
      3DPW : one group per source video -> sequences=[<seq>]
    """
    if args.dataset == 'h36m':
        return [(a, dict(action=a)) for a in H36M_ACTIONS]

    split = SPLITS_3DPW[mode_t]
    seqs = sorted({e['source_seq'] for e in dataset[split].values()})
    return [(s, dict(sequences=[s])) for s in seqs]


def setup_dataset_args(args):
    """Dataset-specific checks / fixed settings (call after the yaml merge)."""
    if args.dataset == '3dpw':
        missing = [n for n in ('dataset_3dpw_path', 'dataset_3dpw_feats') if not getattr(args, n)]
        if missing:
            raise ValueError(f'--dataset 3dpw needs: {", ".join("--" + m for m in missing)}')
        args.out_all = True          # loader must return all T frames of gt_3D
        args.split_train_3dpw = SPLITS_3DPW['train']
        args.split_val_3dpw = SPLITS_3DPW['val']
        args.split_test_3dpw = SPLITS_3DPW['test']
        args.normalize_2d_3dpw = True
    return args


def build_eval_loaders(args, Fusion, dataset, mode_t, world_size, rank, workers):
    """One DataLoader per eval group. Groups with no full window (3DPW segments
    shorter than seq_len) are skipped so they never produce a 0/0 MPJPE."""
    loaders, sets = {}, {}
    for name, kw in eval_groups(args, dataset, mode_t):
        ds = Fusion(args, dataset, train=False, mode_t=mode_t, **kw)
        if len(ds) == 0:
            print(f"[{mode_t}] skip group '{name}': no window of {args.seq_len} frames")
            continue
        sampler = torch.utils.data.distributed.DistributedSampler(
            ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        loaders[name] = torch.utils.data.DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            num_workers=workers, pin_memory=True, sampler=sampler)
        sets[name] = ds
    if not loaders:
        raise RuntimeError(f'No {mode_t} group has a window of {args.seq_len} frames.')
    return loaders, sets


def check_shapes(args, probe_ds):
    """Adapt num_joints / latent_dim to the real data (3DPW features and joints
    differ from H36M) and check the window length."""
    sample = probe_ds[0]
    real_feat_dim = int(sample[3].shape[-1])
    real_joints = int(sample[1].shape[-2])
    real_T = int(sample[1].shape[0])
    if real_joints != args.num_joints:
        print(f"[warn] num_joints={args.num_joints} but GT has {real_joints}; using {real_joints}.")
        args.num_joints = real_joints
    if real_feat_dim != args.latent_dim:
        print(f"[warn] --latent_dim={args.latent_dim} but features are {real_feat_dim}-d; "
              f"using {real_feat_dim}.")
        args.latent_dim = real_feat_dim
    assert real_T == args.seq_len, (
        f"Dataset returns {real_T} frames per window but past+future={args.seq_len}.")


# ─────────────────────────── train / val / test ──────────────────────────────

def train_epoch(model, args, dataloader, optimizer, epoch, writer, local_rank):
    model.train()
    epoch_loss         = 0.0
    epoch_loss_l3d     = 0.0
    epoch_loss_bone    = 0.0
    epoch_loss_vel     = 0.0
    epoch_loss_delta5  = 0.0
    epoch_loss_delta10 = 0.0
    counter = 0

    torch.cuda.set_device(local_rank)
    pbar = tqdm(dataloader, desc=f'Epoch {epoch + 1}', disable=(dist.get_rank() != 0))

    for data in pbar:
        # data: (cam, gt_3D, input_2D_GT, feat_window, action, subject, cam_ind)
        batch_size = data[1].size(0)

        feat_window = data[3].to(local_rank, non_blocking=True)   # (B, T, 2048)
        gt_3D       = data[1].to(local_rank, non_blocking=True)   # (B, T, J, 3)

        optimizer.zero_grad()

        # FIX 3: Use new phase 1 API
        phi, joints_dict = model.module.forward_phase1(feat_window)

        # FIX 1: Loss over ALL frames, not just the last one
        l3d   = mpjpe_3d(joints_dict[0], gt_3D)
        lbone = bone_length_loss(joints_dict[0], gt_3D)
        # Velocity term (paper's L_const role — temporal consistency)
        l_vel = loss_velocity(joints_dict[0], gt_3D)

        # FIX 2: Delta offset losses to force f_movie to encode temporal info
        l_delta5  = mpjpe_3d(joints_dict[-5][:, 5:, :, :],   gt_3D[:, :-5, :, :])
        l_delta10 = mpjpe_3d(joints_dict[-10][:, 10:, :, :], gt_3D[:, :-10, :, :])

        loss = l3d + args.lambda_bone * lbone + args.lambda_vel * l_vel + l_delta5 + l_delta10

        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                args.grad_clip
            )
        optimizer.step()

        epoch_loss         += loss.item()       * batch_size
        epoch_loss_l3d     += l3d.item()        * batch_size
        epoch_loss_bone    += lbone.item()      * batch_size
        epoch_loss_vel     += l_vel.item()      * batch_size
        epoch_loss_delta5  += l_delta5.item()   * batch_size
        epoch_loss_delta10 += l_delta10.item()  * batch_size
        counter            += batch_size

        pbar.set_postfix({
            'loss':  f'{loss.item():.4f}',
            'l3d':   f'{l3d.item():.4f}',
            'lbone': f'{lbone.item():.4f}',
            'vel':   f'{l_vel.item():.4f}',
            'd5':    f'{l_delta5.item():.4f}',
            'd10':   f'{l_delta10.item():.4f}',
            'LR':    f'{optimizer.param_groups[0]["lr"]:.2e}',
        })

    avg_loss         = epoch_loss         / counter
    avg_loss_l3d     = epoch_loss_l3d     / counter
    avg_loss_bone    = epoch_loss_bone    / counter
    avg_loss_vel     = epoch_loss_vel     / counter
    avg_loss_delta5  = epoch_loss_delta5  / counter
    avg_loss_delta10 = epoch_loss_delta10 / counter

    if dist.is_initialized():
        dev = torch.device(f'cuda:{local_rank}')
        ws  = dist.get_world_size()
        avg_loss         = reduce_tensor(torch.tensor(avg_loss,         device=dev), ws).item()
        avg_loss_l3d     = reduce_tensor(torch.tensor(avg_loss_l3d,     device=dev), ws).item()
        avg_loss_bone    = reduce_tensor(torch.tensor(avg_loss_bone,    device=dev), ws).item()
        avg_loss_vel     = reduce_tensor(torch.tensor(avg_loss_vel,     device=dev), ws).item()
        avg_loss_delta5  = reduce_tensor(torch.tensor(avg_loss_delta5,  device=dev), ws).item()
        avg_loss_delta10 = reduce_tensor(torch.tensor(avg_loss_delta10, device=dev), ws).item()

    return avg_loss, avg_loss_l3d, avg_loss_bone, avg_loss_vel, avg_loss_delta5, avg_loss_delta10


def validate_epoch(model, args, dataloader, local_rank):
    model.eval()
    torch.cuda.set_device(local_rank)

    epoch_mpjpe = torch.zeros(1).to(local_rank)
    counter = 0

    with torch.no_grad():
        for data in dataloader:
            batch_size = data[1].size(0)

            feat_window = data[3].to(local_rank, non_blocking=True)
            gt_3D       = data[1].to(local_rank, non_blocking=True)

            # FIX 3: Use new phase 1 API
            phi, joints_dict = model.module.forward_phase1(feat_window)

            # FIX 1: Evaluate over all frames
            batch_mpjpe = mpjpe_3d(joints_dict[0], gt_3D)
            epoch_mpjpe += batch_mpjpe * batch_size
            counter     += batch_size

    avg_mpjpe = epoch_mpjpe / counter

    if dist.is_initialized():
        avg_mpjpe = reduce_tensor(avg_mpjpe, dist.get_world_size())

    return avg_mpjpe.item()


def test_epoch(model, args, dataloader, local_rank, act):
    model.eval()
    torch.cuda.set_device(local_rank)

    total_mpjpe = 0.0
    counter = 0

    with torch.no_grad():
        for data in dataloader:
            batch_size = data[1].size(0)

            feat_window = data[3].to(local_rank, non_blocking=True)
            gt_3D       = data[1].to(local_rank, non_blocking=True)

            # FIX 3: Use new phase 1 API
            phi, joints_dict = model.module.forward_phase1(feat_window)
            joints_pred = joints_dict[0]

            # FIX 1: Evaluate over all frames
            batch_mpjpe = mpjpe_3d(joints_pred, gt_3D)
            total_mpjpe += batch_mpjpe.item() * batch_size
            counter     += batch_size

    avg_mpjpe = total_mpjpe / counter

    if dist.get_rank() == 0:
        if act in ['walking', 'walkingdog', 'walkingtogether']:
            i = 3
            os.makedirs('pic', exist_ok=True)
            helper_plot_3d_2im(
                joints_pred[i],  # (T, J, 3) estimated
                gt_3D[i],        # (T, J, 3) GT
                name=f'esttimated_phase1_{act}.png',
                n_joint=args.num_joints,
                axis_limit=1.0
            )
            print("[VIS] Saved GT vs estimated frames to pic/")

    if dist.is_initialized():
        avg_mpjpe_t = torch.tensor(avg_mpjpe, device=torch.device(f'cuda:{local_rank}'))
        avg_mpjpe_t = reduce_tensor(avg_mpjpe_t, dist.get_world_size())
        avg_mpjpe = avg_mpjpe_t.item()

    avg_mpjpe_mm = avg_mpjpe * 1000
    print(f"  MPJPE: {avg_mpjpe_mm:.2f} mm")
    return avg_mpjpe_mm


# ─────────────────────────── checkpoint helpers ──────────────────────────────

def save_checkpoint(model, optimizer, scheduler, epoch, loss, path):
    if dist.get_rank() != 0:
        return
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'loss': loss,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")


def load_checkpoint_for_resume(model, optimizer, scheduler, path):
    checkpoint = torch.load(path, map_location='cpu')

    state_dict = checkpoint['model_state_dict']
    has_module = any(k.startswith('module.') for k in state_dict)
    if has_module:
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.module.load_state_dict(state_dict)

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scheduler.base_lrs = [g['lr'] for g in optimizer.param_groups]

    start_epoch = checkpoint['epoch'] + 1
    best_loss   = checkpoint['loss']
    print(f"Resumed from epoch {checkpoint['epoch']}, best loss: {best_loss:.6f}")
    return start_epoch, best_loss


# ─────────────────────────── main ────────────────────────────────────────────

from torch.distributed.elastic.multiprocessing.errors import record

@record
def main():
    seed = 1126
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark     = False
    # torch.backends.cudnn.deterministic = True

    parser = argparse.ArgumentParser()

    # ── dataset choice ──
    parser.add_argument('--dataset', type=str, default='h36m', choices=['h36m', '3dpw'])

    # ── 3DPW paths (splits are always train / validation / test, eval per video) ──
    parser.add_argument('--dataset_3dpw_path',  type=str, default=None,
                        help='3DPW meta file read by dataloader.ThreeDPWDataset.')
    parser.add_argument('--dataset_3dpw_feats', type=str, default=None,
                        help='3DPW features .npy (--out of losd_data_3dpw.py).')
    parser.add_argument('--img_dir_3dpw',       type=str, default=None,
                        help='<3DPW root>/imageFiles (only needed for frame_info / viz).')

    # ── H36M paths ──
    parser.add_argument('--dataset_path',         type=str, default='data/data_3d_h36m.npz')
    parser.add_argument('--dataset_2d_path',      type=str, default='data/data_2d_h36m_gt.npz')
    # parser.add_argument('--dataset_2d_vit_path',  type=str, default='data/compatible_keypoints.pkl')
    parser.add_argument('--features_path', type=str, default='output/checkpoints')

    # ── checkpoint ──
    parser.add_argument('--checkpoint_dir',        type=str, default='work_dir/checkpoints')
    parser.add_argument('--checkpoint_save_name',  type=str, default='best_phd_model_phase1.pth')
    parser.add_argument('--resume',                type=str, default='',
                        help='Path to checkpoint to resume training from')

    # ── work dir ──
    parser.add_argument('--work_dir', type=str, default='./work_dir')

    # ── config (for seq_len / joint count etc.) ──
    parser.add_argument('--config_path', type=str, default='config/MB_train_h36m_long.yaml')

    # ── training hyper-params ──
    parser.add_argument('--epochs',      type=int,   default=80)
    parser.add_argument('--batch_size',  type=int,   default=16)
    parser.add_argument('--lr',            type=float, default=3e-5)
    parser.add_argument('--weight_decay',  type=float, default=1e-3)
    parser.add_argument('--warmup_epochs', type=int,   default=3)
    parser.add_argument('--grad_clip',     type=float, default=1.0)
    parser.add_argument('--lambda_bone', type=float, default=1.0)
    parser.add_argument('--lambda_vel',  type=float, default=1.0)
    parser.add_argument('--save_freq',   type=int,   default=10)
    parser.add_argument('--val_freq',    type=int,   default=1)
    parser.add_argument('--workers',     type=int,   default=16)
    parser.add_argument('--workers_val', type=int,   default=2)

    # ── dataset config ──
    parser.add_argument('--subjects_train', type=str, default='S1') #,S5,S6,S7,S8')
    parser.add_argument('--subjects_val',   type=str, default= 'S1') #'S11')
    parser.add_argument('--subjects_test',  type=str, default= 'S1') #'S9')
    parser.add_argument('--data_augmentation',     type=bool, default=False)
    parser.add_argument('--reverse_augmentation',  type=bool, default=False)
    parser.add_argument('--downsample',  type=int,  default=2)
    parser.add_argument('--stride',      type=int,  default=2)
    parser.add_argument('--img_size', nargs=2, type=int, default=[192, 256])

    # ── model ──
    parser.add_argument('--latent_dim', type=int,   default=2048)
    parser.add_argument('--dropout',    type=float, default=0.2)
    parser.add_argument('--mode',       type=str,   default='train', help='train / test')

    # ──── wandb ────
    parser.add_argument('--wandb_project', type=str, default='baseline_hmr')
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--wandb_mode', type=str, default='online', choices=['online', 'offline', 'disabled'])

    
    args = parser.parse_args()


    # merge yaml config (provides past_length, future_length, num_joints, …)
    cli_dataset_args = {k: getattr(args, k) for k in DATASET_ARGS}
    args2 = get_config_aux(args.config_path)
    args_dict = vars(args)
    args_dict.update(vars(args2))
    args_dict.update(cli_dataset_args)
    args = argparse.Namespace(**args_dict)

    args.past_length = 15
    args.future_length = 25
    args.seq_len    = args.past_length + args.future_length
    args.frames     = args.seq_len
    args.root_joint = 0
    args = setup_dataset_args(args)

    # ── DDP init ──
    world_size = int(os.environ['WORLD_SIZE'])
    rank       = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=world_size, rank=rank,
                            device_id=torch.device(f'cuda:{local_rank}'))
    dist.barrier()

    if rank == 0:
        print(f"DDP initialized, world_size={world_size}", flush=True)

    # silence non-rank-0 workers
    if dist.get_rank() != 0:
        f = open(os.devnull, 'w')
        sys.stdout = f
        sys.stderr = f

    # ── dirs / tensorboard ──
    if dist.get_rank() == 0:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        os.makedirs(args.work_dir,       exist_ok=True)
        log_dir = os.path.join(args.work_dir, 'logs', datetime.now().strftime('%Y%m%d_%H%M%S'))
        writer  = SummaryWriter(log_dir)

        wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or datetime.now().strftime('%Y%m%d_%H%M%S'),
        config=vars(args),
        mode=args.wandb_mode,
        )
    else:
        writer = None

    # ── dataset ──
    Fusion, dataset = build_raw_dataset(args)

    if args.mode == 'train':
        train_data = Fusion(args, dataset, train=True)
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_data, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, sampler=train_sampler,
            persistent_workers=args.workers > 0,
            prefetch_factor=2 if args.workers > 0 else None, drop_last=True)

        val_loaders, _ = build_eval_loaders(
            args, Fusion, dataset, 'val', world_size, rank, args.workers_val)
        acts = list(val_loaders)
        probe_ds = train_data
    else:
        test_loaders, test_sets = build_eval_loaders(
            args, Fusion, dataset, 'test', world_size, rank, args.workers)
        acts = list(test_loaders)
        probe_ds = test_sets[acts[0]]

    print(f"[{args.dataset}] {len(acts)} eval group(s): "
          f"{'actions' if args.dataset == 'h36m' else 'videos'}")
    check_shapes(args, probe_ds)

    # ── model ──
    model = PHDFor3DJoints(
        latent_dim=args.latent_dim,
        joints_num=args.num_joints,
        dropout=args.dropout,
    )
    model = model.to(local_rank)

    # FIX 6: No find_unused_parameters needed since f_AR is excluded from optimizer
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank,
        find_unused_parameters=True)  # still needed because f_AR params exist in DDP but aren't used in forward_phase1

    print(f"PHD model params: {sum(p.numel() for p in model.parameters()):,}")

    # FIX 4: Only train f_movie + f_3D in phase 1, exclude f_AR
    optimizer = optim.AdamW(
        list(model.module.f_movie.parameters()) + list(model.module.f_3D.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )

    warmup  = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs)
    cosine  = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=args.lr * 0.01)
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[args.warmup_epochs])

    # ── train ──
    if args.mode == 'train':
        start_epoch   = 0
        best_val_loss = float('inf')

        if args.resume and os.path.exists(args.resume):
            print(f"Resuming from: {args.resume}")
            start_epoch, best_val_loss = load_checkpoint_for_resume(
                model, optimizer, scheduler, args.resume)

        for epoch in range(start_epoch, args.epochs):
            args.epoch = epoch
            train_sampler.set_epoch(epoch)

            print(f'\n=== Epoch {epoch + 1}/{args.epochs} ===')

            train_loss, train_l3d, train_lbone, train_vel, train_d5, train_d10 = train_epoch(
                model, args, train_loader, optimizer, epoch, writer, local_rank)

            avg_val_loss = 0.0
            for act in acts:
                avg_val_loss += validate_epoch(model, args, val_loaders[act], local_rank)
            avg_val_loss /= len(acts)

            print(f'Train Loss: {train_loss:.6f}  (l3d={train_l3d:.6f}  lbone={train_lbone:.6f}  vel={train_vel:.6f}  d5={train_d5:.6f}  d10={train_d10:.6f})  |  Val MPJPE: {avg_val_loss * 1000:.2f} mm')

            if dist.get_rank() == 0:
                metrics = {
                    'Train/Loss':      train_loss,
                    'Train/MPJPE':     train_l3d,
                    'Train/BoneLoss':  train_lbone,
                    'Train/Velocity':  train_vel,
                    'Train/Delta5':    train_d5,
                    'Train/Delta10':   train_d10,
                    'Val/MPJPE_m':     avg_val_loss,
                    'Train/LR':        optimizer.param_groups[0]['lr'],
                    'epoch':           epoch,
                }

                for k, v in metrics.items():
                    writer.add_scalar(k, v, epoch)

                wandb.log(metrics, step=epoch)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    save_checkpoint(
                        model, optimizer, scheduler, epoch, avg_val_loss,
                        os.path.join(args.checkpoint_dir, args.checkpoint_save_name))
                    print(f"New best model saved: {best_val_loss * 1000:.2f} mm")

            if (epoch + 1) % args.save_freq == 0:
                save_checkpoint(
                    model, optimizer, scheduler, epoch, train_loss,
                    os.path.join(args.checkpoint_dir, f'phd_checkpoint_epoch_{epoch + 1}.pth'))

            scheduler.step()

        if rank == 0:
            writer.close()
            wandb.finish()
        dist.barrier()
        dist.destroy_process_group()
        print("Training completed!")

    else:  # test
        checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint_save_name)
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint  = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict  = checkpoint['model_state_dict']
        has_module  = any(k.startswith('module.') for k in state_dict)
        if has_module:
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.module.load_state_dict(state_dict)

        print("Starting inference...")
        all_mm = []
        for act in acts:
            print('-' * 50)
            print(f"{'Action' if args.dataset == 'h36m' else 'Video'}: {act}")
            all_mm.append(test_epoch(model, args, test_loaders[act], local_rank, act))

        if dist.get_rank() == 0:
            print('=' * 50)
            print(f"[{args.dataset}] mean MPJPE over {len(all_mm)} group(s): "
                  f"{np.mean(all_mm):.2f} mm")


        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()