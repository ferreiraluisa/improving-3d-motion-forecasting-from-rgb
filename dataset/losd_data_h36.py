import sys
import os
import numpy as np
import torch
import torch.utils.data as data

from dataset.camera import *
from dataset.generator import ChunkedGenerator
from dataset.utils import deterministic_random


class Fusion(data.Dataset):
    """
    H36M dataset returning, per window:
     cam, gt_3D, gt_2D, feat_window, action, subject, cam_ind, frame_idx
    """
    def __init__(self, opt, dataset, train=True, mode_t='val', action=''):
        self.action = action
        self.opt = opt
        opt.out_all = getattr(opt, 'out_all', True)
        self.mode = opt.mode
        self.mode_t = mode_t
        self.train = train

        self.train_list = opt.subjects_train.split(',')
        self.val_list = opt.subjects_val.split(',')
        self.test_list = opt.subjects_test.split(',')
        self.downsample = opt.downsample
        self.stride = opt.stride

        self.img_size = getattr(opt, 'img_size', [192, 256])
        self.seq_len = opt.seq_len
        self.action_dict = dict()
        self.action_dict["walking"] = ['Walking', 'Walking 1']
        self.action_dict["eating"] = ['Eating', 'Eating 1']
        self.action_dict["smoking"] = ['Smoking 1', 'Smoking']
        self.action_dict["discussion"] = ['Discussion 2', 'Discussion 1']
        self.action_dict["directions"] = ['Directions 1', 'Directions']
        self.action_dict["greeting"] = ['Greeting', 'Greeting 1']
        self.action_dict["phoning"] = ['Phoning 1', 'Phoning']
        self.action_dict["posing"] = ['Posing', 'Posing 1']
        self.action_dict["purchases"] = ['Purchases', 'Purchases 1']
        self.action_dict["sitting"] = ['Sitting', 'Sitting 1']
        self.action_dict["sittingdown"] = ['SittingDown', 'SittingDown 1']
        self.action_dict["waiting"] = ['Waiting 1', 'Waiting']
        self.action_dict["walkingtogether"] = ['WalkTogether', 'WalkTogether 1']
        self.action_dict["takingphoto"] = ['Photo 1', 'Photo']
        self.action_dict["walkingdog"] = ['WalkDog', 'WalkDog 1']

        self.action_dict_1 = self.action_dict.copy()
        self.action_dict_1["eating"] = ['Eating', 'Eating 2']
        self.action_dict_1["discussion"] = ['Discussion', 'Discussion 1']
        self.action_dict_1["sitting"] = ['Sitting 1', 'Sitting 2']
        self.action_dict_1["sittingdown"] = ['SittingDown', 'SittingDown 2']

        self.action_dict_11 = self.action_dict.copy()
        self.action_dict_11["smoking"] = ['Smoking']
        self.action_dict_11["directions"] = ['Directions 1']
        self.action_dict_11["greeting"] = ['Greeting']
        self.action_dict_11["phoning"] = ['Phoning 3']

        self.action_dict_7 = self.action_dict.copy()
        self.action_dict_7["walking"] = ['Walking 1', 'Walking 2']

        self.action_dict_5 = self.action_dict.copy()
        self.action_dict_5["discussion"] = ['Discussion 2', 'Discussion 3']
        self.action_dict_5["directions"] = ['Directions 1', 'Directions 2']
        self.action_dict_5["greeting"] = ['Greeting 1', 'Greeting 2']
        self.action_dict_5["waiting"] = ['Waiting 1', 'Waiting 2']
        self.action_dict_5["takingphoto"] = ['Photo', 'Photo 2']

        if not self.train:
            self.stride = opt.seq_len

        if self.train:
            subjects_list = self.train_list
        elif self.mode_t == 'val':
            subjects_list = self.val_list
        else:
            subjects_list = self.test_list

        self.keypoints_GT = self.prepare_data(self.opt, dataset, subjects_list)
        self.h_w, self.cameras, self.poses_3d, self.poses_2d_GT = self.fetch(dataset, subjects_list)

        self.generator = ChunkedGenerator(
            opt.batch_size, self.cameras, self.poses_3d,
            self.poses_2d_GT, opt.seq_len, self.stride,
            reverse_aug=opt.reverse_augmentation if self.train else False,
            out_all=opt.out_all)

        if self.train:
            print('Training on {} frames'.format(self.generator.num_frames()))
        else:
            self.key_index = self.generator.saved_index


    def get_action(self, dataset, subject):
        if not self.train:
            if subject == 'S11':
                return self.action_dict_11[self.action]
            if subject == 'S1':
                return self.action_dict_1[self.action]
            if subject == 'S5':
                return self.action_dict_5[self.action]
            if subject == 'S7':
                return self.action_dict_7[self.action]
            return self.action_dict[self.action]
        return dataset[subject].keys()

    def prepare_data(self, args, dataset, folder_list):
        # 3D GT in camera coords and root-relative
        for subject in folder_list:
            for action in self.get_action(dataset, subject):    
                try:
                    anim = dataset[subject][action]
                except KeyError as e:
                    print(f"KeyError: {e} | subject: {subject} | action: {action}")
                    print(f"Available keys for subject '{subject}': {list(dataset[subject].keys())}")
                    raise

                if 'positions' in anim:
                    positions_3d = []
                    for cam in anim['cameras']:
                        pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                        pos_3d -= pos_3d[:, :1]  # Remove global offset
                        positions_3d.append(pos_3d)
                    anim['positions_3d'] = positions_3d

        # 2D GT keypoints
        keypoints_GT = np.load(self.opt.dataset_2d_path, allow_pickle=True)['positions_2d'].item()

        # per frame features - dict[subject][action] = array(n_cams, N, feat_dim)
        self.feats = np.load(self.opt.features_path, allow_pickle=True).item()

        return keypoints_GT

    def fetch(self, dataset, subjects, parse_3d_poses=True):
        out_poses_3d = {}
        out_poses_2d_GT = {}
        out_camera_params = {}
        out_feats = {}
        
        h_w = {}

        for subject in subjects:
            for action in self.get_action(dataset, subject):
                if subject not in self.feats :
                    print(f"[SKIP] Subject {subject} not found in features")
                    continue

                if action not in self.feats[subject]:
                    print(f"[SKIP] {subject}/{action} not found in features")
                    continue

                poses_2d_GT = self.keypoints_GT[subject][action]

                # Number of cameras/features available
                num_cams = min(len(poses_2d_GT), len(self.feats[subject][action]))

                for i in range(num_cams):
                    key = (subject, action, i)
                    out_poses_2d_GT[key] = poses_2d_GT[i]
                    out_feats[key] = self.feats[subject][action][i]

                # Camera information
                if subject in dataset.cameras():
                    for i, cam in enumerate(dataset.cameras()[subject]):
                        key = (subject, action, i)
                        if key not in out_feats:
                            continue
                        if 'intrinsic' in cam:
                            out_camera_params[key] = cam['intrinsic']
                        if 'res_w' in cam:
                            h_w[key] = (cam['res_w'], cam['res_h'])


                # 3D poses
                if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                    poses_3d = dataset[subject][action]['positions_3d']

                    for i in range(min(len(poses_3d), len(poses_2d_GT))):
                        key = (subject, action, i)
                        if key not in out_feats:
                            continue

                        out_poses_3d[key] = poses_3d[i]

        if len(out_camera_params) == 0:
            out_camera_params = None

        if len(out_poses_3d) == 0:
            out_poses_3d = None

        if self.downsample > 1:
            for key in out_poses_2d_GT.keys():
                out_poses_2d_GT[key] = out_poses_2d_GT[key][::self.downsample]
                out_feats[key] = out_feats[key][::self.downsample]
                if out_poses_3d is not None and key in out_poses_3d:
                    out_poses_3d[key] = out_poses_3d[key][::self.downsample]
        

        self.feats = out_feats

        return h_w, out_camera_params, out_poses_3d, out_poses_2d_GT

    def __len__(self):
        return len(self.generator.pairs)

    def _window_features(self, seq_feat, start_2d,end_2d, reverse):
        low = max(start_2d, 0)
        high = min(end_2d, seq_feat.shape[0])
        pad_left = low - start_2d
        pad_right = end_2d - high

        window_feat = seq_feat[low:high]
        if pad_left != 0 or pad_right != 0:
            window_feat = np.pad(window_feat, ((pad_left, pad_right), (0, 0)), 'edge')
        if reverse:
            window_feat = window_feat[::-1].copy()

        return window_feat

    def __getitem__(self, index):
        seq_name = self.generator.pairs[index]
        start_3d = self.generator.bounds_1[index]
        reverse = self.generator.reverse_augment_vectors[index]

        cam, gt_3D, gt_2D, action, subject, cam_ind = \
            self.generator.get_batch(seq_name, start_3d, reverse)

        # -- features: window the same way as 2D (seq_len frames centred on start_3d) --
        seq_key = (subject, action, int(cam_ind))
        seq_feat = self.feats[seq_key] # (N, feat_dim)

        pad = self.generator.pad

        start_2d = start_3d - pad
        if self.seq_len % 2 == 0:
            end_2d = start_3d + pad  
        else:
            end_2d = start_3d + pad + 1

        feat_window = self._window_features(seq_feat, start_2d, end_2d, reverse)  

        # frame indices of the clip (in the DOWNSAMPLED sequence: real frame = idx * downsample)
        frame_idx = np.clip(np.arange(start_2d, end_2d), 0, seq_feat.shape[0] - 1)

        if reverse:
            frame_idx   = frame_idx[::-1].copy()

        gt_3D       = torch.from_numpy(gt_3D)
        gt_2D       = torch.from_numpy(gt_2D)

        feat_window = torch.from_numpy(np.ascontiguousarray(feat_window))
        frame_idx   = torch.from_numpy(frame_idx).long()

        return [cam, gt_3D, gt_2D, feat_window, action, subject, cam_ind, frame_idx]