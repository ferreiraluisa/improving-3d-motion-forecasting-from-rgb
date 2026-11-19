import numpy as np


class ChunkedGenerator:
    """
    Splits every (subject, action, cam) sequence into windows of seq_len frames.

    get_batch returns: cam, gt_3D, gt_2D, action, subject, cam_index
    """
    def __init__(self, batch_size, cameras, poses_3d, poses_2d_GT, seq_len,
                 stride=1, causal_shift=0,
                 shuffle=False, random_seed=1234,
                 reverse_aug=False, endless=False, out_all=False):
        assert poses_3d is None or len(poses_3d) == len(poses_2d_GT), (len(poses_3d), len(poses_2d_GT))
        assert cameras is None or len(cameras) == len(poses_2d_GT)

        pairs, bounds_1, reverse_augment_vectors = [], [], []
        self.seq_len = seq_len
        self.saved_index = {}
        start_index = 0

        for key in poses_2d_GT.keys():
            n_frames = poses_2d_GT[key].shape[0]
            assert poses_3d is None or poses_3d[key].shape[0] == n_frames, \
                (key, poses_3d[key].shape[0], n_frames)
            n_chunks = (n_frames+ stride - 1) // stride
            offset = (n_chunks * stride - n_frames) // 2
            bounds = np.arange(n_chunks + 1) * stride - offset
            no_reverse = np.full(len(bounds) - 1, False, dtype=bool)

            keys = np.tile(np.array(key).reshape([1, 3]), (len(bounds) - 1, 1))

            pairs += list(keys)
            bounds_1 += list(bounds[:-1])
            reverse_augment_vectors += list(no_reverse)

            if reverse_aug:
                pairs += list(keys)
                bounds_1 += list(bounds[:-1])
                reverse_augment_vectors += list(~no_reverse)

            end_index = start_index + n_frames
            self.saved_index[key] = [start_index, end_index]
            start_index = end_index

        self.pairs = np.array(pairs)
        self.bounds_1 = np.array(bounds_1)
        self.reverse_augment_vectors = np.array(reverse_augment_vectors)

        self.num_batches = (len(pairs) + batch_size - 1) // batch_size
        self.batch_size = batch_size
        self.random = np.random.RandomState(random_seed)

        self.shuffle = shuffle
        self.pad = self.seq_len // 2
        self.causal_shift = causal_shift
        self.endless = endless
        self.state = None

        self.cameras = cameras
        self.poses_3d = poses_3d
        self.poses_2d_GT = poses_2d_GT

        self.reverse_aug = reverse_aug
        self.out_all = out_all

    def num_frames(self):
        return self.num_batches * self.batch_size

    def random_state(self):
        return self.random

    def set_random_state(self, random):
        self.random = random


    def next_pairs(self):
        if self.state is None:
           pairs = self.random.permutation(self.pairs) if self.shuffle else self.pairs
           return 0, pairs
        return self.state
    
    def get_batch(self, seq_i, start_3d, reverse):
        subject, action, cam_index = seq_i
        seq_name = (subject, action, int(cam_index))
        end_3d = start_3d + 1

        start_2d = start_3d - self.pad - self.causal_shift
        if self.seq_len % 2 ==0 :
            end_2d = end_3d + self.pad - self.causal_shift -1
        else:
            end_2d = end_3d + self.pad - self.causal_shift

        # -- 2D GT --
        seq_2d_GT = self.poses_2d_GT[seq_name]

        low_2d = max(start_2d, 0)
        high_2d = min(end_2d, seq_2d_GT.shape[0])
        pad_left_2d = low_2d - start_2d
        pad_right_2d = end_2d - high_2d

        batch_2d_GT = seq_2d_GT[low_2d:high_2d]

        if pad_left_2d != 0 or pad_right_2d != 0:
            batch_2d_GT = np.pad(batch_2d_GT, ((pad_left_2d, pad_right_2d), (0, 0), (0, 0)), 'edge')
        if reverse:
            batch_2d_GT = batch_2d_GT[::-1]

        # -- 3D GT --
        batch_3d = None
        if self.poses_3d is not None:
            seq_3d = self.poses_3d[seq_name]
            if self.out_all:
                low_3d, high_3d = low_2d, high_2d
                pad_left_3d, pad_right_3d = pad_left_2d, pad_right_2d
            else:
                low_3d = max(start_3d, 0)
                high_3d = min(end_3d, seq_3d.shape[0])
                pad_left_3d = low_3d - start_3d
                pad_right_3d = end_3d - high_3d
            batch_3d = seq_3d[low_3d:high_3d]

            if pad_left_3d != 0 or pad_right_3d != 0:
                batch_3d = np.pad(batch_3d, ((pad_left_3d, pad_right_3d), (0, 0), (0, 0)), 'edge')
            if reverse:
                batch_3d = batch_3d[::-1]
            batch_3d = batch_3d.copy()

        # -- camera -- 
        if self.cameras is not None:
            batch_cam = self.cameras[seq_name].copy()
        else:
            batch_cam = np.zeros(9) if self.poses_3d is not None else np.zeros(3)

        return batch_cam, batch_3d, batch_2d_GT.copy(), action, subject, int(cam_index)