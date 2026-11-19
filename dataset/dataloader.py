# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import copy
# from datasets.dataloader import MocapDataset
from dataset.camera import normalize_screen_coordinates

# ----------------------------------------------------------------------------
# 17-joint skeleton definition (replaces the old Skeleton class)
# ----------------------------------------------------------------------------
# Indices of the 17 joints kept from the original 32-joint H3.6M skeleton.
# Equivalent to the old remove_joints([4, 5, 9, 10, 11, 16, 20, 21, 22, 23,
# 24, 28, 29, 30, 31]) followed by the shoulder rewiring.
KEPT_JOINTS = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]
 
# 17-joint 3D skeleton (post remove_joints + shoulder rewiring)
SKELETON_3D = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7],
               [7, 8], [8, 9], [9, 10], [8, 11], [11, 12], [12, 13],
               [8, 14], [14, 15], [15, 16]]
LEFT_3D  = {(0, 4), (4, 5), (5, 6), (8, 11), (11, 12), (12, 13)}
RIGHT_3D = {(0, 1), (1, 2), (2, 3), (8, 14), (14, 15), (15, 16)}
JOINTS_LEFT  = sorted({child for _, child in LEFT_3D})   # [4, 5, 6, 11, 12, 13]
JOINTS_RIGHT = sorted({child for _, child in RIGHT_3D})  # [1, 2, 3, 14, 15, 16]

h36m_cameras_intrinsic_params = [
    {
        'id': '54138969',
        'center': [512.54150390625, 515.4514770507812],
        'focal_length': [1145.0494384765625, 1143.7811279296875],
        'radial_distortion': [-0.20709891617298126, 0.24777518212795258, -0.0030751503072679043],
        'tangential_distortion': [-0.0009756988729350269, -0.00142447161488235],
        'res_w': 1000,
        'res_h': 1002,
        'azimuth': 70,  # Only used for visualization
    },
    {
        'id': '55011271',
        'center': [508.8486328125, 508.0649108886719],
        'focal_length': [1149.6756591796875, 1147.5916748046875],
        'radial_distortion': [-0.1942136287689209, 0.2404085397720337, 0.006819975562393665],
        'tangential_distortion': [-0.0016190266469493508, -0.0027408944442868233],
        'res_w': 1000,
        'res_h': 1000,
        'azimuth': -70,  # Only used for visualization
    },
    {
        'id': '58860488',
        'center': [519.8158569335938, 501.40264892578125],
        'focal_length': [1149.1407470703125, 1148.7989501953125],
        'radial_distortion': [-0.2083381861448288, 0.25548800826072693, -0.0024604974314570427],
        'tangential_distortion': [0.0014843869721516967, -0.0007599993259645998],
        'res_w': 1000,
        'res_h': 1000,
        'azimuth': 110,  # Only used for visualization
    },
    {
        'id': '60457274',
        'center': [514.9682006835938, 501.88201904296875],
        'focal_length': [1145.5113525390625, 1144.77392578125],
        'radial_distortion': [-0.198384091258049, 0.21832367777824402, -0.008947807364165783],
        'tangential_distortion': [-0.0005872055771760643, -0.0018133620033040643],
        'res_w': 1000,
        'res_h': 1002,
        'azimuth': -110,  # Only used for visualization
    },
]

h36m_cameras_extrinsic_params = {
    'S1': [
        {
            'orientation': [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088],
            'translation': [1841.1070556640625, 4955.28466796875, 1563.4454345703125],
        },
        {
            'orientation': [0.6157187819480896, -0.764836311340332, -0.14833825826644897, 0.11794740706682205],
            'translation': [1761.278564453125, -5078.0068359375, 1606.2650146484375],
        },
        {
            'orientation': [0.14651472866535187, -0.14647851884365082, 0.7653023600578308, -0.6094175577163696],
            'translation': [-1846.7777099609375, 5215.04638671875, 1491.972412109375],
        },
        {
            'orientation': [0.5834008455276489, -0.7853162288665771, 0.14548823237419128, -0.14749594032764435],
            'translation': [-1794.7896728515625, -3722.698974609375, 1574.8927001953125],
        },
    ],
    'S2': [
        {},
        {},
        {},
        {},
    ],
    'S3': [
        {},
        {},
        {},
        {},
    ],
    'S4': [
        {},
        {},
        {},
        {},
    ],
    'S5': [
        {
            'orientation': [0.1467377245426178, -0.162370964884758, -0.7551892995834351, 0.6178938746452332],
            'translation': [2097.3916015625, 4880.94482421875, 1605.732421875],
        },
        {
            'orientation': [0.6159758567810059, -0.7626792192459106, -0.15728192031383514, 0.1189815029501915],
            'translation': [2031.7008056640625, -5167.93310546875, 1612.923095703125],
        },
        {
            'orientation': [0.14291371405124664, -0.12907841801643372, 0.7678384780883789, -0.6110143065452576],
            'translation': [-1620.5948486328125, 5171.65869140625, 1496.43701171875],
        },
        {
            'orientation': [0.5920479893684387, -0.7814217805862427, 0.1274748593568802, -0.15036417543888092],
            'translation': [-1637.1737060546875, -3867.3173828125, 1547.033203125],
        },
    ],
    'S6': [
        {
            'orientation': [0.1337897777557373, -0.15692396461963654, -0.7571090459823608, 0.6198879480361938],
            'translation': [1935.4517822265625, 4950.24560546875, 1618.0838623046875],
        },
        {
            'orientation': [0.6147197484970093, -0.7628812789916992, -0.16174767911434174, 0.11819244921207428],
            'translation': [1969.803955078125, -5128.73876953125, 1632.77880859375],
        },
        {
            'orientation': [0.1529948115348816, -0.13529130816459656, 0.7646096348762512, -0.6112781167030334],
            'translation': [-1769.596435546875, 5185.361328125, 1476.993408203125],
        },
        {
            'orientation': [0.5916101336479187, -0.7804774045944214, 0.12832270562648773, -0.1561593860387802],
            'translation': [-1721.668701171875, -3884.13134765625, 1540.4879150390625],
        },
    ],
    'S7': [
        {
            'orientation': [0.1435241848230362, -0.1631336808204651, -0.7548328638076782, 0.6188824772834778],
            'translation': [1974.512939453125, 4926.3544921875, 1597.8326416015625],
        },
        {
            'orientation': [0.6141672730445862, -0.7638262510299683, -0.1596645563840866, 0.1177929937839508],
            'translation': [1937.0584716796875, -5119.7900390625, 1631.5665283203125],
        },
        {
            'orientation': [0.14550060033798218, -0.12874816358089447, 0.7660516500473022, -0.6127139329910278],
            'translation': [-1741.8111572265625, 5208.24951171875, 1464.8245849609375],
        },
        {
            'orientation': [0.5912848114967346, -0.7821764349937439, 0.12445473670959473, -0.15196487307548523],
            'translation': [-1734.7105712890625, -3832.42138671875, 1548.5830078125],
        },
    ],
    'S8': [
        {
            'orientation': [0.14110587537288666, -0.15589867532253265, -0.7561917304992676, 0.619644045829773],
            'translation': [2150.65185546875, 4896.1611328125, 1611.9046630859375],
        },
        {
            'orientation': [0.6169601678848267, -0.7647668123245239, -0.14846350252628326, 0.11158157885074615],
            'translation': [2219.965576171875, -5148.453125, 1613.0440673828125],
        },
        {
            'orientation': [0.1471444070339203, -0.13377119600772858, 0.7670128345489502, -0.6100369691848755],
            'translation': [-1571.2215576171875, 5137.0185546875, 1498.1761474609375],
        },
        {
            'orientation': [0.5927824378013611, -0.7825870513916016, 0.12147816270589828, -0.14631995558738708],
            'translation': [-1476.913330078125, -3896.7412109375, 1547.97216796875],
        },
    ],
    'S9': [
        {
            'orientation': [0.15540587902069092, -0.15548215806484222, -0.7532095313072205, 0.6199594736099243],
            'translation': [2044.45849609375, 4935.1171875, 1481.2275390625],
        },
        {
            'orientation': [0.618784487247467, -0.7634735107421875, -0.14132238924503326, 0.11933968216180801],
            'translation': [1990.959716796875, -5123.810546875, 1568.8048095703125],
        },
        {
            'orientation': [0.13357827067375183, -0.1367100477218628, 0.7689454555511475, -0.6100738644599915],
            'translation': [-1670.9921875, 5211.98583984375, 1528.387939453125],
        },
        {
            'orientation': [0.5879399180412292, -0.7823407053947449, 0.1427614390850067, -0.14794869720935822],
            'translation': [-1696.04345703125, -3827.099853515625, 1591.4127197265625],
        },
    ],
    'S11': [
        {
            'orientation': [0.15232472121715546, -0.15442320704460144, -0.7547563314437866, 0.6191070079803467],
            'translation': [2098.440185546875, 4926.5546875, 1500.278564453125],
        },
        {
            'orientation': [0.6189449429512024, -0.7600917220115662, -0.15300633013248444, 0.1255258321762085],
            'translation': [2083.182373046875, -4912.1728515625, 1561.07861328125],
        },
        {
            'orientation': [0.14943228662014008, -0.15650227665901184, 0.7681233882904053, -0.6026304364204407],
            'translation': [-1609.8153076171875, 5177.3359375, 1537.896728515625],
        },
        {
            'orientation': [0.5894251465797424, -0.7818877100944519, 0.13991211354732513, -0.14715361595153809],
            'translation': [-1590.738037109375, -3854.1689453125, 1578.017578125],
        },
    ],
}


class MocapDataset:
    def __init__(self, fps):
        self._fps = fps
        self._data = None  # Must be filled by subclass
        self._cameras = None  # Must be filled by subclass

    def remove_joints(self, kept_joints=KEPT_JOINTS):
        """Keep only the given joint indices along the joint axis of every sequence."""
        for subject in self._data.keys():
            for action in self._data[subject].keys():
                s = self._data[subject][action]
                if 'positions' in s:
                    s['positions'] = s['positions'][:, kept_joints]

    def __getitem__(self, key):
        return self._data[key]

    def subjects(self):
        return self._data.keys()

    def fps(self):
        return self._fps

    def skeleton(self):
        return self._skeleton

    def cameras(self):
        return self._cameras

    def supports_semi_supervised(self):
        # This method can be overridden
        return False


class Human36mDataset(MocapDataset):
    def __init__(self, path, remove_static_joints=True):
        super().__init__(fps=50)

        self._cameras = copy.deepcopy(h36m_cameras_extrinsic_params)
        for cameras in self._cameras.values():
            for i, cam in enumerate(cameras):
                cam.update(h36m_cameras_intrinsic_params[i])
                for k, v in cam.items():
                    if k not in ['id', 'res_w', 'res_h']:
                        cam[k] = np.array(v, dtype='float32')

                # Normalize camera frame
                cam['center'] = normalize_screen_coordinates(cam['center'], w=cam['res_w'], h=cam['res_h']).astype(
                    'float32')
                cam['focal_length'] = cam['focal_length'] / cam['res_w'] * 2
                if 'translation' in cam:
                    cam['translation'] = cam['translation'] / 1000  # mm to meters

                # Add intrinsic parameters vector
                cam['intrinsic'] = np.concatenate((cam['focal_length'],
                                                   cam['center'],
                                                   cam['radial_distortion'],
                                                   cam['tangential_distortion']))

        # Load serialized dataset
        data = np.load(path, allow_pickle=True)['positions_3d'].item()

        self._data = {}
        for subject, actions in data.items():
            self._data[subject] = {}
            for action_name, positions in actions.items():
                self._data[subject][action_name] = {
                    'positions': positions,
                    'cameras': self._cameras[subject],
                }

        if remove_static_joints:
            self.remove_joints(kept_joints=KEPT_JOINTS)

    def joints_left(self):
        return JOINTS_LEFT

    def joints_right(self):
        return JOINTS_RIGHT

    def supports_semi_supervised(self):
        return True

# ----------------------------------------------------------------------------
# 3DPW
# ----------------------------------------------------------------------------
# SMPL-24 joint order used by feats_3dpw.py (meta 'joints2d' / 'joints_cam'):
# 0 pelvis, 1 l_hip, 2 r_hip, 3 spine1, 4 l_knee, 5 r_knee, 6 spine2, 7 l_ankle,
# 8 r_ankle, 9 spine3, 10 l_foot, 11 r_foot, 12 neck, 13 l_collar, 14 r_collar,
# 15 head, 16 l_shoulder, 17 r_shoulder, 18 l_elbow, 19 r_elbow, 20 l_wrist,
# 21 r_wrist, 22 l_hand, 23 r_hand
SMPL24_JOINTS_LEFT  = [1, 4, 7, 10, 13, 16, 18, 20, 22]
SMPL24_JOINTS_RIGHT = [2, 5, 8, 11, 14, 17, 19, 21, 23]
 
# APPROXIMATE SMPL-24 -> H36M-17 (same order/skeleton as SKELETON_3D above).
# SMPL joints are not H36M joints: spine/thorax/neck sit at slightly different
# heights and SMPL has no head top, which is extrapolated from neck -> head.
# Good enough to feed a model trained on H36M-17; for paper-comparable 3DPW
# numbers use the SMPL vertices + J_regressor_h36m, or evaluate on 'smpl24'.
#   H36M idx:       0  1  2  3  4  5  6  7  8   9  10  11  12  13  14  15  16
SMPL_TO_H36M17 = [0, 2, 5, 8, 1, 4, 7, 6, 12, 15, -1, 16, 18, 20, 17, 19, 21]
H36M_HEAD_TOP_EXTRAPOLATION = 0.5   # head_top = head + 0.5 * (head - neck)
 
 
def smpl24_to_h36m17(j):
    """(..., 24, C) SMPL joints -> (..., 17, C) approximate H36M joints."""
    j = np.asarray(j)
    idx = [i if i >= 0 else 15 for i in SMPL_TO_H36M17]
    out = j[..., idx, :].copy()
    head, neck = j[..., 15, :], j[..., 12, :]
    out[..., 10, :] = head + H36M_HEAD_TOP_EXTRAPOLATION * (head - neck)
    return out

class ThreeDPWDataset(MocapDataset):
    """3DPW ground truth for the segments produced by feats_3dpw.py.
 
    Built from <out>_meta.npy, so it has exactly the same segment keys as the
    feature file:
 
        self._data[split][segment] = {
            'source_seq':    str        3DPW video name (imageFiles/<source_seq>/)
            'frame_start':   int        first image index (inclusive)
            'frame_end':     int        last image index (inclusive)
            'actor_ids':     [A]        original 3DPW actor of list entry k
            'img_frame_ids': (n,)       image index of every row -> image_%05d.jpg
            'positions_3d':  [A x (n, J, 3)]  joints in CAMERA coords (m), absolute
            'positions_2d':  [A x (n, J, 2)]  projected joints, PIXELS
            'box':           [A x (n, 3)]     crop (cx, cy, side), pixels
            'valid':         [A x (n,)]       campose_valid
            'camera':        dict (intrinsic vector like H36M, K, res_w, res_h)
        }
 
    Row r of a segment (features, 2D, 3D) is image img_frame_ids[r] of
    source_seq. The camera moves in 3DPW, so 3D is already given per frame in
    camera coordinates; there is no world->camera step as in H36M.
 
    joints: 'smpl24' (default) or 'h36m17' (approximate mapping, see above).
    """
 
    def __init__(self, meta_path, joints='smpl24', splits=None):
        super().__init__(fps=30)
        if joints not in ('smpl24', 'h36m17'):
            raise ValueError(f"joints must be 'smpl24' or 'h36m17', got {joints}")
        self.joints = joints
        meta = np.load(meta_path, allow_pickle=True).item()
 
        self._data, self._cameras = {}, {}
        for split, segments in meta.items():
            if splits is not None and split not in splits:
                continue
            self._data[split], self._cameras[split] = {}, {}
            for seg, m in segments.items():
                cam = self._make_camera(m['K'], m['image_hw'])
                j3d = [np.asarray(x, np.float32) for x in m['joints_cam']]
                j2d = [np.asarray(x, np.float32) for x in m['joints2d']]
                if joints == 'h36m17':
                    j3d = [smpl24_to_h36m17(x).astype(np.float32) for x in j3d]
                    j2d = [smpl24_to_h36m17(x).astype(np.float32) for x in j2d]
                n = len(m['img_frame_ids'])
                for x in j3d + j2d:
                    assert x.shape[0] == n, f"{split}/{seg}: {x.shape[0]} rows vs {n} frames"
                self._cameras[split][seg] = cam
                self._data[split][seg] = {
                    'source_seq': m['source_seq'],
                    'frame_start': int(m['frame_start']),
                    'frame_end': int(m['frame_end']),
                    'actor_ids': list(m['actor_ids']),
                    'img_frame_ids': np.asarray(m['img_frame_ids'], np.int64),
                    'positions_3d': j3d,
                    'positions_2d': j2d,
                    'box': [np.asarray(b, np.float32) for b in m['box']],
                    'valid': [np.asarray(v, bool) for v in m['valid']],
                    'camera': cam,
                }
 
    @staticmethod
    def _make_camera(K, image_hw):
        K = np.asarray(K, np.float32)
        H, W = int(image_hw[0]), int(image_hw[1])
        center = normalize_screen_coordinates(np.array([K[0, 2], K[1, 2]], np.float32),
                                              w=W, h=H).astype('float32')
        focal = (np.array([K[0, 0], K[1, 1]], np.float32) / W * 2).astype('float32')
        # 3DPW images are undistorted: zero radial (3) / tangential (2) terms,
        # so the 9-dim vector has the same layout as the H36M one.
        intrinsic = np.concatenate((focal, center, np.zeros(3, np.float32),
                                    np.zeros(2, np.float32))).astype('float32')
        return {'K': K, 'res_w': W, 'res_h': H, 'center': center,
                'focal_length': focal, 'intrinsic': intrinsic}
 
    def remove_joints(self, kept_joints=None):
        raise NotImplementedError("3DPW joints are chosen with joints='smpl24'|'h36m17'")
 
    def splits(self):
        return list(self._data.keys())
 
    def joints_left(self):
        return JOINTS_LEFT if self.joints == 'h36m17' else SMPL24_JOINTS_LEFT
 
    def joints_right(self):
        return JOINTS_RIGHT if self.joints == 'h36m17' else SMPL24_JOINTS_RIGHT
 
    def skeleton(self):
        return SKELETON_3D if self.joints == 'h36m17' else None