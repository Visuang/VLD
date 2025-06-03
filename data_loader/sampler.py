import numpy as np
import copy
import random
from collections import defaultdict
from torch.utils.data.sampler import Sampler
from tools.utils import *

def GenIdx(train_color_label, train_thermal_label):
    color_pos = []
    unique_label_color = np.unique(train_color_label)
    for i in range(len(unique_label_color)):
        tmp_pos = [k for k, v in enumerate(train_color_label) if v == unique_label_color[i]]
        color_pos.append(tmp_pos)

    thermal_pos = []
    unique_label_thermal = np.unique(train_thermal_label)
    for i in range(len(unique_label_thermal)):
        tmp_pos = [k for k, v in enumerate(train_thermal_label) if v == unique_label_thermal[i]]
        thermal_pos.append(tmp_pos)

    return color_pos, thermal_pos

class IdentitySampler(Sampler):
    """Sample person identities evenly in each batch.
        Args:
            train_color_label, train_thermal_label: labels of two modalities
            color_pos, thermal_pos: positions of each identity
            batchSize: batch size
    """

    def __init__(self, train_color_label, train_thermal_label, color_pos, thermal_pos, num_pos, batchSize):
        uni_label = np.unique(train_color_label)
        self.n_classes = len(uni_label)

        N = np.maximum(len(train_color_label), len(train_thermal_label))
        for j in range(int(N / (batchSize * num_pos)) + 1):
            batch_idx = np.random.choice(uni_label, batchSize, replace=False)
            for i in range(batchSize):
                sample_color = np.random.choice(color_pos[batch_idx[i]], num_pos)
                sample_thermal = np.random.choice(thermal_pos[batch_idx[i]], num_pos)

                if j == 0 and i == 0:
                    index1 = sample_color
                    index2 = sample_thermal
                else:
                    index1 = np.hstack((index1, sample_color))
                    index2 = np.hstack((index2, sample_thermal))

        self.index1 = index1
        self.index2 = index2
        self.N = N

    def __iter__(self):
        return iter(np.arange(len(self.index1)))

    def __len__(self):
        return self.N

'''

class IdentitySampler(Sampler):

    def __init__(self, train_color_label, train_thermal_label, color_pos, thermal_pos, batchSize, per_img):
        uni_label = np.unique(train_color_label)
        self.n_classes = len(uni_label)

        sample_color = np.arange(batchSize)
        sample_thermal = np.arange(batchSize)
        N = np.maximum(len(train_color_label), len(train_thermal_label))

        # per_img = 4
        per_id = batchSize / per_img
        for j in range(N // batchSize + 1):
            batch_idx = np.random.choice(uni_label, int(per_id), replace=False)

            for s, i in enumerate(range(0, batchSize, per_img)):
                sample_color[i:i + per_img] = np.random.choice(color_pos[batch_idx[s]], per_img, replace=False)
                sample_thermal[i:i + per_img] = np.random.choice(thermal_pos[batch_idx[s]], per_img, replace=False)

            if j == 0:
                index1 = sample_color
                index2 = sample_thermal
            else:
                index1 = np.hstack((index1, sample_color))
                index2 = np.hstack((index2, sample_thermal))

        self.index1 = index1
        self.index2 = index2
        self.N = N

    def __iter__(self):
        return iter(np.arange(len(self.index1)))

    def __len__(self):
        return self.N
        '''

class RandomIdentitySampler(Sampler):
    """
    This sampler is for training.
    For each batch, randomly sample Np identities.
    For each identity, randomly sample Nc cameras.
    For each camera, randomly sample Nt tracklets.
    Note that each tracklet has two modalities, i.e., RGB and IR.
    So, the final batch size is equal to `Np * Nc * Nt * 2`
    """
    def __init__(self, dataset, np, nc, nt):
        self.np = np
        self.nc = nc
        self.nt = nt
        self.dataset = dataset

        # This line aims to get the self.length
        self.final_idx = self._get_final_idx()

    def _get_final_idx(self):
        # pid -> cam -> index
        index_dict = defaultdict(lambda: defaultdict(list))
        for index, (pid, modal, cam) in enumerate(self.dataset.fast_iteration()):
            index_dict[pid][cam].append(index)

        # pid -> batch_idx (e.g., [[0,1], [2,3]])
        pid2batch = defaultdict(list)
        for pid, cam2idx in index_dict.items():
            # store those cameras with enough tracklets
            available_cameras = [cam for cam in cam2idx
                                 if len(cam2idx[cam]) >= self.nt]
            while len(available_cameras) >= self.nc:
                batch_idx = []
                cameras = random.sample(available_cameras, self.nc)
                for camera in cameras:
                    sampled_index = random.sample(cam2idx[camera], self.nt)
                    batch_idx.extend(sampled_index)
                    cam2idx[camera] = [idx for idx in cam2idx[camera]
                                       if idx not in sampled_index]
                    if len(cam2idx[camera]) < self.nt:
                        available_cameras.remove(camera)
                pid2batch[pid].append(batch_idx)

        # generate final idx
        final_idx = []
        available_pids = copy.deepcopy(list(pid2batch))
        while len(available_pids) >= self.np:
            sampled_pids = random.sample(available_pids, self.np)
            for pid in sampled_pids:
                batch_idx = pid2batch[pid].pop(0)
                final_idx.extend(batch_idx)
                if len(pid2batch[pid]) == 0:
                    available_pids.remove(pid)

        self.length = len(final_idx)
        return final_idx

    def __iter__(self):
        """
        Call self._get_final_idx() in __iter__,
        to avoid the same sampling results in all epochs.
        """
        return iter(self._get_final_idx())

    def __len__(self):
        return self.length


class ConsistentModalitySampler(Sampler):
    """
    This sampler is for validation.
    It ensures the same modality in one batch.
    """
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

        # This line aims to get the self.length
        self.final_idx = self._get_final_idx()

    def _get_final_idx(self):
        # modality -> index
        index_dict = defaultdict(list)
        for index, (pid, modal, cam) in enumerate(self.dataset.fast_iteration()):
            index_dict[modal].append(index)

        # batch_idx (e.g., [[0,1], [2,3]])
        if self.batch_size > 1:
            idx_ir = index_dict[MODALITY['IR']]
            idx_rgb = index_dict[MODALITY['RGB']]
            dropped = len(idx_rgb) % self.batch_size
            idx_rgb = idx_rgb[:-dropped]  # Warning: This will drop some samples
            final_idx = idx_rgb + idx_ir

        self.length = len(final_idx)
        return final_idx

    def __iter__(self):
        """
        Call self._get_final_idx() in __iter__,
        to avoid the same sampling results in all epochs.
        """
        return iter(self._get_final_idx())

    def __len__(self):
        return self.length


class RandomCameraSampler(Sampler):
    """
    This sampler is for auxiliary training.
    For each batch, randomly sample Nc cameras.
    For each camera, randomly sample Np identities.
    For each identity, randomly sample Nt tracklets.
    Note that each tracklet has two modalities, i.e., RGB and IR.
    So, the final batch size is equal to `Nc * Np * Nt * 2`
    """
    def __init__(self, dataset, nc, np, nt):
        self.nc = nc
        self.np = np
        self.nt = nt
        self.dataset = dataset

        # This line aims to get the self.length
        self.final_idx = self._get_final_idx()

    def _get_final_idx(self):
        # cam -> pid -> index
        index_dict = defaultdict(lambda: defaultdict(list))
        for index, (pid, modal, cam) in enumerate(self.dataset.fast_iteration()):
            index_dict[cam][pid].append(index)

        # cam -> batch_idx (e.g., [[0,1],[2,3]])
        cam2batch = defaultdict(list)
        for cam, pid2idx in index_dict.items():
            # store those pids with enough tracklets
            available_pids = [pid for pid in pid2idx
                              if len(pid2idx[pid]) >= self.nt]
            while len(available_pids) >= self.np:
                batch_idx = []
                pids = random.sample(available_pids, self.np)
                for pid in pids:
                    sampled_index = random.sample(pid2idx[pid], self.nt)
                    batch_idx.extend(sampled_index)
                    pid2idx[pid] = [idx for idx in pid2idx[pid]
                                    if idx not in sampled_index]
                    if len(pid2idx[pid]) < self.nt:
                        available_pids.remove(pid)
                cam2batch[cam].append(batch_idx)

        # generate final idx
        final_idx = []
        available_cams = copy.deepcopy(list(cam2batch))
        while len(available_cams) >= self.nc:
            sampled_cams = random.sample(available_cams, self.nc)
            for cam in sampled_cams:
                batch_idx = cam2batch[cam].pop(0)
                final_idx.extend(batch_idx)
                if len(cam2batch[cam]) == 0:
                    available_cams.remove(cam)

        self.length = len(final_idx)
        return final_idx

    def __iter__(self):
        """
        Call self._get_final_idx() in __iter__,
        to avoid the same sampling results in all epochs.
        """
        return iter(self._get_final_idx())

    def __len__(self):
        return self.length