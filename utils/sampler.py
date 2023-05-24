import math

import numpy as np
from paddle.io import BatchSampler, DistributedBatchSampler

__all__ = ["CustomBatchSampler", "CustomDistributedBatchSampler"]


def _batch_shuffle(indices, batch_size, epoch):
    """将大小相似的实例放入小批量中可以提高效率，并进行批量打乱

    1. 按持续时间对音频剪辑进行排序
    2. 生成一个随机数k， k的范围[0,batch_size)
    3. 随机移动k实例，为不同的epoch训练创建不同的批次
    4. 打乱minibatches.

    :param batch_size: 批量大小。这个大小还用于为批量洗牌生成一个随机数。
    :type batch_size: int
    :param epoch: 当前的轮数。
    :type epoch: int
    :return: Batch shuffled indices.
    :rtype: list
    """
    rng = np.random.RandomState(epoch)
    shift_len = rng.randint(0, batch_size - 1)
    batch_indices = list(zip(*[iter(indices[shift_len:])] * batch_size))
    rng.shuffle(batch_indices)
    batch_indices = [item for batch in batch_indices for item in batch]
    res_len = len(indices) - shift_len - len(batch_indices)
    if res_len != 0:
        batch_indices.extend(indices[-res_len:])
    batch_indices.extend(indices[0:shift_len])
    return batch_indices


class CustomBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size, shuffle=False, drop_last=False):
        """Sampler for one gpu.

        Args:
            dataset (paddle.io.Dataset): 
            batch_size (int): batch size for one gpu
            shuffle (bool, optional): True for do shuffle, or else. Defaults to False.
            drop_last (bool, optional): whether drop last batch which is less than batch size. Defaults to False.
        """
        self.dataset = dataset

        assert isinstance(batch_size, int) and batch_size > 0, "batch_size should be a positive integer"
        self.batch_size = batch_size
        assert isinstance(shuffle, bool), "shuffle should be a boolean value"
        self.shuffle = shuffle
        assert isinstance(drop_last, bool), "drop_last should be a boolean number"
        self.epoch = 0
        self.drop_last = drop_last
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0))
        self.total_size = self.num_samples

    def __iter__(self):
        num_samples = len(self.dataset)
        indices = np.arange(num_samples).tolist()
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # sort (by duration) or batch-wise shuffle the manifest
        if self.shuffle:
            indices = _batch_shuffle(indices, self.batch_size, self.epoch)
        assert len(indices) == self.total_size, f"batch shuffle examples error: {len(indices)} : {self.total_size}"
        assert len(indices) == self.num_samples
        _sample_iter = iter(indices)

        batch_indices = []
        for idx in _sample_iter:
            batch_indices.append(idx)
            if len(batch_indices) == self.batch_size:
                yield batch_indices
                batch_indices = []
        if not self.drop_last and len(batch_indices) > 0:
            yield batch_indices

        self.epoch += 1

    def __len__(self):
        num_samples = self.num_samples
        num_samples += int(not self.drop_last) * (self.batch_size - 1)
        return num_samples // self.batch_size


class CustomDistributedBatchSampler(DistributedBatchSampler):
    def __init__(self,
                 dataset,
                 batch_size,
                 num_replicas=None,
                 rank=None,
                 shuffle=False,
                 drop_last=False):
        """Sortagrad Sampler for multi gpus.

        Args:
            dataset (paddle.io.Dataset):
            batch_size (int): batch size for one gpu
            num_replicas (int, optional): world size or numbers of gpus. Defaults to None.
            rank (int, optional): rank id. Defaults to None.
            shuffle (bool, optional): True for do shuffle, or else. Defaults to False.
            drop_last (bool, optional): whether drop last batch which is less than batch size. Defaults to False.
        """
        super().__init__(dataset=dataset, batch_size=batch_size, num_replicas=num_replicas, rank=rank, shuffle=shuffle,
                         drop_last=drop_last)

    def __iter__(self):
        num_samples = len(self.dataset)
        indices = np.arange(num_samples).tolist()
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size
        if self.shuffle:
            indices = _batch_shuffle(indices, self.batch_size * self.nranks, self.epoch)
        assert len(indices) == self.total_size, f"batch shuffle examples error: {len(indices)} : {self.total_size}"

        # slice `self.batch_size` examples by rank id
        def _get_indices_by_batch_size(indices):
            subsampled_indices = []
            last_batch_size = self.total_size % (self.batch_size * self.nranks)
            assert last_batch_size % self.nranks == 0
            last_local_batch_size = last_batch_size // self.nranks

            for i in range(self.local_rank * self.batch_size, len(indices) - last_batch_size,
                           self.batch_size * self.nranks):
                subsampled_indices.extend(indices[i:i + self.batch_size])

            indices = indices[len(indices) - last_batch_size:]
            subsampled_indices.extend(
                indices[self.local_rank * last_local_batch_size:(self.local_rank + 1) * last_local_batch_size])
            return subsampled_indices

        if self.nranks > 1:
            indices = _get_indices_by_batch_size(indices)

        assert len(indices) == self.num_samples
        _sample_iter = iter(indices)

        batch_indices = []
        for idx in _sample_iter:
            batch_indices.append(idx)
            if len(batch_indices) == self.batch_size:
                yield batch_indices
                batch_indices = []
        if not self.drop_last and len(batch_indices) > 0:
            yield batch_indices

        self.epoch += 1

    def __len__(self):
        num_samples = self.num_samples
        num_samples += int(not self.drop_last) * (self.batch_size - 1)
        return num_samples // self.batch_size
