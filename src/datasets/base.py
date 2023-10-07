import torch
import numpy as np
from src.datasets.transforms import ClipGrayscale, ClipNormalize
from torchvision.transforms import Resize, InterpolationMode


class BaseDataset(torch.utils.data.Dataset):

    def __init__(self, normalise=True, scale=1, grayscale=True):
        h, w = self._get_dimensions()
        mean, std = self._get_mean_std()
        self.mean = mean
        self.std = std

        self._grayscale_trans = ClipGrayscale() if grayscale else None
        self._normalise_trans = ClipNormalize(mean=[mean], std=[std]) if normalise else None
        scaled_h, scaled_w = int(scale * h), int(scale * w)
        self._scale_trans = Resize((scaled_h, scaled_w), interpolation=InterpolationMode.BICUBIC) if scale != 1 else None

    def __getitem__(self, i):
        x, y = self._get_dataset_item(i)
        x = torch.from_numpy(x) if type(x) in [np.array, np.ndarray] else x
        y = torch.from_numpy(y) if type(y) in [np.array, np.ndarray] else y

        if self._grayscale_trans is not None:
            x = self._grayscale_trans(x)

        if self._normalise_trans is not None:
            x = self._normalise_trans(x)

        if self._scale_trans is not None:
            x = self._scale_trans(x)

        return x, y

    @property
    def y(self):
        return self._y

    def _get_dimensions(self):
        raise NotImplementedError

    def _get_mean_std(self):
        raise NotImplementedError

    def _get_dataset_item(self, i):
        raise NotImplementedError
