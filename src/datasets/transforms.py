# All these transforms are taken from brainbox but copy-pasted because prednet requires older packages not
# compatible with brainbox :(

import torch


class BBTransform:
    @property
    def hyperparams(self):
        hyperparams = {"name": self.__class__.__name__}

        return hyperparams


class ClipGrayscale(BBTransform):
    def __call__(self, clip):
        r, g, b = clip.unbind(dim=0)
        clip = (0.2989 * r + 0.587 * g + 0.114 * b).unsqueeze(dim=0)

        return clip


class ClipNormalize(BBTransform):
    def __init__(self, mean, std, inplace=False):
        self._mean = mean
        self._std = std
        self._inplace = inplace

    def __call__(self, clip):

        if not self._inplace:
            clip = clip.clone()

        dtype = clip.dtype
        mean = torch.as_tensor(self._mean, dtype=dtype, device=clip.device)
        std = torch.as_tensor(self._std, dtype=dtype, device=clip.device)
        clip.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])

        return clip

    @property
    def hyperparams(self):
        hyperparams = {**super().hyperparams, "mean": self._mean, "std": self._std}

        return hyperparams