from src.datasets.base import BaseDataset
from src.datasets.mineault.pvc1 import PVC1


class PVC1Dataset(BaseDataset):

    def __init__(self, root, split, ntau, idx, repeats=True, normalise=True, scale=1):
        super().__init__(normalise, scale, True)
        split = "traintune" if split == "train" else "report"
        path = f"{root}/data/processed/crcns-pvc1"
        self._pvc1 = PVC1(path,
                          split=split,
                          nt=1,
                          nx=112,
                          ny=112,
                          ntau=ntau,
                          nframedelay=0,
                          repeats=repeats,
                          single_cell=idx,
                          nframestart=ntau)

    def __len__(self):
        return len(self._pvc1)

    def _get_dimensions(self):
        return 112, 112

    def _get_mean_std(self):
        return 0.08094, 1.03749

    def _get_dataset_item(self, i):
        x, _, y, _ = self._pvc1[i]
        return x, y


class SinglePVC1Dataset(PVC1Dataset):

    def __init__(self, root, split, ntau, normalise=True, scale=1):
        super().__init__(root, split, ntau, -1, repeats=False, normalise=normalise, scale=scale)

    def _get_dataset_item(self, i):
        x, _, _, y = self._pvc1[i]
        return x, y[:, 0]


class MultiPVC1Dataset(PVC1Dataset):

    def __init__(self, root, split, ntau, idx, normalise=True, scale=1):
        super().__init__(root, split, ntau, idx, repeats=True, normalise=normalise, scale=scale)

    def _get_dataset_item(self, i):
        x, _, y, _ = self._pvc1[i]
        return x, y[0]
