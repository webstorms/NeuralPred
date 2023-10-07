import sys

import torch

from src.datasets.base import BaseDataset


class CadenaDataset(BaseDataset):

    def __init__(self, root, split, ntau, normalise=True, scale=1):
        super().__init__(normalise, scale, False)
        # Load dependencies
        sys.path.append(f"{root}/dependencies/Cadena2019PlosCB")
        from cnn_sys_ident import data as cadena_data
        data_dict = cadena_data.Dataset.get_clean_data()
        data = cadena_data.MonkeyDataset(data_dict, seed=1000, train_frac=0.8, subsample=1, crop=14)

        if split == "train":
            x_train, y_train, _ = data.train()
            x_val, y_val, _ = data.val()
            x_train = torch.from_numpy(x_train)
            y_train = torch.from_numpy(y_train)
            x_val = torch.from_numpy(x_val)
            y_val = torch.from_numpy(y_val)
            self._x = torch.cat([x_train, x_val])
            self._y = torch.cat([y_train, y_val])
        elif split == "test":
            x_test, y_test, _ = data.test()
            self._x = torch.from_numpy(x_test)
            self._y = torch.from_numpy(y_test).permute(1, 0, 2)

        if split == "train":
            self._y = self._y.unsqueeze(1)

        self._x = self._x.permute(0, 3, 1, 2).repeat(1, ntau, 1, 1).unsqueeze(1)

    def __len__(self):
        return len(self._x)

    def _get_dimensions(self):
        return 112, 112

    def _get_mean_std(self):
        return -0.0009, 0.9993

    def _get_dataset_item(self, i):
        x, y = self._x[i], self._y[i]

        return x, y.permute(1, 0)
