import os
from pathlib import Path
import shutil

import torch
from sklearn.decomposition import IncrementalPCA

from src import util


class PCBuilder:

    def __init__(self, root, model_name, layer, dataset_name, ntau, nlat, nspan, scale, n_pca):
        self._root = root
        self._model_name = model_name
        self._layer = layer
        self._dataset_name = dataset_name
        self._ntau = ntau
        self._nlat = nlat
        self._nspan = nspan
        self._scale = scale
        self._n_pca = n_pca

        if not os.path.exists(f"{self._root}/data/x/{self.train_name}.pt"):
            self._model = util.get_model(root, model_name, ntau, nlat, nspan, layer)
            self._train_dataset = util.get_dataset(root, dataset_name, "train", ntau, 0, normalise=True, scale=scale)
            self._test_dataset = util.get_dataset(root, dataset_name, "test", ntau, 0, normalise=True, scale=scale)

            self._pca = None

    @property
    def train_name(self):
        return util.get_dataset_x_name(self._model_name, self._dataset_name, "train", self._ntau, self._nlat, self._nspan, self._scale, self._n_pca, self._layer)

    @property
    def test_name(self):
        return util.get_dataset_x_name(self._model_name, self._dataset_name, "test", self._ntau, self._nlat, self._nspan, self._scale, self._n_pca, self._layer)

    def fit_and_transform(self, batch=1000):
        if os.path.exists(f"{self._root}/data/x/{self.train_name}.pt"):
            print(f"Already built for {self.train_name}")
            return

        n_train_batches = self.generate_model_outputs(self._train_dataset, self.train_name, batch)
        n_test_batches = self.generate_model_outputs(self._test_dataset, self.test_name, batch)
        self.fit_pca(self.train_name, n_train_batches)
        train_pcs = self.get_pcs(self.train_name, n_train_batches)
        test_pcs = self.get_pcs(self.test_name, n_test_batches)
        torch.save(train_pcs, f"{self._root}/data/x/{self.train_name}.pt")
        torch.save(test_pcs, f"{self._root}/data/x/{self.test_name}.pt")

        shutil.rmtree(f"{self._root}/data/temp/{self.train_name}")
        shutil.rmtree(f"{self._root}/data/temp/{self.test_name}")

    def generate_model_outputs(self, dataset, name, batch):
        # Create intermediate directory for model activations
        Path(f"{self._root}/data/temp/{name}").mkdir(parents=True, exist_ok=False)

        batch_idx = 0
        n_batches = int(len(dataset) / batch)
        batch_activity_list = []

        for i, (x, _) in enumerate(dataset):
            with torch.no_grad():
                activity = self._model(x)
                if self._nlat == 0:
                    activity = activity[-self._nspan:]
                else:
                    activity = activity[-self._nspan: -self._nlat]
                activity = activity.flatten()
                batch_activity_list.append(activity)

            if len(batch_activity_list) == batch and batch_idx < n_batches - 1:
                activity_batch = torch.stack(batch_activity_list)
                torch.save(activity_batch, f"{self._root}/data/temp/{name}/{batch_idx}.pt")
                batch_idx += 1
                batch_activity_list = []

        if len(batch_activity_list) > 0:
            activity_batch = torch.stack(batch_activity_list)
            torch.save(activity_batch, f"{self._root}/data/temp/{name}/{batch_idx}.pt")

        return n_batches

    def fit_pca(self, name, n_batches):
        self._pca = IncrementalPCA(n_components=self._n_pca)

        for i in range(n_batches):
            activity_batch = torch.load(f"{self._root}/data/temp/{name}/{i}.pt")
            activity_batch = activity_batch.numpy()
            self._pca.partial_fit(activity_batch)
            print(f"Completed fitting pca {i}/{n_batches}...")

    def get_pcs(self, name, n_batches):
        pc_list = []

        for i in range(n_batches):
            activity_batch = torch.load(f"{self._root}/data/temp/{name}/{i}.pt")
            activity_batch = activity_batch.numpy()
            pcs = torch.from_numpy(self._pca.transform(activity_batch))
            pc_list.append(pcs)
            print(f"Completed building pca {i}/{n_batches}...")

        return torch.cat(pc_list)


class PCDataset(torch.utils.data.Dataset):

    def __init__(self, root, model, dataset, split, ntau, nlat, nspan, scale, n_pca, layer=None):
        x_name = util.get_dataset_x_name(model, dataset, split, ntau, nlat, nspan, scale, n_pca, layer)
        y_name = util.get_dataset_y_name(dataset, split, ntau)
        self._x = torch.load(f"{root}/data/x/{x_name}.pt")  # b x n (model neurons)
        self._y = torch.load(f"{root}/data/y/{y_name}.pt").permute(1, 0, 2)  # b x n (real neurons) x repeat
        assert len(self._x) == len(self._y), f"x {self._x.shape} y {self._y.shape}"

        # Normalise
        x_train_name = util.get_dataset_x_name(model, dataset, "train", ntau, nlat, nspan, scale, n_pca, layer)
        x_train = torch.load(f"{root}/data/x/{x_train_name}.pt")
        mean, std = x_train.mean(), x_train.std()
        self._x = (self._x - mean) / std

        self._max_ccs = util.get_dataset_max_ccs(dataset)

    def __getitem__(self, i):
        x, y = self._x[i], self._y[i]
        if len(y.shape) == 2:
            y = y.mean(1)

        return x, y

    def __len__(self):
        return len(self._x)

    @property
    def y(self):
        return self._y

    @property
    def x(self):
        return self._x

    @property
    def max_ccs(self):
        return self._max_ccs

    @property
    def hyperparams(self):
        return {}
