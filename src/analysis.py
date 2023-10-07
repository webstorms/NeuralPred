import torch
import pandas as pd
from brainbox.physiology import neural

from src import datasets, train, util


class PVC1ModelQuery:
    static_model_query = {
        "vgg": {"3.1": [0.66, 1.0, 1.5]},
        "bwt": {"None": [0.66, 1.0, 1.5]}
    }
    dynamic_model_query = {
        "stack_prediction_4-6-2_True":
            {"1": [0.66, 1.0, 1.5],
             "2": [0.66, 1.0, 1.5],
             "3": [0.66, 1.0, 1.5],
             "4": [0.66, 1.0, 1.5]
             },
        "stack_compression_4-6-8_True":
            {"1": [0.66, 1.0, 1.5],
             "2": [0.66, 1.0, 1.5],
             "3": [0.66, 1.0, 1.5],
             "4": [0.66, 1.0, 1.5]
             },
        "stack_slowness_4-6_True":
            {"1": [0.66, 1.0, 1.5],
             "2": [0.66, 1.0, 1.5],
             "3": [0.66, 1.0, 1.5],
             "4": [0.66, 1.0, 1.5]
             },
        "prednet":
            {"E0": [0.66, 1.0, 1.5],
             "E1": [0.66, 1.0],
             "E2": [0.66, 1.0, 1.5],
             "E3": [0.66, 1.0, 1.5]
             }
    }

    def __init__(self, root, nspan=3):
        all_model_query = {**PVC1ModelQuery.static_model_query, **PVC1ModelQuery.dynamic_model_query}
        self._pvc1_metrics_builder = DatasetMetricsBuilder(root, all_model_query, "multi_pvc1", nspan=nspan)
        self._pvc1_metrics_builder.build()
        self._metrics_df = self._pvc1_metrics_builder.metrics_df.set_index(["model", "layer", "scale"])

    def get_metrics_df(self):
        return self._metrics_df

    def get_best_df(self):
        model_query = self.get_best_df_idx()
        metrics_df = self.get_metrics_df().loc[model_query].reset_index()

        return metrics_df

    def get_best_df_idx(self):
        return self.get_metrics_df().groupby(["model", "layer", "scale"]).mean().groupby(["model"]).idxmax()["cc_norm"]


class CadenaModelQuery:
    static_model_query = {
        "vgg": {"3.1": [0.66, 1.0, 1.5]},
        "bwt": {"None": [0.66, 1.0, 1.5]}
    }
    dynamic_model_query = {
        "stack_prediction_4-6-2_True":
            {"1": [0.66, 1.0, 1.5],
             "2": [0.66, 1.0, 1.5],
             "3": [0.66, 1.0, 1.5],
             "4": [0.66, 1.0, 1.5]
             },
        "stack_compression_4-6-8_True":
            {"1": [0.66, 1.0, 1.5],
             "2": [0.66, 1.0, 1.5],
             "3": [0.66, 1.0, 1.5],
             "4": [0.66, 1.0, 1.5]
             },
        "stack_slowness_4-6_True":
            {"1": [0.66, 1.0, 1.5],
             "2": [0.66, 1.0, 1.5],
             "3": [0.66, 1.0, 1.5],
             "4": [0.66, 1.0, 1.5]
             },
        "prednet":
            {"E0": [0.66, 1.0, 1.5],
             "E1": [0.66, 1.0],
             "E2": [0.66, 1.0],
             "E3": [0.66, 1.0, 1.5]
             }
    }

    def __init__(self, root, nspan=3):
        self._static_cadena_metrics_builder = DatasetMetricsBuilder(root, CadenaModelQuery.static_model_query, "cadena", nlat=0, nspan=1)
        self._static_cadena_metrics_builder.build()
        self._dynamic_cadena_metrics_builder = DatasetMetricsBuilder(root, CadenaModelQuery.dynamic_model_query, "cadena", nspan=nspan)
        self._dynamic_cadena_metrics_builder.build()
        self._metrics_df = pd.concat([self._static_cadena_metrics_builder.metrics_df.set_index(["model", "layer", "scale"]), self._dynamic_cadena_metrics_builder.metrics_df.set_index(["model", "layer", "scale"])])

    def get_metrics_df(self):
        return self._metrics_df

    def get_best_df(self):
        model_query = self.get_best_df_idx()
        metrics_df = self.get_metrics_df().loc[model_query].reset_index()

        return metrics_df

    def get_best_df_idx(self):
        return self.get_metrics_df().groupby(["model", "layer", "scale"]).mean().groupby(["model"]).idxmax()["cc_norm"]


class DatasetMetricsBuilder:

    # Build results for a given dictionary of models (i.e. model_query) on a given dataset

    def __init__(self, root, model_query, dataset, ntau=25, nlat=1, nspan=6, n_pca=500):
        self._root = root
        self._model_query = model_query
        self._dataset = dataset
        self._ntau = ntau
        self._nlat = nlat
        self._nspan = nspan
        self._n_pca = n_pca

        self._metrics = {}

    @property
    def metrics(self):
        return self._metrics

    @property
    def metrics_df(self):
        model_list = []
        layer_list = []
        scale_list = []
        cc_list = []
        cc_norm_list = []
        cc_spe_list = []

        for key in self._metrics.keys():
            model, layer, scale = key
            cc = self._metrics[key]["cc"]
            cc_norm = self._metrics[key]["cc_norm"]
            cc_spe = self._metrics[key]["cc_spe"]
            n = len(cc_spe)

            model_list.extend(n * [model])
            layer_list.extend(n * [layer])
            scale_list.extend(n * [scale])
            cc_list.extend(cc.numpy())
            cc_norm_list.extend(cc_norm.numpy())
            cc_spe_list.extend(cc_spe.numpy())

        return pd.DataFrame({"model": model_list, "layer": layer_list, "scale": scale_list, "cc": cc_list, "cc_norm": cc_norm_list, "cc_spe": cc_spe_list})

    def get_summary(self, contract_model_layer=False):
        multi_index_list = []
        cc_list = []
        cc_norm_list = []
        cc_spe_list = []

        for model in self._model_query.keys():
            for layer in self._model_query[model].keys():
                for scale in self._model_query[model][layer]:
                    scale = str(scale)
                    multi_index_list.append([model, layer, scale])
                    cc_list.append(self._metrics[(model, layer, scale)]["cc"].mean().item())
                    cc_norm_list.append(self._metrics[(model, layer, scale)]["cc_norm"].mean().item())
                    cc_spe_list.append(self._metrics[(model, layer, scale)]["cc_spe"].mean().item())

        multi_index_df = pd.DataFrame(multi_index_list, columns=["model", "layer", "scale"])
        summary_df = pd.DataFrame({"cc": cc_list, "cc_norm": cc_norm_list, "cc_spe": cc_spe_list}, index=pd.MultiIndex.from_frame(multi_index_df))

        if contract_model_layer:
            summary_df = summary_df.reset_index()
            summary_df["model"] = summary_df["model"] + "_" + summary_df["layer"]
            del summary_df["layer"]

        return summary_df

    def build(self):
        for model in self._model_query.keys():
            for layer in self._model_query[model].keys():
                for scale in self._model_query[model][layer]:
                    scale = str(scale)
                    metrics = ModelMetricsBuilder(self._root, self._dataset, ntau=self._ntau, nlat=self._nlat, nspan=self._nspan, scale=scale, model=model, layer=layer, n_pca=self._n_pca)
                    metrics.build()
                    self._metrics[(model, layer, scale)] = metrics.metrics


class ModelMetricsBuilder:

    # Build results for a single model on a given dataset

    def __init__(self, root, dataset, ntau, nlat, nspan, scale, model, layer, n_pca):
        self._dataset = dataset

        self._model = self._load_model(root, model, dataset, ntau, nlat, nspan, scale, n_pca, layer)
        self._test_data = datasets.PCDataset(root, model, dataset, "test", ntau=ntau, nlat=nlat, nspan=nspan, scale=scale, n_pca=n_pca, layer=layer)

        self.output = self._compute_output().permute(1, 0)
        self.target = self._test_data.y
        if len(self.target.shape) == 3:  # Average over repeats
            self.target = self.target.mean(dim=2).detach().cpu().permute(1, 0)
        mask = util.get_valid_neurons(dataset, thresh=0.2)  # 0.2
        self.output = self.output[mask]
        self.target = self.target[mask]
        self.max_cc = util.get_dataset_max_ccs(dataset, "ccnorm")[mask]
        self.max_spe = util.get_dataset_max_ccs(dataset, "spe")[mask]

        self._metrics = {}

    @property
    def metrics(self):
        if len(self._metrics) == 0:
            self._metrics = self.build()

        return self._metrics

    def build(self):
        cc = neural.cc(self.output, self.target)
        cc_norm = cc / self.max_cc
        cc_spe = cc / self.max_spe

        metrics = {"cc": cc, "cc_norm": cc_norm, "cc_spe": cc_spe}

        return metrics

    def _compute_output(self):
        with torch.no_grad():
            return self._model(self._test_data.x.cuda().float()).cpu().detach()

    def _load_model(self, root, model, dataset, ntau, nlat, nspan, scale, n_pca, layer):
        model_name = util.get_dataset_x_name(model, dataset, "train", ntau, nlat, nspan, scale, n_pca, layer)
        model_path = f"{root}/results/{model_name}/models"

        return train.Trainer.load_model(model_path, "0")
