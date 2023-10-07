from pathlib import Path

import torch
import torch.nn.functional as F
from brainbox import trainer
from brainbox.physiology.neural import cc

from src import readout


class Trainer(trainer.Trainer):

    def __init__(self, root, model, train_dataset, n_epochs, batch_size, lr, lam, shuffle=True, device="cuda", id=None):
        super().__init__(root, model, train_dataset, n_epochs, batch_size, lr, optimizer_func=torch.optim.Adam, scheduler_func=None, scheduler_kwargs={}, loader_kwargs={"shuffle": shuffle}, device=device, grad_clip_type=None, grad_clip_value=0, id=id)
        self._lam = lam

    @staticmethod
    def load_model(root, model_id):
        def model_loader(hyperparams):
            model_params = hyperparams["model"]
            del model_params["name"]
            del model_params["weight_initializers"]

            return readout.Readout(**model_params)

        return trainer.load_model(root, model_id, model_loader)

    @property
    def hyperparams(self):
        return {**super().hyperparams, "lam": self._lam}

    def on_epoch_complete(self, save):
        n_epochs = len(self.log["train_loss"])
        if n_epochs % 100 == 0:
            print(f"Completed {n_epochs}...")
        # Save logs and hyperparams
        if save:
            self.save_model()
            self.save_model_log()
            self.save_hyperparams()

    def loss(self, output, target, model):
        assert output.shape == target.shape

        # Compute neural loss
        pred_loss = F.poisson_nll_loss(output, target, log_input=False, full=False, eps=1e-08, reduction="mean")

        # Compute reg loss
        reg_loss = 0
        for param in model.get_params():
            reg_loss += self._lam * torch.norm(param, p=1)
        total_loss = pred_loss + reg_loss

        return total_loss

    def train(self, save=True):
        super().train(save)


class CrossValidationTrainer(trainer.KFoldValidationTrainer):

    LAMBDAS = [10**-2.5, 10**-3, 10**-3.5, 10**-4, 10**-4.5, 10**-5, 10**-5.5, 10**-6, 10**-6.5]

    def __init__(self, root, model, train_dataset, n_epochs, batch_size, lr, k, final_repeat=1):
        Path(root).mkdir(parents=True, exist_ok=True)
        val_batch_size = len(train_dataset)
        val_loss = lambda output, target: -(cc(output.permute(1, 0), target.permute(1, 0))).mean()
        trainer_kwargs = {"n_epochs": n_epochs, "batch_size": batch_size, "lr": lr}
        super().__init__(root, model, train_dataset, Trainer, trainer_kwargs, CrossValidationTrainer.LAMBDAS, k, minimise_score=True, final_repeat=final_repeat, val_loss=val_loss, val_batch_size=val_batch_size)
