import argparse
from pathlib import Path

from src import datasets, readout, train, util


def main(args):
    readout_model = readout.Readout(args.n_pca, util.get_dataset_number_of_neurons(args.dataset))
    train_dataset = datasets.PCDataset(args.root, args.model, args.dataset, "train", args.ntau, args.nlat, args.nspan, args.scale, args.n_pca, args.layer)

    file_path = util.get_dataset_x_name(args.model, args.dataset, "train", args.ntau, args.nlat, args.nspan, args.scale, args.n_pca, args.layer)
    # file_path = "/".join(file_path.replace("_train", "").split("_"))
    train_root = f"{args.root}/results/{file_path}"
    Path(train_root).mkdir(parents=True, exist_ok=True)
    cv_trainer = train.CrossValidationTrainer(train_root, readout_model, train_dataset, args.n_epochs, args.batch_size, args.lr, args.k, args.final_repeat)
    cv_trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--dataset", type=str, default="pvc1")
    parser.add_argument("--ntau", type=int, default=25)
    parser.add_argument("--nlat", type=int, default=1)
    parser.add_argument("--nspan", type=int, default=5)
    parser.add_argument("--scale", type=float, default=1)  # 0.66, 1, 1.5
    parser.add_argument("--model", type=str, default="stacktp")
    parser.add_argument("--layer", type=str, default=None)
    parser.add_argument("--n_pca", type=int, default=500)  # Number of pca components
    parser.add_argument("--n_epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--final_repeat", type=int, default=1)

    args = parser.parse_args()
    main(args)
