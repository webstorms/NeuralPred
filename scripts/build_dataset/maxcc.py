import os
import argparse

import torch
from brainbox.physiology import neural

from src import util

# This script computes max cc of each neuron for a given dataset


def main(args):
    save_path = f"{args.root}/data/y/"
    unit_tensor_name = os.path.join(save_path, f"{util.get_dataset_y_name(args.dataset, args.split, args.ntau)}.pt")

    y = torch.load(unit_tensor_name)
    n, t, r = y.shape

    max_ccs = []
    for i in range(n):
        max_cc = neural.compute_max_cc(y[i]) if args.type == "ccnorm" else neural.compute_max_spe(y[i])
        max_ccs.append(max_cc if type(max_cc) == float else max_cc.item())

    print(f"max_ccs {max_ccs}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--split", type=str)
    parser.add_argument("--ntau", type=int)
    parser.add_argument("--type", type=str)

    args = parser.parse_args()
    main(args)

# python maxcc.py --root=/home/luketaylor/PycharmProjects/NeuralPred --dataset=multi_pvc1 --split=test --ntau=25
# python maxcc.py --root=/home/luketaylor/PycharmProjects/NeuralPred --dataset=cadena --split=test --ntau=25