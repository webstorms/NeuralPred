import argparse

import torch

from src.datasets import util

# obtain mean and std of dataset


def main(args):
    dataset = util.get_dataset(args.root, name=args.dataset, split="train", ntau=args.ntau, idx=0, normalise=False)

    frames_list = []
    for x, y in dataset:
        frames_list.append(x)
    movie_tensor = torch.stack(frames_list)

    print(f"dataset {args.dataset} {movie_tensor.shape}")
    print(f"mean {movie_tensor.mean()}")
    print(f"std {movie_tensor.std()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str)
    parser.add_argument("--dataset", help="Dataset")
    parser.add_argument("--ntau", type=int, default=1)

    args = parser.parse_args()
    main(args)
