import os
import argparse

import torch

from src import util

# Build em response tensors

def build_unit_activity_tensor(dataset):
    unit_activity_list = []

    for i, (_, y) in enumerate(dataset):
        if i % 10000 == 0:
            print(f"i={i}/{len(dataset)}")
        unit_activity_list.append(y)

    return torch.stack(unit_activity_list)


def main(args):
    # Create directory to store all tensors
    unit_activity_list = []

    if args.dataset == "multi_pvc1":
        for i in range(util.get_dataset_number_of_neurons(args.dataset)):
            print(f"Building unit activity {i}...")
            dataset = util.get_dataset(args.root, name=args.dataset, split=args.split, ntau=args.ntau, idx=i)
            unit_activity_tensor = build_unit_activity_tensor(dataset)
            print(unit_activity_tensor.shape)
            unit_activity_list.append(unit_activity_tensor)
            all_unit_activity_tensor = torch.stack(unit_activity_list)
    else:
        dataset = util.get_dataset(args.root, name=args.dataset, split=args.split, ntau=args.ntau, idx=None)
        all_unit_activity_tensor = build_unit_activity_tensor(dataset)

        if args.dataset == "cadena":
            all_unit_activity_tensor = all_unit_activity_tensor.permute(1, 0, 2)

    save_path = f"{args.root}/data/y/"
    unit_tensor_name = os.path.join(save_path, f"{util.get_dataset_y_name(args.dataset, args.split, args.ntau)}.pt")
    torch.save(all_unit_activity_tensor.float(), unit_tensor_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--split", type=str)
    parser.add_argument("--ntau", type=int)

    args = parser.parse_args()
    main(args)
