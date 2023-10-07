import argparse
from src import datasets


def main(args):
    pc_builder = datasets.PCBuilder(args.root, args.model, args.layer, args.dataset, args.ntau, args.nlat, args.nspan, args.scale, args.n_pca)
    pc_builder.fit_and_transform(args.batch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build model activity")
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--dataset", type=str, default="pvc1")
    parser.add_argument("--ntau", type=int, default=25)
    parser.add_argument("--nlat", type=int, default=1)
    parser.add_argument("--nspan", type=int, default=5)
    parser.add_argument("--scale", type=float, default=1)  # 0.66, 1, 1.5
    parser.add_argument("--model", type=str, default="stacktp")
    parser.add_argument("--layer", type=str, default=None)
    parser.add_argument("--n_pca", type=int, default=500)  # Number of pca components
    parser.add_argument("--batch", type=int, default=1000)

    args = parser.parse_args()
    main(args)
