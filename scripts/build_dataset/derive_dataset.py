import os
import sys
import argparse

import matplotlib
import matplotlib.image
import numpy as np
import tables

sys.path.append("../../../")

# Code adapted from Mineault et al. 2021
# This script builds the PVC1 dataset

def derive_pvc1(args):
    root = f"{args.data_root}/crcns-pvc1"
    movie_info = {}

    out_path = f"{args.output_dir}/crcns-pvc1"
    try:
        os.makedirs(out_path)
    except FileExistsError:
        pass

    h5file = tables.open_file(f"{out_path}/movies.h5", "w")

    # Crop and downsample the frames to 112x112. This reduces the
    # interlacing artifacts
    for i in range(30):
        for j in range(4):
            print(i, j)
            root_ = os.path.join(root, "movie_frames", f"movie{j:03}_{i:03}.images")
            with open(os.path.join(root_, "nframes"), "r") as f:
                nframes = int(f.read())

                ims = []
                for frame in range(nframes):
                    im_name = f"movie{j:03}_{i:03}_{frame:03}.jpeg"
                    the_im = matplotlib.image.imread(os.path.join(root_, im_name))

                    assert the_im.shape[0] == 240
                    the_im = the_im.reshape((120, 2, 160, 2, 3)).mean(3).mean(1)
                    the_im = the_im[8:, 24:136, :].transpose((2, 0, 1))
                    ims.append(the_im.astype(np.uint8))

                m = np.stack(ims, axis=0)
                h5file.create_array("/", f"movie{j:03}_{i:03}", m, "Movie")

    h5file.close()

    os.system(f'cp -r "{args.data_root}/crcns-pvc1/neurodata" "{out_path}"')


def main(args):
    try:
        os.makedirs(args.output_dir)
    except FileExistsError:
        pass

    if args.dataset == "pvc1":
        derive_pvc1(args)
    else:
        raise NotImplementedError(f"dataset '{args.dataset}' Not implemented")


if __name__ == "__main__":
    desc = "Derive a dataset"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument("--dataset", help="Dataset")
    parser.add_argument("--data_root", help="Data path")
    parser.add_argument("--output_dir", help="Output path")

    args = parser.parse_args()
    main(args)
