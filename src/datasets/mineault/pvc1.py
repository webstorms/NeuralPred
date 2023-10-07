from . import utils

import datetime
import glob
import numpy as np
import os
import subprocess
import tables
import time

import torch.utils.data

movie_cache = {}

# Code adapted from Mineault et al. 2021

class PVC1(torch.utils.data.Dataset):

    def __init__(
        self,
        root="./crcns-pvc1",
        nx=224,
        ny=224,
        nt=20,
        ntau=1,
        nframedelay=2,
        nframestart=15,
        split="train",
        single_cell=-1,
        virtual=False,
        repeats=False,
    ):

        framerate = 30.0

        if split not in ("train", "tune", "report", "traintune"):
            raise NotImplementedError("Split is set to an unknown value")

        if ntau + nframedelay > nframestart:
            raise NotImplementedError(
                "ntau + nframedelay > nframestart, sequence starts before frame 0"
            )

        self.nx = nx
        self.ny = ny
        self.nt = nt
        self.ntau = ntau
        self.nframedelay = nframedelay
        self.nframestart = nframestart
        self.split = split
        self.movie_info = _movie_info(root)
        self.root = root

        paths = []
        for item in glob.glob(os.path.join(root, "neurodata", "*", "*.mat")):
            paths.append(item)

        paths = sorted(paths)

        # if single_cell != -1:
        #    paths = [paths[single_cell]]

        # Create a mapping from a single index to the necessary information needed to load the corresponding data
        sequence = []
        self.mat_files = {}
        set_num = 0

        splits = {
            "train": [0, 1, 2, 3, 5, 6, 7, 8],
            "tune": [4],
            "report": [9],
            "traintune": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        }

        nblocks = 10

        cumulative_electrodes = 0
        nrepeats = []
        for path in paths:
            mat_file = utils.load_mat_as_dict(path)
            key = path
            self.mat_files[key] = mat_file

            batch = mat_file["pepANA"]["listOfResults"][0]
            if batch["noRepeats"] > 1:
                n_electrodes = len(batch["repeat"][0]["data"])
            else:
                n_electrodes = len(batch["repeat"]["data"])

            # Load all the conditions.
            n_electrodes_seen = 0
            for j, condition in enumerate(mat_file["pepANA"]["listOfResults"]):
                if condition["symbols"][0] != "movie_id":
                    # This is not movie data, skip.
                    # print(f'non-movie dataset, skipping {key}, {j}')
                    continue

                if n_electrodes_seen == 0:
                    nrepeats += [batch["noRepeats"]] * n_electrodes

                n_electrodes_seen = n_electrodes

                set_num += 1

                # The train, tune and report splits
                if set_num % nblocks not in splits[split]:
                    continue

                which_movie = condition["values"]
                cond = self.movie_info[tuple(which_movie)]
                nframes = cond["nframes"]
                nskip = nt
                ntrainings = int(np.ceil((nframes - nt) / nskip) + 1)

                for start_time in range(self.nframestart, nframes, nskip):
                    if start_time + nskip + 1 > nframes:
                        # Incomplete frame.
                        continue

                    end_time = min((nframes, start_time + nskip + 1))
                    spike_frames = np.arange(start_time, end_time)
                    bins = spike_frames / framerate
                    for i in range(n_electrodes):
                        # Although this data was recorded multiple electrodes at a time, give it one electrode at a time
                        # to fit better with other data, e.g. Jack's
                        sequence.append(
                            {
                                "key": key,
                                "movie_path": os.path.join(
                                    root,
                                    "movie_frames",
                                    f"movie{which_movie[0]:03}_{which_movie[1]:03}.images",
                                ),
                                "movie": which_movie[0],
                                "segment": which_movie[1],
                                "result": j,
                                "start_frame": start_time
                                - self.nframedelay
                                - self.ntau,
                                "end_frame": end_time - self.nframedelay - 2,
                                "abs_electrode_num": cumulative_electrodes + i,
                                "rel_electrode_num": i,
                                "bins": bins,
                                "spike_frames": spike_frames,
                                "nframes": nframes,
                            }
                        )

            cumulative_electrodes += n_electrodes_seen

        self.underlying_electrodes = cumulative_electrodes
        if virtual:
            transforms = virtualize.list_transformations(virtual)
            long_seq = []
            delta = 0
            for t in transforms:
                long_seq += [
                    {
                        "virtual_electrode_num": v["abs_electrode_num"] + delta,
                        "transform": t,
                        **v,
                    }
                    for v in sequence
                ]
                delta += cumulative_electrodes
            sequence = long_seq
            cumulative_electrodes = cumulative_electrodes * len(transforms)

        self.nrepeats = np.array(nrepeats)
        self.sequence = sequence
        self.total_electrodes = cumulative_electrodes

        if repeats:
            idx = self.nrepeats > 1
            elecnum = np.where(idx)[0]
            elecmap = {y: x for x, y in enumerate(elecnum)}
            self.nrepeats = self.nrepeats[idx]
            seq = []
            for el in self.sequence:
                if el["abs_electrode_num"] in elecnum:
                    el["abs_electrode_num"] = elecmap[el["abs_electrode_num"]]
                    seq.append(el)

            self.sequence = seq
            self.total_electrodes = len(elecnum)

        if single_cell != -1:
            seq = []
            for el in self.sequence:
                if el["abs_electrode_num"] == single_cell:
                    el["abs_electrode_num"] = 0
                    seq.append(el)

            self.sequence = seq
            self.total_electrodes = 1
            self.nrepeats = self.nrepeats[single_cell]

    def __getitem__(self, idx):
        # Load a single segment of length idx from disk.
        tgt = self.sequence[idx]
        bins = tgt["bins"]

        global movie_cache

        # Lazy load the set of images.
        index = (tgt["movie"], tgt["segment"])
        if index not in movie_cache:
            path = os.path.join(self.root, "movies.h5")
            h5file = tables.open_file(path, "r")
            node = f'/movie{tgt["movie"]:03}_{tgt["segment"]:03}'
            movie = h5file.get_node(node)[:]
            movie_cache[index] = movie
            h5file.close()
        else:
            movie = movie_cache[index]

        assert tgt["start_frame"] >= 0 and tgt["end_frame"] <= movie.shape[0]

        # Movie segments are in the shape nframes x nchannels x ny x nx
        imgs = movie[tgt["start_frame"] : tgt["end_frame"], ...].transpose((1, 0, 2, 3))

        # Center and normalize.
        # This seems like a random order, but it's to fit with the ordering
        # the standard ordering of conv3d.
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html
        X = (
            imgs.astype(np.float32)
            - np.array([83, 81, 73], dtype=np.float32).reshape((3, 1, 1, 1))
        ) / 64.0

        if "transform" in tgt:
            X = virtualize.transform(torch.tensor(X), tgt["transform"])
            abs_electrode_num = tgt["virtual_electrode_num"]
        else:
            abs_electrode_num = tgt["abs_electrode_num"]

        mat_file = self.mat_files[tgt["key"]]

        batch = mat_file["pepANA"]["listOfResults"][tgt["result"]]

        el = tgt["rel_electrode_num"]
        y = []

        w = None

        if batch["noRepeats"] > 1:
            y_ = 0
            w = []
            for i in range(len(batch["repeat"])):
                # Bin the total number of spikes. This is simply the multi-unit activity.
                d_, _ = np.histogram(batch["repeat"][i]["data"][el][0], bins)
                y_ += d_
                w.append(d_)
            y.append(y_ / float(len(batch["repeat"])))
            w = np.stack(w).T
        else:
            n_electrodes = len(batch["repeat"]["data"])

            # Bin the total number of spikes. This is simply the multi-unit activity.
            d_, _ = np.histogram(batch["repeat"]["data"][el][0], bins)
            y.append(d_)

        y = np.array(y).T.astype(np.float32)

        # Create a mask from the electrode range
        m = np.zeros((self.total_electrodes), dtype=bool)
        m[abs_electrode_num] = True

        Y = np.zeros((self.total_electrodes, y.shape[0]))
        Y[abs_electrode_num, :] = y.T

        return (X, m, w, Y)

    def __len__(self):
        # Returns the length of a dataset
        return len(self.sequence)


def _movie_info(root):
    """
    Build up a hashmap from tuples of (movie, segment) to info about the
    movie, including location and duration
    """
    path = os.path.join(root, "movies.h5")
    h5file = tables.open_file(path, "r")
    movie_info = {}
    for i in range(30):
        for j in range(4):
            node = f"/movie{j:03}_{i:03}"
            nframes = len(h5file.get_node(node))

            movie_info[(j, i)] = {"nframes": nframes}

    h5file.close()
    return movie_info


def download(root, url=None):
    """Download the dataset to disk.

    Arguments:
        root: root folder to download to.
        url: the root URL to grab the data from.

    Returns:
        True if downloaded correctly
    """
    if url is None:
        url = os.getenv("GCS_ROOT")

    zip_name = "crcns-pvc1.zip"

    out_file = os.path.join(root, "zip", zip_name)
    if os.path.exists(out_file) and os.stat(out_file).st_size == 1798039870:
        print(f"Already fetched {zip_name}")
    else:
        try:
            os.makedirs(os.path.join(root, "zip"))
        except FileExistsError:
            pass

        # Instead of downloading in Python and taking up a bunch of memory, use curl.
        process = subprocess.Popen(
            ["wget", "--quiet", url + zip_name, "-O", out_file],
            stdout=subprocess.DEVNULL,
        )

        t0 = datetime.datetime.now()
        progress = "|\\-/"
        while process.poll() is None:
            dt = (datetime.datetime.now() - t0) / datetime.timedelta(seconds=0.5)
            char = progress[int(dt) % 4]
            print("\r" + char, end="")
            time.sleep(0.5)
        print("\n")

        # Check everything good
        if not os.path.exists(out_file):
            # Something bad happened during download
            print(f"Failed to download {zip_name}")
            return False

    # Now unzip the data if necessary.
    if os.path.exists(os.path.join(root, "crcns-ringach-data")):
        print("Already unzipped")
        return True
    else:
        process = subprocess.Popen(
            ["unzip", out_file, "-d", root],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        process.communicate()
        return True
