import json
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
import shutil
>>>>>>> origin/3x3
from pathlib import Path
>>>>>>> origin/3x3

import numpy as np

from NN import NNModel
from Utils import Annotator, Util


def unpack_dataset(dir):
    dir = Path(dir)
    for file in dir.glob("*.zip.[0-9][0-9][0-9]"):
        basename = file.name.split(".")[0]
        if dir.joinpath(f"{basename}.dat").exists():
            continue
        Util.combine_split_zip(
            dir.joinpath(basename),
            f"{dir.joinpath(basename)}.zip",
        )
    for file in dir.glob("*.zip"):
        basename = file.name.split(".")[0]
        if dir.joinpath(f"{basename}.dat").exists():
            continue
        shutil.unpack_archive(f"{dir.joinpath(basename)}.zip", dir)


def train():
<<<<<<< HEAD
    output_dir = "./dataset"
    annotator = Annotator(None, output_dir)
    data_path = "./dataset/data.dat"
    model_path = "./model/game"
=======
    dataset_dir = "./dataset"
    model_path = "./model/game"
    annotator = Annotator(None, dataset_dir)
>>>>>>> origin/3x3
    nn = NNModel(model_path)
    nn.make_model(5)

    unpack_dataset(dataset_dir)
    batch_size = 512
    epochs = 1000
    validation_split = 0.7
    x = []
    y = []
<<<<<<< HEAD
<<<<<<< HEAD
    with open(data_path) as f:
        for line in f:
            feature, target = json.loads(line).values()
            x.append(np.array(feature, dtype=np.int8))
            y.append(target)
            features_annotate, targets_annotate = annotator.make_augmentation(
                np.array(feature, dtype=np.int8), np.argmax(target)
            )
            x += features_annotate
            y += targets_annotate
=======
=======

>>>>>>> origin/3x3
    for dataset in Path(dataset_dir).glob("*.dat"):
        with open(dataset) as f:
            for line in f:
                feature, target = json.loads(line).values()
                x.append(np.array(feature, dtype=np.int8))
                y.append(target)
                features_annotate, targets_annotate = annotator.make_augmentation(
                    np.array(feature, dtype=np.int8), np.argmax(target)
                )
                x += features_annotate
                y += targets_annotate
>>>>>>> origin/3x3
    x = np.array(x)
    y = np.array(y)
    print(x.shape, y.shape)

    nn.train(x, y, batch_size, epochs, validation_split)


if __name__ == "__main__":
    train()
