import json
import shutil
from pathlib import Path

import numpy as np

from NN import NNModel
from Utils import Annotator, Util


def unpack_dataset(dir):
    dir = Path(dir)
    is_combine = {}
    for file in dir.glob("*.zip.[0-9][0-9][0-9]"):
        basename = file.name.rsplit(2)[0]
        if any("zip" not in item for item in dir.glob(f"{basename}.*")):
            continue
        Util.combine_split_zip(dir, basename, dir)
        is_combine[basename] = True
    for file in dir.glob("*.zip"):
        basename = file.stem
        shutil.unpack_archive(file, dir)
        if is_combine.get(basename):
            file.unlink()


def train():
    dataset_dir = "./dataset"
    model_path = "./model"
    model_name = "game"
    annotator = Annotator(None, dataset_dir)
    nn = NNModel()
    nn.make_model(5)

    unpack_dataset(dataset_dir)
    batch_size = 512
    epochs = 1000
    validation_split = 0.7
    x = []
    y = []

    for dataset in Path(dataset_dir).glob("*.dat"):
        print(dataset)
        with open(dataset) as f:
            for line in f:
                feature, target = json.loads(line).values()
                x.append(np.array(feature, dtype=np.int8))
                y.append(target)
                features_annotate, targets_annotate = annotator.make_augmentation(
                    np.array(feature, dtype=np.int8), target
                )
                x += features_annotate
                y += targets_annotate
    x = np.array(x)
    y = np.array(y)
    print(x.shape, y.shape)

    nn.train(x, y, batch_size, epochs, validation_split)
    nn.save_model(model_path, model_name)

    nn.test_model(x, y)


if __name__ == "__main__":
    train()
