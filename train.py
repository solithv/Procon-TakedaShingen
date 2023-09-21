import json
from pathlib import Path

import numpy as np

from annotator import Annotator
from NN import NNModel


def train():
    dataset_dir = "./dataset"
    model_path = "./model/game"
    annotator = Annotator(None, dataset_dir)
    nn = NNModel(model_path)
    nn.make_model(5)
    batch_size = 128
    epochs = 1000
    validation_split = 0.7
    x = []
    y = []
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
    x = np.array(x)
    y = np.array(y)
    print(x.shape, y.shape)

    nn.train(x, y, batch_size, epochs, validation_split)


if __name__ == "__main__":
    train()
