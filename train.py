<<<<<<< HEAD
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

=======
>>>>>>> origin/3x3
from NN import NNModel


def train():
<<<<<<< HEAD
    output_dir = "./dataset"
    annotator = Annotator(None, output_dir)
    data_path = "./dataset/data.dat"
    model_path = "./model/game"
=======
    dataset_dir = "./dataset"
<<<<<<< HEAD
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
=======
    model_path = "./model"
    model_name = "game"
    checkpoint_dir = "./checkpoint"
    log_dir = "./log"
    batch_size = 1024
    epochs = 1000
    validation_split = 0.7

    nn = NNModel()
>>>>>>> origin/3x3

    nn.train(
        batch_size,
        epochs,
        validation_split,
        dataset_dir=dataset_dir,
        model_path=model_path,
        model_name=model_name,
        checkpoint_dir=checkpoint_dir,
    )


if __name__ == "__main__":
    train()
