import json
import shutil
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from Utils import Annotator, Util


class DatasetUtil:
    def unpack_dataset(self, dir):
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

    def load_dataset(self, dataset_dir):
        self.unpack_dataset(dataset_dir)
        x = []
        y = []
        for dataset in Path(dataset_dir).glob("*.dat"):
            print(dataset)
            with open(dataset) as f:
                for line in f:
                    feature, target = json.loads(line).values()
                    feature = np.array(feature, dtype=np.int8)
                    target = np.array(target, dtype=np.int8)
                    x.append(feature)
                    y.append(target)
                    features_annotate, targets_annotate = Annotator.make_augmentation(
                        feature, target
                    )
                    x += features_annotate
                    y += targets_annotate
        x = np.array(x)
        y = np.array(y)
        print(x.shape, y.shape)
        return x, y

    def make_generators(self, dataset_dir, batch_size, split_rate=0.7):
        features, targets = self.load_dataset(dataset_dir)
        train_dataset, valid_dataset = keras.utils.split_dataset(
            (features, targets), split_rate, shuffle=True
        )
        train_dataset: tf.data.Dataset
        valid_dataset: tf.data.Dataset
        train_size = int(train_dataset.cardinality())
        valid_size = int(valid_dataset.cardinality())
        train_generator: tf.data.Dataset = (
            train_dataset.shuffle(train_size).batch(batch_size).repeat()
        )
        valid_generator: tf.data.Dataset = (
            train_dataset.map(self.augmentation).batch(batch_size).repeat()
        )

        return train_generator, valid_generator, train_size, valid_size
