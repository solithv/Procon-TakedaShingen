import json

import numpy as np
from MyEnv import NNModel


def train():
    data_path = "./dataset/data.dat"
    model_path = "./model/game"
    nn = NNModel(model_path)
    nn.make_model(5)
    batch_size = 12
    epochs = 1000
    validation_split = 0.7

    x = []
    y = []
    with open(data_path) as f:
        for line in f:
            feature, target = json.loads(line).values()
            x.append(np.array(feature, dtype=np.int8))
            y.append(target)
    x = np.array(x)
    y = np.array(y)

    nn.train(x, y, batch_size, epochs, validation_split)


if __name__ == "__main__":
    train()
