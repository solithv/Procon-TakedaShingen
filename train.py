from nn import NNModel


def train():
    model_path = "./model/game"
    nn = NNModel(model_path)
    batch_size = 128
    epochs = 1000
    validation_split = 0.7

    x = None
    y = None

    nn.train(x, y, batch_size, epochs, validation_split)


if __name__ == "__main__":
    train()
