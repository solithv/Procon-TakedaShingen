from NN import NNModel


def train():
    dataset_dir = "./dataset"
    model_path = "./model"
    model_name = "game"
    log_dir = "./log"
    batch_size = 128
    epochs = 1000
    validation_split = 0.7

    nn = NNModel()

    nn.train(
        batch_size,
        epochs,
        validation_split,
        dataset_dir,
        model_path,
        model_name,
        log_dir,
    )


if __name__ == "__main__":
    train()
