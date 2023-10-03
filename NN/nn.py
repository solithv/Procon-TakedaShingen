import tracemalloc
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import layers, models

from MyEnv import Game
from Utils import Util

from .dataset import DatasetUtil


class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


class NNModel:
    def make_model(self, sides: int = 5):
        """モデルを作成

        Args:
            sides (int, optional): 1辺の長さ. Defaults to 3.
        """
        input_shape = (len(Game.CELL[: Game.CELL.index("worker_A0")]) + 2, sides, sides)
        output_size = len(Game.ACTIONS)
        self.model = self.define_model(input_shape, output_size, patch_size=5)

    def define_model(
        self,
        input_shape,
        num_classes,
        learning_rate=0.001,
        weight_decay=0.0001,
        batch_size=256,
        num_epochs=100,
        image_size=72,  # We'll resize input images to this size
        patch_size=6,  # Size of the patches to be extract from the input images
        projection_dim=64,
        num_heads=4,
        transformer_layers=8,
        mlp_head_units=[2048, 1024],  # Size of the dense layers of the final classifier
    ):
        def mlp(x, hidden_units, dropout_rate):
            for units in hidden_units:
                x = layers.Dense(units, activation=tf.nn.gelu)(x)
                x = layers.Dropout(dropout_rate)(x)
            return x

        def create_vit_classifier(input_shape, num_classes):
            inputs = layers.Input(shape=input_shape)
            print(inputs)
            preshape = tf.transpose(inputs, (0, 2, 3, 1))
            print(preshape)

            # Create patches.
            patches = Patches(patch_size)(preshape)
            # Encode patches.
            encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

            # Create multiple layers of the Transformer block.
            for _ in range(transformer_layers):
                # Layer normalization 1.
                x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
                # Create a multi-head attention layer.
                attention_output = layers.MultiHeadAttention(
                    num_heads=num_heads, key_dim=projection_dim, dropout=0.1
                )(x1, x1)
                # Skip connection 1.
                x2 = layers.Add()([attention_output, encoded_patches])
                # Layer normalization 2.
                x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
                # MLP.
                x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
                # Skip connection 2.
                encoded_patches = layers.Add()([x3, x2])

            # Create a [batch_size, projection_dim] tensor.
            representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            representation = layers.Flatten()(representation)
            representation = layers.Dropout(0.5)(representation)
            # Add MLP.
            features = mlp(
                representation, hidden_units=mlp_head_units, dropout_rate=0.5
            )
            # Classify outputs.
            logits = layers.Dense(num_classes)(features)
            # Create the Keras model.
            model = keras.Model(inputs=inputs, outputs=logits)
            return model

        num_patches = (image_size // patch_size) ** 2
        transformer_units = [
            projection_dim * 2,
            projection_dim,
        ]  # Size of the transformer layers

        return create_vit_classifier(input_shape, num_classes)

    def save_model(self, model_dir: str, model_name: str):
        """モデルを保存

        Args:
            model_dir (str): モデルの保存先
            model_name (str): モデルの保存名
        """
        model_dir: Path = Path(model_dir)
        model_name: Path = model_name
        model_dir.mkdir(exist_ok=True)
        model_save = model_dir / model_name
        self.model.save(model_save, save_format="tf")
        if any(
            f.stat().st_size > 100 * (1024**2)
            for f in model_save.glob("**/*")
            if f.is_file()
        ):
            Util.compress_and_split(model_save, model_name, model_dir)

    def load_model(self, model_dir: str, model_name: str, from_zip: bool = True):
        """モデルを読み込み

        Args:
            model_dir (str): モデルの保存先
            model_name (str): モデルの保存名
            from_zip (bool, optional): 分割zipファイルからの読み込みを優先. Defaults to True.
        """
        model_dir: Path = Path(model_dir)
        model_name: Path = model_name
        model_file = model_dir / model_name
        if from_zip and list(model_dir.glob(f"{model_name}.zip*")):
            Util.combine_and_unpack(model_dir, model_name)
        self.model = models.load_model(model_file)

    def train(
        self,
        batch_size: int,
        epochs: int,
        validation_split: float,
        dataset_dir: str = "./dataset",
        model_path: str = "./model",
        model_name: str = "game",
        checkpoint_dir: str = None,
        log_dir: str = None,
        plot: bool = True,
        load_model: str = None,
    ):
        """学習

        Args:
            batch_size (int): バッチサイズ
            epochs (int): エポック数
            validation_split (float): 検証用データ割合
            dataset_dir (str, optional): データセットのパス. Defaults to "./dataset".
            model_path (str, optional): モデルの保存先. Defaults to "./model".
            model_name (str, optional): モデルの保存名. Defaults to ".game".
            checkpoint_dir (str, optional): checkpointの保存先. Defaults to None.
            log_dir (str, optional): tensorboardログの保存先. Defaults to None.
            plot (bool, optional): 学習履歴を可視化するか. Defaults to True.
        """
        tracemalloc.start()

        if load_model:
            self.model: models.Model = models.load_model(load_model)
        else:
            self.make_model(5)
            self.model.compile(
                optimizer="adam",
                loss="categorical_crossentropy",
                metrics=["accuracy"],
                run_eagerly=True,
            )
        self.model.summary()

        x, y = DatasetUtil().load_dataset(dataset_dir)

        callbacks = []
        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=20, verbose=1, restore_best_weights=True
        )
        callbacks.append(early_stopping)
        if checkpoint_dir:
            checkpoint = keras.callbacks.ModelCheckpoint(
                checkpoint_dir, save_best_only=True
            )
            callbacks.append(checkpoint)
        if log_dir:
            log_dir: Path = Path(log_dir)
            log_dir.mkdir(exist_ok=True)
            tensorboard = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
            callbacks.append(tensorboard)

        history = self.model.fit(
            x,
            y,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=callbacks,
        )
        self.save_model(model_path, model_name)

        if plot:
            fig, axes = plt.subplots(2, 1)
            fig.subplots_adjust(hspace=0.6)
            axes[0].plot(history.history["accuracy"])
            axes[0].plot(history.history["val_accuracy"])
            axes[0].set_title("Model accuracy")
            axes[0].set_ylabel("Accuracy")
            axes[0].set_xlabel("Epoch")
            axes[0].legend(["Train", "Validation"], loc="upper left")

            axes[1].plot(history.history["loss"])
            axes[1].plot(history.history["val_loss"])
            axes[1].set_title("Model loss")
            axes[1].set_ylabel("Loss")
            axes[1].set_xlabel("Epoch")
            axes[1].legend(["Train", "Validation"], loc="upper left")
            plt.show()

        tracemalloc.stop()

    def test_model(self, x, y):
        test_loss, test_acc = self.model.evaluate(x, y)

        print(f"test_loss: {test_loss}")
        print(f"test_acc: {test_acc}")

    def predict(self, inputs: list[np.ndarray]):
        """予測

        Args:
            inputs (list[np.ndarray]): 職人の周囲

        Returns:
            list[int]: 行動のリスト
        """
        out = self.model.predict(np.array(inputs, dtype=np.int8))
        args = np.argmax(out, axis=1)
        return args
