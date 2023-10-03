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


class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                "Embedding dimension should be divisible by the number of heads."
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = tf.reshape(query, (batch_size, -1, self.num_heads, self.projection_dim))
        key = tf.reshape(key, (batch_size, -1, self.num_heads, self.projection_dim))
        value = tf.reshape(value, (batch_size, -1, self.num_heads, self.projection_dim))
        query = tf.transpose(query, perm=[0, 2, 1, 3])
        key = tf.transpose(key, perm=[0, 2, 1, 3])
        value = tf.transpose(value, perm=[0, 2, 1, 3])
        attention_scores = tf.matmul(query, key, transpose_b=True)
        attention_scores = tf.math.divide(
            attention_scores, tf.math.sqrt(tf.cast(self.projection_dim, tf.float32))
        )
        attention_scores = tf.nn.softmax(attention_scores, axis=-1)
        output = tf.matmul(attention_scores, value)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.embed_dim))
        return self.combine_heads(output)


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs):
        attn_output = self.att(inputs)
        out1 = self.layernorm1(inputs + self.dropout1(attn_output))
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + self.dropout2(ffn_output))


class NNModel:
    def make_model(self, sides: int = 5, *args, **kwargs):
        """モデルを作成

        Args:
            sides (int, optional): 1辺の長さ. Defaults to 3.
        """
        input_shape = (sides, sides, len(Game.CELL[: Game.CELL.index("worker_A0")]) + 2)
        output_size = len(Game.ACTIONS)
        self.model = self.define_model(input_shape, output_size, *args, **kwargs)

    def define_model(self, input_shape, num_classes, num_heads=8, dim_feedforward=256):
        input_layer = tf.keras.layers.Input(shape=input_shape)

        positional_encoding_layer = tf.keras.layers.Embedding(
            input_shape[0] * input_shape[1], input_shape[2]
        )(tf.range(input_shape[0] * input_shape[1]))
        positional_encoding_layer = tf.reshape(
            positional_encoding_layer, (input_shape[0], input_shape[1], input_shape[2])
        )
        positional_encoding_layer = tf.expand_dims(positional_encoding_layer, axis=0)
        positional_encoding_layer = tf.tile(
            positional_encoding_layer, [tf.shape(input_layer)[0], 1, 1, 1]
        )
        x = tf.keras.layers.Add()([input_layer, positional_encoding_layer])

        x = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=input_shape[2]
        )(x, x)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

        x = tf.keras.layers.Conv1D(
            filters=dim_feedforward, kernel_size=1, activation="relu"
        )(x)
        x = tf.keras.layers.Conv1D(filters=input_shape[2], kernel_size=1)(x)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        output_layer = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
        return model

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
        out = self.model.predict(np.array(inputs, dtype=np.float32))
        args = np.argmax(out, axis=1)
        return args
