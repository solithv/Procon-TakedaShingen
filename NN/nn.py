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


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, input_dim, output_dim):
        super(PositionalEmbedding, self).__init__()
        self.pos_embedding = self.add_weight(
            "pos_embedding",
            shape=(1, sequence_length, input_dim),
            initializer=tf.keras.initializers.RandomNormal(),
            trainable=True,
        )
        self.input_dim = input_dim
        self.output_dim = output_dim
        print(self.pos_embedding)

    def call(self, inputs):
        return inputs + self.pos_embedding


class NNModel:
    def make_model(self, sides: int = 5):
        """モデルを作成

        Args:
            sides (int, optional): 1辺の長さ. Defaults to 3.
        """
        input_shape = (len(Game.CELL[: Game.CELL.index("worker_A0")]) + 2, sides, sides)
        output_size = len(Game.ACTIONS)
        self.model = self.define_model(input_shape, output_size)

    def define_model(
        self,
        input_shape: tuple[int],
        output_dim: int,
        num_heads: int = 4,
        ff_dim: int = 32,
        num_layers: int = 12,
    ):
        """モデルを定義

        Args:
            input_shape (tuple[int]): 入力の形状
            output_dim (int): 出力の次元数
            num_heads (int, optional): マルチヘッドアテンションのヘッド数. Defaults to 4.
            ff_dim (int, optional): フィードフォワードネットワークの次元数. Defaults to 32.
            num_layers (int, optional): トランスフォーマーレイヤーの数. Defaults to 2.
        """

        # 入力パッチ化
        input_layer = keras.layers.Input(shape=input_shape)
        patches = layers.Reshape((input_shape[0] * input_shape[1], input_shape[2]))(
            input_layer
        )

        # 位置エンコーディング
        positional_embedding = layers.Embedding(
            input_dim=input_shape[0] * input_shape[1], output_dim=input_shape[2]
        )(tf.range(input_shape[0] * input_shape[1]))

        # パッチエンベディングと位置エンコーディングを結合
        patch_embeddings = layers.Add()([patches, positional_embedding])

        # トランスフォーマブロック
        num_layers = 12  # トランスフォーマブロックの数
        num_heads = 4  # 注意ヘッドの数
        embedding_dim = input_shape[2]  # 埋め込み次元

        for _ in range(num_layers):
            # Multi-Head Self Attention
            attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim // num_heads)(value=patch_embeddings,query=patch_embeddings,key=patch_embeddings,attention_mask=None,)
            attention_output = layers.LayerNormalization(epsilon=1e-6)(
                layers.Add()([patch_embeddings, attention_output])
            )

            # Feed Forward Network
            feed_forward = layers.Conv1D(
                filters=embedding_dim, kernel_size=1, activation="relu"
            )(attention_output)
            feed_forward = layers.Conv1D(filters=embedding_dim, kernel_size=1)(
                feed_forward
            )
            feed_forward = layers.LayerNormalization(epsilon=1e-6)(
                layers.Add()([attention_output, feed_forward])
            )

            patch_embeddings = feed_forward

        # クラス予測用の出力ヘッド
        cls_token = patch_embeddings[:, 0, :]  # クラス情報はCLSトークンから取得
        output_layer = layers.Dense(output_dim, activation="softmax")(cls_token)

        # モデルを構築
        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        return model

        # inputs = keras.Input(input_shape)
        # x = tf.transpose(inputs, (0, 2, 3, 1))
        # # x = layers.Conv2D(64, (5, 5), padding="same", activation="relu")(x)
        # x = layers.Conv2D(16, (3, 3), padding="same", activation="relu")(x)
        # x = layers.SpatialDropout2D(0.2)(x)
        # # x = layers.Dropout(0.2)(x)
        # # x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
        # x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
        # # x = layers.Dropout(0.5)(x)
        # x = layers.Flatten()(x)
        # x = layers.Dense(64, activation="relu")(x)
        # x = layers.Dropout(0.5)(x)
        # # x = layers.Dense(32, activation="relu")(x)
        # outputs = layers.Dense(output_size, activation="softmax")(x)

        # return models.Model(inputs=inputs, outputs=outputs)

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
