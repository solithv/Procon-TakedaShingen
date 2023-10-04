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


class NNModel:
    def make_model(self, sides: int = 5, *args, **kwargs):
        """モデルを作成

        Args:
            sides (int, optional): 1辺の長さ. Defaults to 3.
        """
        input_shape = (sides, sides, len(Game.CELL[: Game.CELL.index("worker_A0")]) + 2)
        output_size = len(Game.ACTIONS)
        self.model = self.define_model(input_shape, output_size, *args, **kwargs)

    def define_model(
        self,
        input_shape: tuple[int],
        num_classes: int,
        patch_size: tuple[int] = (1, 1),
        embedding_dim: int = 64,
        num_heads: int = 4,
        mlp_dim: int = 128,
        num_layers: int = 4,
        dropout_rate: float = 0.1,
    ):
        """モデルを定義

        Args:
            input_shape (tuple[int]): 入力の形状
            num_classes (int): 出力のクラス数
            patch_size (tuple[int], optional): パッチのサイズ. Defaults to (1, 1).
            embedding_dim (int, optional): パッチの埋め込み次元. Defaults to 64.
            num_heads (int, optional): 注意ヘッドの数. Defaults to 4.
            mlp_dim (int, optional): MLPの隠れ層の次元. Defaults to 128.
            num_layers (int, optional): Transformerレイヤーの数. Defaults to 4.
            dropout_rate (float, optional): ドロップアウト率. Defaults to 0.1.
        """
        num_patches = (input_shape[0] // patch_size[0]) * (
            input_shape[1] // patch_size[1]
        )
        # 入力レイヤー
        inputs = layers.Input(shape=input_shape)

        # パッチの抽出
        patch_dim = input_shape[2]
        patches = layers.Reshape((num_patches, patch_dim))(inputs)

        # パッチの埋め込み
        embedding_layer = layers.Dense(embedding_dim, activation="relu")
        embedded_patches = embedding_layer(patches)

        # ポジショナルエンコーディング
        position_embedding_layer = layers.Embedding(
            input_dim=num_patches, output_dim=embedding_dim
        )
        positions = tf.range(start=0, limit=num_patches, delta=1)
        position_embeddings = position_embedding_layer(positions)

        # パッチ埋め込みとポジションエンコーディングの結合
        patch_embeddings = embedded_patches + position_embeddings

        # Transformerエンコーダーの構築
        x = patch_embeddings
        for _ in range(num_layers):
            # Multi-Head Self-Attention
            x = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=embedding_dim // num_heads
            )(x, x)
            x = layers.LayerNormalization(epsilon=1e-6)(
                x + patch_embeddings
            )  # Residual Connection
            x = layers.Dropout(rate=dropout_rate)(x)  # ドロップアウト

            # MLPブロック
            y = layers.Dense(mlp_dim, activation="relu")(x)
            y = layers.Dense(embedding_dim)(y)
            x = layers.LayerNormalization(epsilon=1e-6)(x + y)  # Residual Connection
            x = layers.Dropout(rate=dropout_rate)(x)  # ドロップアウト

        # 全結合層を追加してクラス分類を行う
        x = layers.Flatten()(x)
        x = layers.Dropout(rate=dropout_rate)(x)  # ドロップアウト
        outputs = layers.Dense(num_classes, activation="softmax")(x)

        # モデルを構築
        model = models.Model(inputs=inputs, outputs=outputs)
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
        out = self.model.predict(
            np.array(inputs, dtype=np.float32).transpose((0, 2, 3, 1))
        )
        args = np.argmax(out, axis=1)
        return args
