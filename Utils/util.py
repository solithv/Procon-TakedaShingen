from pathlib import Path


class Util:
    @staticmethod
    def split_zip(
        input_zip_file: str, output_directory: str, chunk_size: int = 100 * (1024**2)
    ) -> None:
        """GitHubで学習済みモデルデータを管理できるようにzipファイルを分割

        Args:
            input_zip_file (str): 分割するzipファイルのパス
            output_directory (str): 出力先のパス
            chunk_size (int, optional): 1ファイル当たりのサイズ. Defaults to 100*(1024**2).
        """
        for filename in Path(output_directory).parent.glob("*.zip.[0-9][0-9][0-9]"):
            filename.unlink()
        with open(input_zip_file, "rb") as f:
            data = f.read()
        for i in range((len(data) + chunk_size - 1) // chunk_size):
            chunk_data = data[i * chunk_size : (i + 1) * chunk_size]
            with open(f"{output_directory}.zip.{i+1:0>3}", "wb") as chunk_file:
                chunk_file.write(chunk_data)

    @staticmethod
    def combine_split_zip(input_directory, output_zip_file):
        """分割したzipファイルを結合

        Args:
            input_directory (_type_): 分割zipファイルのパス
            output_zip_file (_type_): 出力先のパス
        """
        temp_buffer = b""
        for filename in sorted(
            Path(input_directory).parent.glob("*.zip.[0-9][0-9][0-9]")
        ):
            if filename.is_file():
                with open(filename, "rb") as chunk_file:
                    temp_buffer += chunk_file.read()

        with open(output_zip_file, "wb") as output_file:
            output_file.write(temp_buffer)


def split_model():
    import shutil

    model_path = Path("./model/game")
    shutil.make_archive(
        model_path,
        format="zip",
        root_dir=model_path.parent,
        base_dir=f"{model_path.name}.keras",
    )
    Util.split_zip(f"{model_path}.zip", model_path)


def restore_model():
    import os
    import shutil

    model_path = "./model/game"
    os.makedirs(model_path, exist_ok=True)
    Util.combine_split_zip(model_path, f"{model_path}.zip")
    shutil.unpack_archive(f"{model_path}.zip", model_path)


if __name__ == "__main__":
    split_model()
    restore_model()
