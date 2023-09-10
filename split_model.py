from pathlib import Path


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


def main():
    import shutil

    model_path = Path("./model/game")
    shutil.make_archive(
        model_path,
        format="zip",
        root_dir=model_path.parent,
        base_dir=f"{model_path.name}.keras",
    )
    split_zip(f"{model_path}.zip", model_path)


if __name__ == "__main__":
    main()
