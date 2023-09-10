from pathlib import Path


def combine_split_zip(input_directory, output_zip_file):
    """分割したzipファイルを結合

    Args:
        input_directory (_type_): 分割zipファイルのパス
        output_zip_file (_type_): 出力先のパス
    """
    temp_buffer = b""
    for filename in sorted(Path(input_directory).parent.glob("*.zip.[0-9][0-9][0-9]")):
        if filename.is_file():
            with open(filename, "rb") as chunk_file:
                temp_buffer += chunk_file.read()

    with open(output_zip_file, "wb") as output_file:
        output_file.write(temp_buffer)


def main():
    import os
    import shutil

    model_path = "./model/game"
    os.makedirs(model_path, exist_ok=True)
    combine_split_zip(model_path, f"{model_path}.zip")
    shutil.unpack_archive(f"{model_path}.zip", model_path)


if __name__ == "__main__":
    main()
