import csv
import os
import pickle
import shutil
import zipfile
from pathlib import Path
from typing import Iterable, Union

import numpy as np

from MyEnv import Game


class Util:
    @staticmethod
    def split_zip(
        input_zip_file: str,
        output_dir: str = "./",
        chunk_size: int = 100 * (1024**2),
    ) -> None:
        """zipファイルを分割

        Args:
            input_zip_file (str): 分割するzipファイルのパス
            output_dir (str): 出力先のパス. Defaults to "./".
            chunk_size (int, optional): 1ファイル当たりのサイズ.
                Defaults to 100*(1024**2).
        """
        input_zip_file: Path = Path(input_zip_file)
        output_dir: Path = Path(output_dir)
        for filename in output_dir.glob(f"{input_zip_file.stem}.zip.[0-9][0-9][0-9]"):
            filename.unlink()
        with input_zip_file.open("rb") as f:
            data = f.read()
        for i in range((len(data) + chunk_size - 1) // chunk_size):
            chunk_data = data[i * chunk_size : (i + 1) * chunk_size]
            with (output_dir / f"{input_zip_file.stem}.zip.{i+1:0>3}").open(
                "wb"
            ) as chunk_file:
                chunk_file.write(chunk_data)

    @staticmethod
    def compress_and_split(
        input: Union[str, Iterable[str]],
        output_name: str = None,
        output_dir: str = "./",
        chunk_size: int = 100 * (1024**2),
        delete: bool = True,
    ):
        """ファイルを圧縮して必要ならば分割

        Args:
            input (Union[str, Iterable[str]]): 圧縮するファイル・フォルダ
            output_name (str, optional): 出力ファイル名. Defaults to None.
            output_dir (str, optional): 出力先のパス. Defaults to "./".
            chunk_size (int, optional): 1ファイル当たりのサイズ.
                Defaults to 100*(1024**2).
            delete (bool): 分割した後zipファイルを削除するか. Defaults to True.
        """
        output_dir: Path = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        if isinstance(input, (str, Path)):
            input: Path = Path(input)
            if output_name is None:
                output_name = input.stem
            if input.suffix == ".zip":
                pass
            elif input.is_dir():
                shutil.make_archive(
                    str(output_dir / output_name), format="zip", base_dir=input
                )
            elif input.is_file():
                shutil.make_archive(
                    str(output_dir / output_name),
                    format="zip",
                    base_dir=input.name,
                    root_dir=input.parent,
                )
        elif isinstance(input, Iterable):
            if output_name is None:
                output_name = "out"
            with zipfile.ZipFile(
                output_dir / f"{output_name}.zip",
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
            ) as zf:
                for item in input:
                    item = Path(item)
                    if item.is_file():
                        zf.write(item, item.name)
                    elif item.is_dir():
                        for root, dirs, files in sorted(os.walk(item)):
                            for f in files:
                                zf.write(
                                    os.path.join(root, f), os.path.join(item.name, f)
                                )
        zip_file = output_dir / f"{output_name}.zip"
        if zip_file.stat().st_size > chunk_size:
            Util.split_zip(zip_file, output_dir, chunk_size)
            if delete:
                zip_file.unlink()

    @staticmethod
    def combine_split_zip(input_dir: str, basename: str, output_dir: str = "./"):
        """分割したzipファイルを結合

        Args:
            input_dir (str): 入力ファイル群のパス
            basename (str): 入力zipファイルのファイル名
            output_dir (str, optional): 出力先のパス. Defaults to "./".
        """
        temp_buffer = b""
        for filename in sorted(Path(input_dir).glob(f"{basename}.zip.[0-9][0-9][0-9]")):
            if filename.is_file():
                with open(filename, "rb") as chunk_file:
                    temp_buffer += chunk_file.read()

        with (Path(output_dir) / f"{basename}.zip").open("wb") as output_file:
            output_file.write(temp_buffer)

    @staticmethod
    def combine_and_unpack(
        input_dir: str,
        basename: str = None,
        output_dir: str = None,
        delete: bool = True,
    ):
        """分割zipファイルを分割して展開

        Args:
            input_dir (str): 入力ファイルのパス
            basename (str): zipファイルのファイル名. Defaults to None.
            output_dir (str, optional): 出力先のパス Noneなら入力ファイルのパスに出力.
                Defaults to None.
            delete (bool): 結合したzipファイルを展開後削除するか. Defaults to True.
        """
        input_dir: Path = Path(input_dir)
        if output_dir is None:
            output_dir = input_dir
        if basename is not None:
            zip_file = input_dir / f"{basename}.zip"
            is_combine = False
            if list(input_dir.glob(f"{basename}.zip.[0-9][0-9][0-9]")):
                Util.combine_split_zip(input_dir, basename, output_dir)
                is_combine = True
            if zip_file.exists():
                shutil.unpack_archive(zip_file, output_dir)
                if is_combine and delete:
                    zip_file.unlink()
        else:
            is_combine = {}
            for file in input_dir.glob("*.zip.[0-9][0-9][0-9]"):
                basename = file.name.rsplit(2)[0]
                Util.combine_split_zip(input_dir, basename, output_dir)
                is_combine[basename] = True
            for file in input_dir.glob("*.zip"):
                basename = file.stem
                shutil.unpack_archive(file, output_dir)
                if is_combine.get(basename) and delete:
                    file.unlink()

    def dump_pond_map(csv_folder, save_name):
        csv_folder: Path = Path(csv_folder)
        assert csv_folder.is_dir()
        pond_fileds = {}
        for csv_file in sorted(csv_folder.glob("*.csv")):
            if "inv" in csv_file.stem:
                continue

            name = csv_file.stem
            size = Game.FIELD_MAX
            board = np.full((size, size), -1, dtype=np.int8)

            with csv_file.open() as f:
                reader = csv.reader(f)
                for y, row in enumerate(reader):
                    for x, item in enumerate(row):
                        if item == "1":
                            board[y, x] = 1
                        else:
                            board[y, x] = 0
            pond_fileds[name] = board
        pickle.dump(pond_fileds, (csv_folder / f"{save_name}.pkl").open("wb"))
