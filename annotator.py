import glob
import os

from Utils import Annotator


def main():
    output_dir = "./dataset"
    csv_dir = "./field_data"
    filename = "data.dat"
    # random: ランダム
    # smart: 強ランダム
    # human: 手動
    enemy = "smart"
    # 最大ターン数を指定
    # 数値入力で指定可能
    # Noneでマップサイズに応じて可変
    max_steps = None
    annotator = Annotator(
        glob.glob(os.path.join(csv_dir, "A17.csv")),
        output_dir,
        filename,
        size=7,
        max_steps=max_steps,
        use_pyautogui=True,
    )
    for _ in range(1):
        annotator.reset()
        annotator.play_game_annotator(enemy)
        # annotator.do_annotate()
    annotator.finish()


if __name__ == "__main__":
    main()
