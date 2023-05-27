import tkinter as Tk
import numpy as np
import copy
import random


class GameField:
    def __init__(self):
        self.CELL = {
            "black": True,
            "position_A": False,
            "position_B": False,
            "open_position_A": False,
            "open_position_B": False,
            "rampart_A": False,
            "rampart_B": False,
            "castle": False,
            "pond": False,
            "worker_A": False,
            "worker_B": False,
        }
        self.WORKER = {"action": True, "x": None, "y": None}
        self.point_multiplier = {"castle": 100, "camp": 50, "rampart": 10}
        self.field = [
            copy.deepcopy(
                [copy.deepcopy(self.CELL) for _ in range(random.randint(11, 25))]
            )
            for _ in range(random.randint(11, 25))
        ]

    def setup(self):
        cells_count = len(self.field) * len(self.field[0])
        used = []

        def put_element(element):
            while True:
                select_cell = random.randint(0, cells_count - 1)

        self.pieces_count = random.randint(2, 6)
        self.pieces_A = list(
            map(
                self.set_team,
                [(copy.deepcopy(self.WORKER), "A") for _ in range(self.pieces_count)],
            )
        )
        self.pieces_B = list(
            map(
                self.set_team,
                [(copy.deepcopy(self.WORKER), "B") for _ in range(self.pieces_count)],
            )
        )
        print(self.pieces_A)
        print(self.pieces_B)

    def set_team(self, params):
        piece, color = params
        piece["team"] = color
        return piece

    def pint_collector(self):
        pass

    def update_self_cells(self):
        pass


if __name__ == "__main__":
    f = GameField()
    print(f.field)
    # print(len(f.field), "*", len(f.field[0]))
    # print(f.pieces_A)
