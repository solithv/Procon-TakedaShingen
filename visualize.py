import csv
import glob
import pygame
import numpy as np
from MyEnv.game import Game

class Renderer:
    CELL_SIZE = 50

    imageScaler = np.array((CELL_SIZE, CELL_SIZE))
    blankImage = pygame.transform.scale(
        pygame.image.load("./assets/blank.png"), imageScaler
    )
    pondImage = pygame.transform.scale(
        pygame.image.load("./assets/pond.png"), imageScaler
    )
    castleImage = pygame.transform.scale(
        pygame.image.load("./assets/castle.png"), imageScaler
    )
    workerAImage = pygame.transform.scale(
        pygame.image.load("./assets/worker_A.png"), imageScaler
    )
    workerBImage = pygame.transform.scale(
        pygame.image.load("./assets/worker_B.png"), imageScaler
    )

    def __init__(self, fieldWidth, fieldHeight):
        pygame.init()
        self.fieldWidth = fieldWidth
        self.fieldHeight = fieldHeight
        self.windowSurface = pygame.display.set_mode(
            (self.CELL_SIZE * self.fieldWidth, self.CELL_SIZE * self.fieldHeight)
        )

    def drawGrids(self):
        for i in range(1, self.fieldWidth):
            pygame.draw.line(
                self.windowSurface,
                (0, 0, 0),
                (i * self.CELL_SIZE, 0),
                (i * self.CELL_SIZE, self.CELL_SIZE * self.fieldHeight),
                1,
            )
        for i in range(1, self.fieldHeight):
            pygame.draw.line(
                self.windowSurface,
                (0, 0, 0),
                (0, i * self.CELL_SIZE),
                (self.CELL_SIZE * self.fieldWidth, i * self.CELL_SIZE),
                1,
            )

    def placeImage(self, img, i, j):
        placement = (j * self.CELL_SIZE, i * self.CELL_SIZE)
        img = pygame.transform.scale(img, self.imageScaler)
        self.windowSurface.blit(img, placement)
        if j == 0 or i == 0:
            font = pygame.font.SysFont(None, 30)
            text = font.render(str(i if j == 0 else j), False, (0, 0, 0))
            text_rect = text.get_rect(
                center=(j * self.CELL_SIZE + 10, i * self.CELL_SIZE + 10)
            )
            self.windowSurface.blit(text, text_rect)

    def captureField(self, field, fileName):
        for i in range(self.fieldHeight):
            for j in range(self.fieldWidth):
                cellInfo = field[i][j]
                self.placeImage(self.blankImage, i, j)

                if "0" in cellInfo:
                    pass
                elif "1" in cellInfo:
                    self.placeImage(self.pondImage, i, j)
                elif "2" in cellInfo:
                    self.placeImage(self.castleImage, i, j)
                elif "b" in cellInfo:
                    self.placeImage(self.workerBImage, i, j)
                elif "a" in cellInfo:
                    self.placeImage(self.workerAImage, i, j)

        self.drawGrids()
        pygame.image.save(self.windowSurface, fileName)


if __name__ == "__main__":
    fieldPaths = glob.glob("./field_data/???.csv")
    for fieldPath in fieldPaths:
        # print(fieldPath)
        # g = Game()
        # g.load_from_csv(path=fieldPath)
        # print(g.board[g.CELL.index("pond")])
        with open(fieldPath) as f:
            field = [row for row in csv.reader(f)]
            renderer = Renderer(fieldWidth=len(field[0]), fieldHeight=len(field))
            renderer.captureField(
                field,
                fileName="./field_data/field_visualized/" + fieldPath[-7:-4] + ".png",
            )
