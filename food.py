import math

import pygame

from game_settings import *
from snake import Snake


class Food:
    def pick_food_location(self, rnd):
        cols = math.floor(WIDTH / SCALE)
        rows = math.floor(HEIGHT / SCALE)
        x = math.floor(rnd.randint(0, cols)) * SCALE
        y = math.floor(rnd.randint(0, rows)) * SCALE

        while (x == self.player.x and y == self.player.y) or any(
                tail.x == x and tail.y == y for tail in self.player.tail) or (
                x >= WIDTH or y >= HEIGHT):
            x = math.floor(rnd.randint(0, cols)) * SCALE
            y = math.floor(rnd.randint(0, rows)) * SCALE

        self.x = x
        self.y = y

    def __init__(self, player: Snake, rnd, game):
        self.y = None
        self.x = None
        self.game = game
        self.player = player
        self.pick_food_location(rnd)

    def show(self):
        pygame.draw.rect(self.game.display, (255, 0, 0), (self.x, self.y, SCALE, SCALE))
