import pygame

from game_settings import *


class Tail:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Snake:
    def __init__(self, game):
        self.fitness = 0
        self.x = 300
        self.y = 300
        self.game = game
        self.tail = []
        self.total = 0
        self.speedx = 1
        self.speedy = 0
        self.inputs_made = 0

    def calculate_free_blocks_right(self):
        free_blocks_right = 0

        for x in range(self.x + SCALE, WIDTH, SCALE):
            for tail in self.tail:
                if tail.x == x and tail.y == self.y:
                    return free_blocks_right
            free_blocks_right += 1

        return free_blocks_right

    def calculate_free_blocks_left(self):
        free_blocks_left = 0

        for x in range(self.x - SCALE, -20, -SCALE):
            for tail in self.tail:
                if tail.x == x and tail.y == self.y:
                    return free_blocks_left
            free_blocks_left += 1

        return free_blocks_left

    def calculate_free_blocks_down(self):
        free_blocks_down = 0

        for y in range(self.y + SCALE, HEIGHT, SCALE):
            for tail in self.tail:
                if tail.y == y and tail.x == self.x:
                    return free_blocks_down
            free_blocks_down += 1

        return free_blocks_down

    def calculate_free_blocks_up(self):
        free_blocks_up = 0

        for y in range(self.y - SCALE, -20, -SCALE):
            for tail in self.tail:
                if tail.y == y and tail.x == self.x:
                    return free_blocks_up
            free_blocks_up += 1

        return free_blocks_up

    def calculate_fitness(self):
        effiency_bonus = max(0, 1 - (self.inputs_made / 100))
        self.fitness = self.total + effiency_bonus - (self.inputs_made * 0.1)

    def update(self):
        for i in range(len(self.tail) - 1):
            self.tail[i] = self.tail[i + 1]
        if self.total >= 1:
            self.tail.append(Tail(self.x, self.y))
            if len(self.tail) > self.total:
                del self.tail[0]

        self.x += self.speedx * SCALE
        self.y += self.speedy * SCALE

    def show(self):
        for i in range(len(self.tail)):
            pygame.draw.rect(self.game.display, (255, 255, 255),
                             (self.tail[i].x, self.tail[i].y, SCALE, SCALE))
        pygame.draw.rect(self.game.display, (255, 255, 255),
                         (self.x, self.y, SCALE, SCALE))

    def eat(self, food):
        if self.x == food.x and self.y == food.y:
            self.total += 1
            return True
        else:
            return False

    def death(self):
        for i in range(len(self.tail)):
            pos = self.tail[i]
            if self.x == pos.x and self.y == pos.y:
                self.fitness -= 10
                return True
        if self.x >= WIDTH or self.x < 0:
            self.fitness -= 1
            return True
        if self.y >= HEIGHT or self.y < 0:
            self.fitness -= 1
            return True
        return False
