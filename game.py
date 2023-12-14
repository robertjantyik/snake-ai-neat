import math
import random

import neat
import numpy
import pygame

from food import Food
from game_settings import *
from snake import Snake


def softmax(x):
    e_x = numpy.exp(x - numpy.max(x))
    return e_x / e_x.sum(axis=0)


class Game:
    def __init__(self, video: bool, debug: bool):
        self.video = video
        self.debug = debug
        self.rnd = random.Random(RANDOM_SEED)
        if video:
            pygame.init()
            self.display = pygame.display.set_mode((WIDTH, HEIGHT))
            self.font = pygame.font.SysFont('Aria', 20)
            self.fps = 60

    def calculate_inputs(self, player: Snake, food: Food):
        food_distances = [
            (math.floor((food.x - player.x) / SCALE)),
            (math.floor((food.y - player.y) / SCALE))
        ]

        free_right = player.calculate_free_blocks_right()
        free_left = player.calculate_free_blocks_left()
        free_down = player.calculate_free_blocks_down()
        free_up = player.calculate_free_blocks_up()

        free_blocks = (
            free_right,
            free_left,
            free_down,
            free_up
        )

        inputs = (
            math.floor(player.x / SCALE),
            math.floor(player.y / SCALE),
            *food_distances,
            *free_blocks
        )

        if self.video and self.debug:
            for x in range(free_right):
                pygame.draw.rect(self.display, (0, 255, 0),
                                 (x * SCALE + player.x + SCALE, player.y, SCALE, SCALE))
            for x in range(free_left):
                pygame.draw.rect(self.display, (0, 255, 0),
                                 (player.x - x * SCALE - SCALE, player.y, SCALE, SCALE))
            for y in range(free_down):
                pygame.draw.rect(self.display, (0, 255, 0),
                                 (player.x, y * SCALE + player.y + SCALE, SCALE, SCALE))
            for y in range(free_up):
                pygame.draw.rect(self.display, (0, 255, 0),
                                 (player.x, player.y - y * SCALE - SCALE, SCALE, SCALE))

            text1 = self.font.render(
                f"Snake's head is at: {math.floor(player.x / SCALE)}, {math.floor(player.y / SCALE)}",
                True, (255, 255, 255))
            text2 = self.font.render(f"Food is at: {math.floor(food.x / SCALE)}, {math.floor(food.y / SCALE)}", True,
                                     (255, 255, 255))
            text3 = self.font.render(f"Distance to food is: {food_distances[0]}, {food_distances[1]}", True,
                                     (255, 255, 255))
            text4 = self.font.render(f"Free blocks to the right: {free_blocks[0]}", True, (255, 255, 255))
            text5 = self.font.render(f"Free blocks to the left: {free_blocks[1]}", True, (255, 255, 255))
            text6 = self.font.render(f"Free blocks downwards: {free_blocks[2]}", True, (255, 255, 255))
            text7 = self.font.render(f"Free blocks upwards: {free_blocks[3]}", True, (255, 255, 255))
            self.display.blit(text1, (0, 320))
            self.display.blit(text2, (0, 340))
            self.display.blit(text3, (0, 360))
            self.display.blit(text4, (0, 380))
            self.display.blit(text5, (0, 400))
            self.display.blit(text6, (0, 420))
            self.display.blit(text7, (0, 440))
        return inputs

    def eval_output(self, player: Snake, output):
        softmax_output = softmax(output)

        action_index = numpy.argmax(softmax_output)

        if self.video and self.debug:
            direction = "right" if action_index == 0 else "left" if action_index == 1 else "up" if action_index == 2 else "down" if action_index == 3 else ""
            text1 = self.font.render(f"Next direction: {direction}", True, (255, 255, 255))
            self.display.blit(text1, (0, 460))
        if action_index == 0:  # go right
            if player.speedx != 1:
                player.inputs_made += 1
            player.speedx = 1
            player.speedy = 0
        if action_index == 1:  # go left
            if player.speedx != -1:
                player.inputs_made += 1
            player.speedx = -1
            player.speedy = 0
        if action_index == 2:  # go up
            if player.speedy != -1:
                player.inputs_made += 1
            player.speedx = 0
            player.speedy = -1
        if action_index == 3:  # go down
            if player.speedy != 1:
                player.inputs_made += 1
            player.speedx = 0
            player.speedy = 1

    def test_ai(self, genome, cfg):
        net = neat.nn.FeedForwardNetwork.create(genome, cfg)
        net.num_inputs = cfg.genome_config.num_inputs
        player = Snake(self)
        food = Food(player, self.rnd, self)
        food.x = 500
        food.y = 300
        clock = pygame.time.Clock()
        pygame.display.set_caption("Snake AI with NEAT")
        text = self.font.render(f"Score: {player.total}", True, (255, 255, 255))
        text2 = self.font.render(f"fitness: {player.fitness}", True, (255, 255, 255))
        run = True
        while run:
            clock.tick(self.fps)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    genome.fitness = player.fitness
                    run = False
                    break
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_F3:
                        self.debug = not self.debug
                    if event.key == pygame.K_F1:
                        if self.fps == 60:
                            self.fps = 1
                        else:
                            self.fps = 60
                    if event.key == pygame.K_F4:
                        self.fps = 0
                    if event.key == pygame.K_F2:
                        pygame.time.wait(10000)
            self.display.fill((0, 0, 0))

            inputs = self.calculate_inputs(player, food)

            output = net.activate(inputs)

            self.eval_output(player, output)

            if player.eat(food):
                food.pick_food_location(self.rnd)
                player.inputs_made = 0
                text = self.font.render(
                    f"Score: {player.total}", True, (255, 255, 255))
            if player.death():
                genome.fitness = player.fitness
                run = False
                break

            player.update()
            player.show()
            food.show()
            self.display.blit(text, (0, 0))
            self.display.blit(text2, (0, 300))
            pygame.display.update()

            player.calculate_fitness()
            text2 = self.font.render(f"fitness: {player.fitness}", True, (255, 255, 255))
            if player.fitness <= -10:
                genome.fitness = player.fitness
                run = False
                break
        pygame.quit()

    def eval_genomes(self, genomes, cfg):
        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, cfg)
            net.num_inputs = cfg.genome_config.num_inputs
            player = Snake(self)
            food = Food(player, self.rnd, self)
            food.x = 500
            food.y = 300
            if self.video:
                clock = pygame.time.Clock()
                pygame.display.set_caption("Snake AI with NEAT")
                text = self.font.render(
                    f"Score:{player.total}", True, (255, 255, 255))
                text2 = self.font.render(
                    f"fitness: {player.fitness}", True, (255, 255, 255))
            run = True
            while run:
                if self.video:
                    clock.tick(self.fps)
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            genome.fitness = player.fitness
                            run = False
                            break
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_F3:
                                self.debug = not self.debug
                            if event.key == pygame.K_F1:
                                if self.fps == 60:
                                    self.fps = 1
                                else:
                                    self.fps = 60
                            if event.key == pygame.K_F4:
                                self.fps = 0
                            if event.key == pygame.K_F2:
                                pygame.time.wait(10000)
                    self.display.fill((0, 0, 0))

                inputs = self.calculate_inputs(player, food)

                output = net.activate(inputs)

                self.eval_output(player, output)

                if player.eat(food):
                    food.pick_food_location(self.rnd)
                    player.inputs_made = 0
                    if self.video:
                        text = self.font.render(
                            f"Score: {player.total}", True, (255, 255, 255))
                if player.death():
                    genome.fitness = player.fitness
                    run = False
                    break

                player.update()
                if self.video:
                    player.show()
                    food.show()
                    self.display.blit(text, (0, 0))
                    self.display.blit(text2, (0, 300))
                    pygame.display.update()

                player.calculate_fitness()
                if self.video:
                    text2 = self.font.render(
                        f"fitness: {player.fitness}", True, (255, 255, 255))
                if player.fitness <= -10:
                    genome.fitness = player.fitness
                    run = False
                    break
            if self.video:
                pygame.quit()
