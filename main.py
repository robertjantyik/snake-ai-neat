import pygame
import math
import random
import neat
import os
import numpy
import pickle
import argparse

parser = argparse.ArgumentParser(
    prog='main',
    description='Snake AI with NEAT')
parser.add_argument('-m', '--mode', type=str, default='train',
                    help='Run mode. Possible values: train, best. Default value is train.')
parser.add_argument('-v', '--video', type=bool, default=False,
                    help='Switch rendering the game. Default value is False.')
parser.add_argument('-c', '--checkpoint', type=str, default='',
                    help='Name of the checkpoint file to start from.')
parser.add_argument('-g', '--generations', type=int,
                    default=1000, help='Number of generations to run. Default value is 1000.')
parser.add_argument('-d', '--debug', type=bool,
                    default=False, help='Debug mode. Default value is False.')
parser.add_argument('-p', '--num-cores', type=int, default=None,
                    help='Number of CPU cores to use. Default is None, which uses all available cores.')
args = parser.parse_args()

WIDTH = 600
HEIGHT = 600
SCALE = 20


def softmax(x):
    e_x = numpy.exp(x - numpy.max(x))
    return e_x / e_x.sum(axis=0)


class Food:
    def pick_food_location(self, player, rnd):
        cols = math.floor(WIDTH / SCALE)
        rows = math.floor(HEIGHT / SCALE)
        x = math.floor(rnd.randint(0, cols)) * SCALE
        y = math.floor(rnd.randint(0, rows)) * SCALE

        while (x == player.x and y == player.y) or any(tail.x == x and tail.y == y for tail in player.tail) or (
                x >= WIDTH or y >= HEIGHT):
            x = math.floor(rnd.randint(0, cols)) * SCALE
            y = math.floor(rnd.randint(0, rows)) * SCALE

        self.x = x
        self.y = y

    def __init__(self, player, rnd):
        self.y = None
        self.x = None
        self.pick_food_location(player, rnd)

    def show(self, display):
        pygame.draw.rect(display, (255, 0, 0), (self.x, self.y, SCALE, SCALE))


class Tail:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Snake:
    def __init__(self, x, y):
        self.fitness = 0
        self.x = x
        self.y = y
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

    def show(self, display):
        for i in range(len(self.tail)):
            pygame.draw.rect(display, (255, 255, 255),
                             (self.tail[i].x, self.tail[i].y, SCALE, SCALE))
        pygame.draw.rect(display, (255, 255, 255),
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


class Game:
    def calculate_inputs(self, player, food, display, font):
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

        if display is not None and args.debug:
            for x in range(free_right):
                pygame.draw.rect(display, (0, 255, 0),
                                 (x * SCALE + player.x + SCALE, player.y, SCALE, SCALE))
            for x in range(free_left):
                pygame.draw.rect(display, (0, 255, 0),
                                 (player.x - x * SCALE - SCALE, player.y, SCALE, SCALE))
            for y in range(free_down):
                pygame.draw.rect(display, (0, 255, 0),
                                 (player.x, y * SCALE + player.y + SCALE, SCALE, SCALE))
            for y in range(free_up):
                pygame.draw.rect(display, (0, 255, 0),
                                 (player.x, player.y - y * SCALE - SCALE, SCALE, SCALE))

            text1 = font.render(f"Snake's head is at: {math.floor(player.x / SCALE)}, {math.floor(player.y / SCALE)}",
                                True, (255, 255, 255))
            text2 = font.render(f"Food is at: {math.floor(food.x / SCALE)}, {math.floor(food.y / SCALE)}", True,
                                (255, 255, 255))
            text3 = font.render(f"Distance to food is: {food_distances[0]}, {food_distances[1]}", True, (255, 255, 255))
            text4 = font.render(f"Free blocks to the right: {free_blocks[0]}", True, (255, 255, 255))
            text5 = font.render(f"Free blocks to the left: {free_blocks[1]}", True, (255, 255, 255))
            text6 = font.render(f"Free blocks downwards: {free_blocks[2]}", True, (255, 255, 255))
            text7 = font.render(f"Free blocks upwards: {free_blocks[3]}", True, (255, 255, 255))
            display.blit(text1, (0, 320))
            display.blit(text2, (0, 340))
            display.blit(text3, (0, 360))
            display.blit(text4, (0, 380))
            display.blit(text5, (0, 400))
            display.blit(text6, (0, 420))
            display.blit(text7, (0, 440))
        return inputs

    def eval_output(self, player, output, display, font):
        softmax_output = softmax(output)

        action_index = numpy.argmax(softmax_output)

        if display != None and args.debug:
            direction = "right" if action_index == 0 else "left" if action_index == 1 else "up" if action_index == 2 else "down" if action_index == 3 else ""
            text1 = font.render(f"Next direction: {direction}", True, (255, 255, 255))
            display.blit(text1, (0, 460))
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

    def test_ai(self, genome, config):
        fps = 60
        rnd = random.Random()
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        net.num_inputs = config.genome_config.num_inputs
        player = Snake(300, 300)
        food = Food(player, rnd)
        food.x = 500
        food.y = 300
        pygame.init()
        clock = pygame.time.Clock()
        display = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Snake AI with NEAT")
        font = pygame.font.SysFont('Aria', 20)
        text = font.render(f"Score: {player.total}", True, (255, 255, 255))
        text2 = font.render(f"fitness: {player.fitness}", True, (255, 255, 255))
        run = True
        while run:
            clock.tick(fps)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    genome.fitness = player.fitness
                    run = False
                    break
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_F3:
                        args.debug = not args.debug
                    if event.key == pygame.K_F1:
                        if fps == 60:
                            fps = 1
                        else:
                            fps = 60
                    if event.key == pygame.K_F2:
                        pygame.time.wait(10000)
            display.fill((0, 0, 0))

            inputs = self.calculate_inputs(player, food, display, font)

            output = net.activate(inputs)

            self.eval_output(player, output, display, font)

            if player.eat(food):
                food.pick_food_location(player, rnd)
                player.inputs_made = 0
                text = font.render(
                    f"Score: {player.total}", True, (255, 255, 255))
            if player.death():
                genome.fitness = player.fitness
                run = False
                break

            player.update()
            player.show(display)
            food.show(display)
            display.blit(text, (0, 0))
            display.blit(text2, (0, 300))
            pygame.display.update()

            player.calculate_fitness()
            text2 = font.render(f"fitness: {player.fitness}", True, (255, 255, 255))
            if player.fitness <= -10:
                genome.fitness = player.fitness
                run = False
                break
        pygame.quit()

    def eval_genomes(self, genomes, config):
        fps = 60
        for genome_id, genome in genomes:
            rnd = random.Random(123)
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            net.num_inputs = config.genome_config.num_inputs
            player = Snake(300, 300)
            food = Food(player, rnd)
            food.x = 500
            food.y = 300
            if args.video:
                pygame.init()
                clock = pygame.time.Clock()
                display = pygame.display.set_mode((WIDTH, HEIGHT))
                pygame.display.set_caption("Snake AI with NEAT")
                font = pygame.font.SysFont('Aria', 20)
                text = font.render(
                    f"Score:{player.total}", True, (255, 255, 255))
                text2 = font.render(
                    f"fitness: {player.fitness}", True, (255, 255, 255))
            run = True
            while run:
                if args.video:
                    clock.tick(fps)
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            genome.fitness = player.fitness
                            run = False
                            break
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_F3:
                                args.debug = not args.debug
                            if event.key == pygame.K_F1:
                                if fps == 60:
                                    fps = 1
                                else:
                                    fps = 60
                            if event.key == pygame.K_F2:
                                pygame.time.wait(10000)
                    display.fill((0, 0, 0))
                if args.video:
                    inputs = self.calculate_inputs(player, food, display, font)
                else:
                    inputs = self.calculate_inputs(player, food, None, None)

                output = net.activate(inputs)
                if args.video:
                    self.eval_output(player, output, display, font)
                else:
                    self.eval_output(player, output, None, None)

                if player.eat(food):
                    food.pick_food_location(player, rnd)
                    player.inputs_made = 0
                    if args.video:
                        text = font.render(
                            f"Score: {player.total}", True, (255, 255, 255))
                if player.death():
                    genome.fitness = player.fitness
                    run = False
                    break

                player.update()
                if args.video:
                    player.show(display)
                    food.show(display)
                    display.blit(text, (0, 0))
                    display.blit(text2, (0, 300))
                    pygame.display.update()

                player.calculate_fitness()
                if args.video:
                    text2 = font.render(
                        f"fitness: {player.fitness}", True, (255, 255, 255))
                if player.fitness <= -10:
                    genome.fitness = player.fitness
                    run = False
                    break
            if args.video:
                pygame.quit()


def run_neat(config):
    checkpoint_prefix = 'checkpoint'

    if os.path.exists(f'{args.checkpoint}'):
        population = neat.Checkpointer.restore_checkpoint(f'{args.checkpoint}')
    else:
        population = neat.Population(config)

    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())
    population.add_reporter(neat.Checkpointer(
        generation_interval=100, filename_prefix=checkpoint_prefix))

    population.config = config

    game = Game()

    winner = population.run(game.eval_genomes, args.generations)
    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)


def run_neat_paralell(config):
    checkpoint_prefix = 'checkpoint'

    if os.path.exists(f'{args.checkpoint}'):
        population = neat.Checkpointer.restore_checkpoint(f'{args.checkpoint}')
    else:
        population = neat.Population(config)

    game = Game()

    evaluator = neat.ParallelEvaluator(args.num_cores, game.eval_genomes)
    population.evaluator = evaluator

    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())
    population.add_reporter(neat.Checkpointer(
        generation_interval=100, filename_prefix=checkpoint_prefix))

    population.config = config

    winner = population.run(game.eval_genomes, args.generations)
    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)


def test_ai(config):
    with open("best.pickle", "rb") as f:
        winner = pickle.load(f)
    game = Game()
    game.test_ai(winner, config)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "neat_config.txt")
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)
    if args.mode == 'train':
        #run_neat(config)
        run_neat_paralell(config)
        test_ai(config)
    if args.mode == 'best':
        test_ai(config)
