import pygame, math,random, neat, os, numpy, pickle, argparse

parser = argparse.ArgumentParser(
    prog='main',
    description='Snake AI with NEAT')
parser.add_argument('-m', '--mode', type=str, default='train', help='Run mode. Possible values: train, best')
parser.add_argument('-v', '--video', type=bool, default=False, help='Switch rendering the game.')
parser.add_argument('-c', '--checkpoint', type=str, default='checkpoint1359', help='Name of the checkpoint file to start from.')
parser.add_argument('-g', '--generations', type=int, default=1000, help='Number of generations to run.')
args = parser.parse_args()

WIDTH = 600
HEIGHT = 600
SCALE = 20
FPS = 60

def softmax(x):
    e_x = numpy.exp(x - numpy.max(x))
    return e_x / e_x.sum(axis=0)

class Food:
    def pick_food_location(self, player, rnd):
        cols = math.floor(WIDTH / SCALE)
        rows = math.floor(HEIGHT / SCALE)
        x = math.floor(rnd.randint(0, cols)) * SCALE
        y = math.floor(rnd.randint(0, rows)) * SCALE

        while (x == player.x and y == player.y) or any(tail.x == x and tail.y == y for tail in player.tail) or (x >= WIDTH or y >= HEIGHT):
            x = math.floor(rnd.randint(0, cols)) * SCALE
            y = math.floor(rnd.randint(0, rows)) * SCALE

        self.x = x
        self.y = y
    
    def __init__(self, player, rnd):
        self.pick_food_location(player, rnd)
    
    def show(self, display):
        pygame.draw.rect(display, (255, 0, 0), (self.x, self.y, SCALE, SCALE))

class Tail:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Snake:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tail = []
        self.total = 0
        self.speedx = 1
        self.speedy = 0
    
    
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
            pygame.draw.rect(display, (255, 255, 255), (self.tail[i].x, self.tail[i].y, SCALE, SCALE))    
        pygame.draw.rect(display, (255, 255, 255), (self.x, self.y, SCALE, SCALE))
    
    def eat(self, food):
        if self.x == food.x and self.y == food.y:
            self.total += 1
            return True
        else:
            return False
    
    def death(self, fitness):
        for i in range(len(self.tail)):
            pos = self.tail[i]
            if self.x == pos.x and self.y == pos.y:
                #print(f"tail hit at: {pos.x}, {pos.y}")
                fitness -= 1
                return True
        if self.x >= WIDTH or self.x < 0:
            fitness -= 0.1
            return True
        if self.y >= HEIGHT or self.y < 0:
            fitness -= 0.1
            return True
        return False
            

class Game:
    def test_ai(self, genome, config):
        rnd = random.Random(12366)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        net.num_inputs = config.genome_config.num_inputs
        player = Snake(300, 300)
        food = Food(player, rnd)
        food.x = 500
        food.y = 300
        fitness = 0
        inputs_made = 0
        pygame.init()
        clock = pygame.time.Clock()
        display = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Snake AI with NEAT")
        font = pygame.font.SysFont('Aria', 50)
        text = font.render(f"Score: {player.total}", True, (255, 255, 255))
        run = True
        while run:
            clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    fitness -= 0.1
                    genome.fitness = fitness
                    run = False
                    break
            display.fill((0, 0, 0))
            directions = [(1, 0), (-1, 0), (0, -1), (0, 1)]

            food_distances = [((food.x - player.x) / WIDTH * SCALE, (food.y - player.y) / HEIGHT * SCALE) for dx, dy in directions]

            current_direction = (player.speedx, player.speedy)
            movement_indicator = [1 if current_direction == d else 0 for d in directions]

            if len(player.tail) > 0:
                last_tail = player.tail[-1]
            else:
                last_tail = Tail(player.x, player.y)
            
            distance_to_tail = (
                ((player.x - last_tail.x) / WIDTH * SCALE),
                ((player.y - last_tail.y) / HEIGHT * SCALE)
            )
            
            inputs = (
                player.x / WIDTH * SCALE,
                player.y / HEIGHT * SCALE,
                *food_distances[0],
                *food_distances[1],
                *food_distances[2],
                *food_distances[3],
                *movement_indicator,
                *distance_to_tail
            )
            
            output = net.activate(inputs)
            
            softmax_output = softmax(output)
            action_index = numpy.argmax(softmax_output)
            
            if action_index == 0:
                player.speedx = 1
                player.speedy = 0
                inputs_made += 1
            if action_index == 1:
                player.speedx = -1
                player.speedy = 0
                inputs_made += 1
            if action_index == 2:
                player.speedx = 0
                player.speedy = -1
                inputs_made += 1
            if action_index == 3:
                player.speedx = 0
                player.speedy = 1
                inputs_made += 1
            if action_index == 4:
                # do nothing
                pass
            if player.eat(food):
                food.pick_food_location(player, rnd)
                inputs_made = 0
                fitness += 1
                font = pygame.font.SysFont('Aria', 50)
                text = font.render(f"Score: {player.total}", True, (255, 255, 255))
            if player.death(fitness):
                genome.fitness = fitness
                run = False
                break
            
            player.update()
            player.show(display)
            food.show(display)
            display.blit(text, (20, 0))
            pygame.display.update()
            
            fitness = player.total - (inputs_made * 0.1)
            
            if inputs_made > 100:
                fitness -= 1
                genome.fitness = fitness
                run = False
                break
        pygame.quit()
    
    def eval_genomes(self, genomes, config):
        for genome_id, genome in genomes:
            rnd = random.Random(12366)
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            net.num_inputs = config.genome_config.num_inputs
            player = Snake(300, 300)
            food = Food(player, rnd)
            food.x = 500
            food.y = 300
            fitness = 0
            inputs_made = 0
            if args.video:
                pygame.init()
                clock = pygame.time.Clock()
                display = pygame.display.set_mode((WIDTH, HEIGHT))
                pygame.display.set_caption("Snake AI with NEAT")
                font = pygame.font.SysFont('freesanbold.ttf', 50)
                text = font.render(f"Score:{player.total}", True, (255, 255, 255))
            run = True
            while run:
                if args.video:
                    clock.tick(FPS)
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            fitness -= 0.1
                            genome.fitness = fitness
                            run = False
                            break
                    display.fill((0, 0, 0))
                directions = [(1, 0), (-1, 0), (0, -1), (0, 1)]

                food_distances = [((food.x - player.x) / WIDTH * SCALE, (food.y - player.y) / HEIGHT * SCALE) for dx, dy in directions]

                current_direction = (player.speedx, player.speedy)
                movement_indicator = [1 if current_direction == d else 0 for d in directions]

                if len(player.tail) > 0:
                    last_tail = player.tail[-1]
                else:
                    last_tail = Tail(player.x, player.y)
                
                distance_to_tail = (
                    ((player.x - last_tail.x) / WIDTH * SCALE),
                    ((player.y - last_tail.y) / HEIGHT * SCALE)
                )
                
                inputs = (
                    player.x / WIDTH * SCALE,
                    player.y / HEIGHT * SCALE,
                    *food_distances[0],
                    *food_distances[1],
                    *food_distances[2],
                    *food_distances[3],
                    *movement_indicator,
                    *distance_to_tail
                )
                
                output = net.activate(inputs)
                
                softmax_output = softmax(output)
                action_index = numpy.argmax(softmax_output)
                
                if action_index == 0:
                    player.speedx = 1
                    player.speedy = 0
                    inputs_made += 1
                if action_index == 1:
                    player.speedx = -1
                    player.speedy = 0
                    inputs_made += 1
                if action_index == 2:
                    player.speedx = 0
                    player.speedy = -1
                    inputs_made += 1
                if action_index == 3:
                    player.speedx = 0
                    player.speedy = 1
                    inputs_made += 1
                if action_index == 4:
                    # do nothing
                    pass
                if player.eat(food):
                    food.pick_food_location(player, rnd)
                    inputs_made = 0
                    fitness += 1
                    if args.video:
                        font = pygame.font.SysFont('Aria', 50)
                        text = font.render(f"Score: {player.total}", True, (255, 255, 255))
                if player.death(fitness):
                    genome.fitness = fitness
                    run = False
                    break
                
                player.update()
                if args.video:
                    player.show(display)
                    food.show(display)
                    display.blit(text, (20, 0))
                    pygame.display.update()
                
                fitness = player.total - (inputs_made * 0.1)
                
                if inputs_made > 100:
                    fitness -= 1
                    genome.fitness = fitness
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
    population.add_reporter(neat.Checkpointer(filename_prefix=checkpoint_prefix))

    game = Game()
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
        run_neat(config)
        test_ai(config)
    if args.mode == 'best':
        test_ai(config)
