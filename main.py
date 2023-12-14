import argparse
import os
import pickle

import neat

from game import Game

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


def run_neat(cfg):
    checkpoint_prefix = 'checkpoint'

    if os.path.exists(f'{args.checkpoint}'):
        population = neat.Checkpointer.restore_checkpoint(f'{args.checkpoint}')
    else:
        population = neat.Population(cfg)

    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())
    population.add_reporter(neat.Checkpointer(
        generation_interval=100, filename_prefix=checkpoint_prefix))

    population.config = cfg

    game = Game(args.video, args.debug)

    winner = population.run(game.eval_genomes, args.generations)
    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)


def run_neat_paralell(cfg):
    checkpoint_prefix = 'checkpoint'

    if os.path.exists(f'{args.checkpoint}'):
        population = neat.Checkpointer.restore_checkpoint(f'{args.checkpoint}')
    else:
        population = neat.Population(cfg)

    game = Game(args.video, args.debug)

    evaluator = neat.ParallelEvaluator(args.num_cores, game.eval_genomes)
    population.evaluator = evaluator

    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())
    population.add_reporter(neat.Checkpointer(
        generation_interval=100, filename_prefix=checkpoint_prefix))

    population.config = cfg

    winner = population.run(game.eval_genomes, args.generations)
    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)


def test_ai(cfg):
    with open("best.pickle", "rb") as f:
        winner = pickle.load(f)
    game = Game(args.video, args.debug)
    game.test_ai(winner, cfg)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "neat_config.txt")
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)
    if args.mode == 'train':
        # run_neat(config)
        run_neat_paralell(config)
        test_ai(config)
    if args.mode == 'best':
        test_ai(config)
