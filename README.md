# About
Just a basic Snake game and using the neat-python package to make an AI that learns to play.
Training is still in progress, checkpoints will be uploaded occasionally.

## Fitness function

```python
def calculate_fitness(self):
    effiency_bonus = max(0, 1 - (self.inputs_made / 100))
    self.fitness = self.total + effiency_bonus - (self.inputs_made * 0.1)
```

# Dependencies
* **[Pygame](https://github.com/pygame/)**: Used in game development and user interaction
* **[neat-python](https://github.com/CodeReclaimers/neat-python)**: Python implementation of the NEAT neuroevolution algorithm

# Usage

## Install dependencies

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Controls

- F1: Switch between 1 and 60 fps
- F2: Pause the game for 10 seconds
- F3: Switch debug mode
- F4: Disable fps lock

## Launch options

- -m, --mode: Run mode. Possible values: train, best. Default value is train.
- -v, --video: Switch rendering the game. Default value is False.
- -c, --checkpoint: Name of the checkpoint file to start from.
- -g, --generations: Number of generations to run. Default value is 1000.
- -d, --debug: Debug mode. Default value is False.
- -p, --num-cores: Number of CPU cores to use. Default is None, which uses all available cores.

# Help
Use -h or --help to see the optional launch parameters.