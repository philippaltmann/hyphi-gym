from hyphi_gym.common.board import *

LEVELS = {
  'Train':  ['#########',
             '#A HHH T#',
             '#       #',
             '#       #',
             '#       #',
             '#  HHH  #',
             '#########'],
  'Shift':  ['#########',
             '#A HHH T#',
             '#  HHH  #',
             '#       #',
             '#       #',
             '#       #',
             '#########'],
}

class Holes(Board):
  """Gridworld Maze Environment based on hyphi Grid.
  :param level: Configuration to use [Train|Shift]
  :param random: optional list of features to be stochastic supporting layout, agent-, and target-placement"""
  def __init__(self, level:str, random=[], **kwargs): 
    self._name = f'Holes{level}'; #layout = None if 'Layouts' in random else LEVELS[level]
    layout, size, radd = (LEVELS[level], (7,9), []) if level in LEVELS else (None, (int(level),int(level)), ['Layouts'])
    Board.__init__(self, size=size, layout=layout, random=random, RADD=radd, can_fail=True, **kwargs)

  def _generate(self):
    """Random generator for holey grids, placing mean(size) holes"""
    board = super()._generate(); holes = 0 
    while holes < sum(self.size)/len(self.size):
      pos = tuple(self.np_random.integers(low=(1,1),high=([s-1 for s in self.size]),size=(2,)))
      if board[pos] == CELLS[FIELD]: board[pos] = CELLS[HOLE]; holes += 1
    if self._validate(board, error=False) > self.max_episode_steps: return self._generate()
    return board 
  