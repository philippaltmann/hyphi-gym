from hyphi_gym.envs.common.board import *

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
NUM_HOLES = 6

class Holes(Board):
  """Gridworld Maze Environment based on hyphi Grid.
  :param level: Configuration to use [Train|Shift]
  :param random: optional list of features to be stochastic
    supporting layout, agent-, and target-placement"""
  def __init__(self, level, random=[], **kwargs): 
    self._name = f'Holes{level}'; layout = None if 'Layouts' in random else LEVELS[level]
    Board.__init__(self, size=(7,9), layout=layout, random=random, RADD=['Layouts'], **kwargs)

  def _generate(self):
    """"""
    board = self._grid(['#########', '#A     T#', '#       #', '#       #', '#       #', '#       #', '#########']) 
    holes = 0 
    while holes < NUM_HOLES:
      pos = tuple(self.np_random.integers(low=(1,1),high=([s-2 for s in self.size]),size=(2,)))
      if board[pos] == CELLS[FIELD]: board[pos] = CELLS[HOLE]; holes += 1
    return board 