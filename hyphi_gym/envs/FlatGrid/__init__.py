from hyphi_gym.common.grid import Grid
from hyphi_gym.common.board import Board

class FlatGrid(Grid, Board):
  def __init__(self, size, random=[], render_mode=None, **kwargs):
    self._name = f'FlatGrid{size}' 
    Board.__init__(self, size=(size,size), layout=None, random=random, max_episode_steps=100, **kwargs)
    Grid.__init__(self, render_mode=render_mode)
