from hyphi_gym.common.grid import Grid
from hyphi_gym.common.holes import Holes

class HoleyGrid(Holes, Grid):
  def __init__(self, level='Train', render_mode=None, **kwargs):
    Holes.__init__(self, level, max_episode_steps=100, **kwargs)
    Grid.__init__(self, render_mode=render_mode)
