from hyphi_gym.common import Grid, Maze

class GridMaze(Maze,Grid):
  def __init__(self, render_mode=None, **kwargs):
    Maze.__init__(self, **kwargs)
    Grid.__init__(self, render_mode=render_mode)
