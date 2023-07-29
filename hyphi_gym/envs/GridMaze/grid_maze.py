from hyphi_gym.envs.common import Grid, Maze

class GridMaze(Maze,Grid):
  def __init__(self, render_mode=None, **kwargs):
    Maze.__init__(self, max_episode_steps=100, **kwargs)
    Grid.__init__(self, render_mode=render_mode)
