from hyphi_gym.common import Maze, Point

class PointMaze(Point, Maze): # Point(Mujoco) | Maze(Board(Base))
  def __init__(self, render_mode=None, **kwargs):
    Maze.__init__(self, prefix='Point', **kwargs)
    Point.__init__(self, render_mode=render_mode)
