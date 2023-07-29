from hyphi_gym.envs.common import Maze, Point

class PointMaze(Point, Maze): # Point(Mujoco) | Maze(Board(Base))
  def __init__(self, render_mode=None, **kwargs):
    Maze.__init__(self, max_episode_steps=800, **kwargs) # Init Maze
    Point.__init__(self, render_mode=render_mode)
