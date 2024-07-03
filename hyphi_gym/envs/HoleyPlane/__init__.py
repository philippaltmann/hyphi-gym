from hyphi_gym.common.holes import Holes
from hyphi_gym.common.point import Point

class HoleyPlane(Point, Holes):
  def __init__(self, render_mode=None, **kwargs):
    Holes.__init__(self, max_episode_steps=400, **kwargs)
    Point.__init__(self, render_mode=render_mode)
