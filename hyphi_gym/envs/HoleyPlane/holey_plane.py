from hyphi_gym.envs.common import Holes, Point

class HoleyPlane(Point, Holes):
  def __init__(self, render_mode=None, **kwargs):
    Holes.__init__(self, max_episode_steps=400, **kwargs)
    Point.__init__(self, render_mode=render_mode)
