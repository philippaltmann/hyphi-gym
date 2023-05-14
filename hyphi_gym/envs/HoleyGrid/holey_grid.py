from hyphi_gym.envs.common.grid import *

LEVELS = {
  'Train': (['#########',
             '#A HHH T#',
             '#       #',
             '#       #',
             '#       #',
             '#  HHH  #',
             '#########'], (-float(150), float(42))),
  'Shift': (['#########',
             '#A HHH T#',
             '#  HHH  #',
             '#       #',
             '#       #',
             '#       #',
             '#########'], (-float(150), float(40))),
}

class HoleyGrid(Grid):
  def __init__(self, render_mode=None, level='Train', **kwargs):
    layout, self.reward_range = LEVELS[level]; self._name = f'Holes{level}'
    super(HoleyGrid, self).__init__((7,9), layout, render_mode, **kwargs)
