import gymnasium as gym ; import numpy as np
from typing import Optional; from os import path

from hyphi_gym.common.board import *
from hyphi_gym.common.rendering import Rendering
from hyphi_gym.common.point import Point

class Grid(Rendering, Point): 
  step_scale = 1  # Used for calculating max_episode_steps according to grid size
  base_xml = path.join(path.dirname(path.realpath(__file__)), "../assets/grid.xml")

  metadata = {"render_modes": ["2D", "3D", "blender"], "render_fps": 5, "render_resolution": (720,720)} 
  def __init__(self, render_mode:Optional[str]=None, **simargs):
    assert render_mode is None or render_mode in self.metadata["render_modes"]; self.render_mode = render_mode
    if render_mode is not None: 
      self.renderer = Rendering if render_mode == 'blender' else Point
      rargs = dict(grid=True, render_mode=render_mode, frame_skip=20)
      self.renderer.__init__(self, **({} if render_mode == 'blender' else rargs))
    self.observation_space = gym.spaces.Box(low=min(CELLS.values()), high=max(CELLS.values()), shape=(np.prod(self.size),), dtype=np.int64, seed=self._seed)
    self.action_space = gym.spaces.Discrete(n=4, seed=self._seed)
        
  def render(self) -> Optional[np.ndarray]:
    """Return rendering of current state as np array if render_mode set"""
    if self.render_mode not in self.metadata['render_modes']: return 
    return self.renderer.render(self)

  """Gym API functions"""
  def reset(self, **kwargs)-> tuple[np.ndarray,dict]:
    observation, info = super().reset(**kwargs)
    if self.render_mode is not None: self.renderer.reset_world(self)
    return observation, info

  def _distance(self): return np.linalg.norm(self.getpos(self.board) - self.tpos)

  def execute(self, action: int) -> tuple[np.ndarray, dict]:
    """Helper function to step the environment, executing `action`, returning its consequences"""
    position = self.getpos(self.board); target = self.newpos(position, action);     
    field, info = CHARS[self.board[target]], {'distance': self._distance()}
    if field == TARGET: info = {**info, 'termination_reason':'GOAL'}; 
    if field == HOLE: info = {**info, 'termination_reason':'FAIL'}
    revert = CELLS[TARGET] if all(np.equal(position, self.tpos)) else CELLS[FIELD] 
    if field is not WALL: self.board[tuple(position)] = revert      # Move Agent 
    if field in [FIELD, TARGET]: self.board[target] = CELLS[AGENT]  # Update Board 
    if self.render_mode is not None: self.renderer.update_world(self, action, target, field)
    return self.board.flatten(), info
    