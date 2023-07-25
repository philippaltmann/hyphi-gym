from os import path; from typing import Optional
import numpy as np; from gymnasium import spaces
from hyphi_gym.envs.common.simulation import Simulation

AGENT_SIZE = 0.3

class Point(Simulation):
  base_xml = path.join(path.dirname(path.realpath(__file__)), "../../assets/point.xml")
  metadata = { "render_modes": [ "2D", "3D" ], "render_fps": 100, "render_resolution": (720,720) }

  def __init__(self, render_mode: Optional[str] = None):
    self.render_mode = render_mode; super().__init__(position_noise=0.2); assert self.model is not None; 
    self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64)
    bounds = self.model.actuator_ctrlrange.copy().astype(np.float32) # Set Action Space 
    self.action_space = spaces.Box(low=bounds.T[0], high=bounds.T[1], dtype=np.float32)
    self.action_space.seed(self._seed)
    
  def _step(self, action:np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]: 
    """Helper function to execute `action` returning its consequences.
    Point velocity is clipped (can grow unbounded / ball is force actuated)"""
    assert self.data is not None; action = np.clip(action, -1.0, 1.0) 
    self.set_state(None, np.clip(self.data.qvel[:2], -5.0, 5.0)); self.do_simulation(action)
    obs = self.state_vector(); info = {}; distance = np.linalg.norm(obs[:2] - self.target[:2])
    reward = np.exp(-np.linalg.norm(distance))
    if bool(distance <= AGENT_SIZE): info = {**info, 'termination_reason':'GOAL'}; self._toggle_target(False)
    if self.data.qpos[2] < -AGENT_SIZE: info = {**info, 'termination_reason':'FAIL'}
    return obs, reward, 'termination_reason' in info.keys(), False, info
  
  def reset(self, **kwargs):
    """Reset the environment simulation and randomize if needed"""
    assert self.np_random is not None, "Seeding is required"
    self.board = self._board(self.layout)
    super().reset(**kwargs); self.reset_world()
    return self.state_vector(), {}
