from os import path; from typing import Optional
import numpy as np; from gymnasium import spaces
from hyphi_gym.envs.common.simulation import Simulation

AGENT_SIZE = 0.3

class Point(Simulation):
  step_scale = 10  # Used for calculating max_episode_steps according to grid size
  base_xml = path.join(path.dirname(path.realpath(__file__)), "../../assets/point.xml")
  metadata = { "render_modes": [ "2D", "3D" ], "render_fps": 1000, "render_resolution": (720,720) }

  def __init__(self, render_mode: Optional[str] = None):
    self.render_mode = render_mode; super().__init__(position_noise=0.2); assert self.model is not None; 
    self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64)
    self.observation_space = spaces.Box(-np.inf, np.inf, shape=(2 * (4 if len(self.holes) else 3),), dtype=np.float64) 
    bounds = self.model.actuator_ctrlrange.copy().astype(np.float32) # Set Action Space 
    self.action_space = spaces.Box(low=bounds.T[0], high=bounds.T[1], dtype=np.float32)
    self.action_space.seed(self._seed); self.reward_threshold = None

  def execute(self, action:np.ndarray) -> tuple[dict, dict]:
    """Executes the action, returns the new state, info, and distance between the agent and target"""
    assert self.data is not None; action = np.clip(action, -1.0, 1.0) 
    self.set_state(None, np.clip(self.data.qvel[:2], -5.0, 5.0)); self.do_simulation(action)
    obs, info = self.state_vector(), {}; distance = np.linalg.norm(obs[4:6]); info['distance'] = distance
    if bool(distance <= 2*AGENT_SIZE): info = {**info, 'termination_reason':'GOAL'}; self._toggle_target(False)
    if self.data.qpos[2] < -AGENT_SIZE: info = {**info, 'termination_reason':'FAIL'}
    if obs.shape[0] == 8 and (obs[6:8] < 0).all(): info = {**info, 'termination_reason':'FAIL'}
    return obs, info
  
  def reset(self, **kwargs):
    """Reset the environment simulation and randomize if needed"""
    assert self.np_random is not None, "Seeding is required"
    super().reset(**kwargs); self.reset_world()
    return self.state_vector(), {}
