from hyphi_gym.envs.common.simulation import Simulation
from os import path
from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np
from gymnasium import spaces



class Point(Simulation):
  base_xml = path.join(path.dirname(path.realpath(__file__)), "../../assets/point.xml")
  _target_proximity = 0.3 # 0.45  # Proximity to target for solution 

  metadata = { "render_modes": [ "2D", "3D" ], "render_fps": 100, "render_resolution": (720,720) }

  def __init__(self, render_mode: Optional[str] = None):
    # TODO:? reward_type=reward_type,  continuing_task=continuing_task,
    super().__init__(position_noise=0.2); assert self.model is not None; self.render_mode = render_mode
    self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64)
    # goal_obs = spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float64)
    # observation_space = spaces.Dict(dict(observation=observation_space, achieved_goal=goal_obs, desired_goal=goal_obs))

    # Set Action Space 
    bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
    self.action_space = spaces.Box(low=bounds.T[0], high=bounds.T[1], dtype=np.float32)
    self.action_space.seed(self._seed)
  # TODO: step done by base env -> implement 
  def _step(self, action):
    action = np.clip(action, -1.0, 1.0) # Clip Velocity (can grow unbounded / ball is force actuated)
    self.set_state(self.data.qpos, np.clip(self.data.qvel, -5.0, 5.0)); self.do_simulation(action)

    obs = self.state_vector(); info = {}; distance = np.linalg.norm(obs[:2] - self.target[:2])
    # Observation: agent position and velocity, agent: agent position only (TODO: needed?) target: target position 
    # TODO: test observing only agent pos and vel first before using dict \w goal info
    # dict_obs = { "observation": obs.copy(), "agent": agent.copy(), "target": self.target.copy()}
    reward, info["success"] = np.exp(-np.linalg.norm(distance)), bool(distance <= self._target_proximity)
    return obs, reward, info["success"], False, info
  
  def reset(self, **kwargs):
    assert self.np_random is not None, "Seeding is required"
    self.board = self._board(self.layout)
    super().reset(**kwargs); self.reset_world()
    return self.state_vector(), {}

  def render(self): return self.mujoco_renderer.render('rgb_array') if self.render_mode else None
