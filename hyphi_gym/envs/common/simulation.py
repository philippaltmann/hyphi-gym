""" MuJoCo-Based Physics Simulation Base Class insipired by:
- 'Gymnasium Robotics' by Rodrigo de Lazcano, Kallinteris Andreas, Jun Jet Tai, Seungjae Ryan Lee, Jordan Terry (https://github.com/Farama-Foundation/Gymnasium-Robotics)
- 'D4RL: Datasets for Deep Data-Driven Reinforcement Learning' by Justin Fu, Aviral Kumar, Ofir Nachum, George Tucker, Sergey
Levine. (https://github.com/Farama-Foundation/D4RL)

Original Code adapted to integrate with hyphi maze generation and env registration
"""

from typing import Optional, Union; import numpy as np; from os import path; import re

try:
  from mujoco import MjData as MujocoData                           # type: ignore
  from mujoco import MjModel as MujocoModel                         # type: ignore
  from mujoco import mj_resetData as mujoco_reset                   # type: ignore
  from mujoco import mj_forward as mujoco_forward                   # type: ignore
  from mujoco import mj_step as mujoco_step                         # type: ignore
  from mujoco import mj_rnePostConstraint as mujocoPostConstraint   # type: ignore
  from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
  from hyphi_gym.utils.mujoco_utils import *
except ImportError as e: 
  import gymnasium
  raise gymnasium.error.DependencyNotInstalled( f"{e}. (HINT: you need to install mujoco, run `pip install gymnasium[mujoco]`.)")

get_xml = lambda task: f"{re.sub('(?<=hyphi_gym).*', '', path.dirname(path.realpath(__file__)))}/assets/{task}.xml"

class Simulation: 
  """Mujoco Based Simulation Base Class adapted from gymnasium MujocoEnv for target-based hyphi envs"""

  base_xml: str; default_cam_config: dict
  data:Optional[MujocoData] = None
  model:Optional[MujocoModel] = None
  model_path:Optional[str] = None

  def __init__(self, render_mode: Optional[str] = None, frame_skip=1, position_noise=0):
    """Init mujoco simulation using `render_mode` and `frame_skip` to set simpulation fps.
    For state stochasticity use `position_noise`. To generate a model, supply core via `self.base_xml`.
    Extend `setup_world()` to store an optional model_path, or adapt `load_world()` for specific setup."""
    self.frame_skip = frame_skip; self.render_mode = render_mode; self.position_noise = position_noise; 
    self.width, self.height = self.metadata['render_resolution']; self._target_active = False
    self.setup_world(); self.load_world(); self.metadata["render_fps"] = int(np.round(1.0 / self.dt))

  def setup_world(self): pass
  
  def load_xml(self, path):
    model = MujocoModel.from_xml_path(path); data = MujocoData(model)
    return model, data 

  def load_world(self):
    """Helper function to load a generated world from `self.model_path` falling back to `self.base_xml`"""
    self.model, self.data = self.load_xml(self.model_path or self.base_xml)
    self.model.vis.global_.offwidth, self.model.vis.global_.offheight = self.width, self.height
    self.mujoco_renderer = MujocoRenderer(self.model, self.data, default_cam_config=self.default_cam_config)
    self.names = MujocoModelNames(self.model); self.target_id = self.names.site_name2id["target"]; self.set_world()

  def set_world(self):
    """Save current world state for reset"""
    self.initial_time = self.data.time
    self.initial_qpos = np.copy(self.data.qpos)
    self.initial_qvel = np.copy(self.data.qvel)

  def reset_world(self): 
    assert self.model is not None; mujoco_reset(self.model, self.data)
    self.data.time = self.initial_time; self.data.qpos[:] = np.copy(self.initial_qpos)
    self.data.qvel[:] = np.copy(self.initial_qvel); self._forward()
  
  def render(self) -> Optional[np.ndarray]: 
    return self.mujoco_renderer.render('rgb_array')

  def close(self): self.mujoco_renderer.close()

  @property
  def _robot_obs(self): return robot_get_obs(self.model, self.data, self.names.joint_names)
  
  def _forward(self): 
    if self.model and self.model.na == 0: self.data.act[:] = None
    mujoco_forward(self.model, self.data)

  def _get_pos(self, key): 
    """Helper to retrieve position and velocity"""
    return (get_site_xpos(self.model, self.data, key).copy().ravel(),
      (get_site_xvelp(self.model, self.data, key) * self.dt).ravel()) 
  
  def _get_rot(self, key): 
    """Helper to retrieve rotation and velocity"""
    return (mat2euler(get_site_xmat(self.model, self.data, key)).copy().ravel(),
      (get_site_xvelr(self.model, self.data, key) * self.dt).ravel())

  def _set_vel(self, qvel): 
    [set_joint_qvel(self.model, self.data, name, value) for name, value in qvel.items()]
    self._forward()

  def _set_pos(self, qpos): 
    [set_joint_qpos(self.model, self.data, name, value) for name, value in qpos.items()]; 
    reset_mocap_welds(self.model, self.data); self._forward()

  def _set_mocap(self, pos, rot, key): 
    set_mocap_pos(self.model, self.data, key, pos)
    set_mocap_quat(self.model, self.data, key, rot)
    
  def do_simulation(self, action:Optional[np.ndarray]=None):
    """Step the simulation applying a `ctrl` action `self.frame_skip`-times"""
    # Check control input is contained in the action space
    assert self.data is not None, "No model loaded"
    if action is not None: 
      ctrl_set_action(self.model, self.data, action) # Apply control action
      mocap_set_action(self.model, self.data, action) # Apply mocap control
    mujoco_step(self.model, self.data, nstep=self.frame_skip)
    mujocoPostConstraint(self.model, self.data)

  def _pos(self, idx: Union[np.ndarray, tuple]) -> np.ndarray:
    """Converts a cell index `(i,j)` to x and y position in the MuJoCo simulation"""
    a = np.array; swap = a((-1,1)); return (((a(idx) + 0.5) * swap)[::-1] + a(self.size[::-1])/2 * swap ) # type: ignore
  
  def _noisy(self, pos: np.ndarray, noise=None) -> np.ndarray:
    """Pass an x,y coordinate and it will return the same coordinate with uniform noise added"""
    return pos + self.np_random.uniform(-(noise or self.position_noise), (noise or self.position_noise), pos.shape)

  @property # Return the timestep of each Gymanisum step.
  def dt(self): return self.frame_skip * self.model.opt.timestep
