""" MuJoCo-Based Physics Simulation Base Class insipired by:
- 'Gymnasium Robotics' by Rodrigo de Lazcano, Kallinteris Andreas, Jun Jet Tai, Seungjae Ryan Lee, Jordan Terry (https://github.com/Farama-Foundation/Gymnasium-Robotics)
- 'D4RL: Datasets for Deep Data-Driven Reinforcement Learning' by Justin Fu, Aviral Kumar, Ofir Nachum, George Tucker, Sergey
Levine. (https://github.com/Farama-Foundation/D4RL)

Original Code adapted to integrate with hyphi maze generation and env registration
"""

from typing import Optional, Union; import numpy as np

try:
  from mujoco import MjData as MujocoData                           # type: ignore
  from mujoco import MjModel as MujocoModel                         # type: ignore
  from mujoco import mj_resetData as mujoco_reset                   # type: ignore
  from mujoco import mj_forward as mujoco_forward                   # type: ignore
  from mujoco import mj_step as mujoco_step                         # type: ignore
  from mujoco import mj_rnePostConstraint as mujocoPostConstraint   # type: ignore
  from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
  from hyphi_gym.utils.mujoco_utils import MujocoModelNames
except ImportError as e: 
  import gymnasium
  raise gymnasium.error.DependencyNotInstalled( f"{e}. (HINT: you need to install mujoco, run `pip install gymnasium[mujoco]`.)")

class Simulation: 
  """Mujoco Based Simulation Base Class adapted from gymnasium MujocoEnv for target-based hyphi envs"""

  base_xml: str; default_cam_config: dict
  data:Optional[MujocoData] = None
  model:Optional[MujocoModel] = None
  model_path:Optional[str] = None
  metadata = { "render_modes": [ "2D", "3D" ], "render_fps": 1000, "render_resolution": (720,720) }

  def __init__(self, render_mode: Optional[str] = None, frame_skip=1, position_noise=.0):
    """Init mujoco simulation using `frame_skip` and optionaly `grid` mode. For state stochasticity
      use `position_noise`. To generate a model, supply core via `self.base_xml`. If `self.layout` 
    is not set upon init, use `setup_world(layout)` and `load_world()` once available. """
    self.frame_skip = frame_skip; self.render_mode = render_mode; self.position_noise = position_noise; 
    self.width, self.height = self.metadata['render_resolution']; self._target_active = False
    self.setup_world(); self.load_world() 

  def setup_world(self): 
    """Overwrite to generate world setup from config"""
    pass
  
  def load_world(self):
    """Helper function to load a generated world from `self.model_path` falling back to `self.base_xml`"""
    self.model = MujocoModel.from_xml_path(self.model_path or self.base_xml); self.data = MujocoData(self.model)
    self.model.vis.global_.offwidth, self.model.vis.global_.offheight = self.width, self.height
    self.init_qpos, self.init_qvel = self.data.qpos.ravel().copy(), self.data.qvel.ravel().copy()
    self.mujoco_renderer = MujocoRenderer(self.model, self.data, default_cam_config=self.default_cam_config)
    self.target_site_id = MujocoModelNames(self.model).site_name2id["target"]
    self.metadata["render_fps"] = int(np.round(1.0 / self.model.opt.timestep * self.frame_skip))

  def _toggle_target(self, active:Optional[bool]=None): 
    """Toggles activity of the target site, can be forced using `active`"""
    # TODO: if continue_task -> respawn 
    assert self.model is not None; self._target_active = self._target_active if active is None else active
    self.model.site_pos[self.target_site_id] = self.target - np.array([0,0,1-self._target_active])
    self.set_state()

  def reset_world(self):
    """Reset simulation and reposition agent and target to respective `i_pos`"""
    assert self.model is not None; mujoco_reset(self.model, self.data); HEIGHT = 1.0 #TODO
    self.target = np.append(self._noisy(self.i_tpos), HEIGHT/2)
    self._toggle_target(True)
    self.set_state(*self.agent)
  
  def render(self) -> Optional[np.ndarray]: return self.mujoco_renderer.render('rgb_array')

  def close(self): self.mujoco_renderer.close()

  def set_state(self, qpos:Optional[np.ndarray]=None, qvel:Optional[np.ndarray]=None):
    """Set the position `qpos` and velocity `qvel` of the agent."""
    assert self.data is not None and self.model is not None, "No model loaded"
    if qpos is not None: self.data.qpos[:2]=qpos.copy()
    if qvel is not None: self.data.qvel[:2]=qvel.copy()
    if self.model and self.model.na == 0: self.data.act[:] = None
    mujoco_forward(self.model, self.data)

  def do_simulation(self, ctrl:np.ndarray):
    """Step the simulation applying a `ctrl` action `self.frame_skip`-times"""
    # Check control input is contained in the action space
    assert self.data is not None, "No model loaded"
    if np.array(ctrl).shape != self.action_space.shape:
      raise ValueError(f"Action dimension mismatch. Expected {self.action_space.shape}, found {np.array(ctrl).shape}")
    self.data.ctrl[:] = ctrl; mujoco_step(self.model, self.data, nstep=self.frame_skip)
    mujocoPostConstraint(self.model, self.data)

  @property
  def agent(self)->tuple[np.ndarray, np.ndarray]: 
    """Get noisy initial agent position and velocity"""
    return self._noisy(self.i_apos), self._noisy(self.i_avel)

  def _pos(self, idx: Union[np.ndarray, tuple]) -> np.ndarray:
    """Converts a cell index `(i,j)` to x and y position in the MuJoCo simulation"""
    a = np.array; swap = a((-1,1)); return (((a(idx) + 0.5) * swap)[::-1] + a(self.size[::-1])/2 * swap ) # type: ignore
  
  def _noisy(self, pos: np.ndarray) -> np.ndarray:
    """Pass an x,y coordinate and it will return the same coordinate with uniform noise added"""
    return pos + self.np_random.uniform(-self.position_noise, self.position_noise, pos.shape)
