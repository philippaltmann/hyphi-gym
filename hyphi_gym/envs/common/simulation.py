from typing import Optional, Union
from os import path; import tempfile
import xml.etree.ElementTree as ET
import numpy as np

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

from hyphi_gym.envs.common import Board

HEIGHT = 1.0

class Simulation(Board):
  """Mujoco Based Simulation Base Class adapted from gymnasium MujocoEnv for hyphi board envs
  Can be used for 3D simulation of continuous board envs and rendering of hyphi grids"""

  base_xml: str # Base path to load the model
  data:Optional[MujocoData] = None
  model:Optional[MujocoModel] = None
  model_path:Optional[str] = None

  def __init__(self, frame_skip=1, grid=False, position_noise=.0):
    """Init mujoco simulation using `frame_skip` and optionaly `grid` mode. For state stochasticity
      use `position_noise`. To generate a model, supply core via `self.base_xml`. If `self.layout` 
    is not set upon init, use `setup_world(layout)` and `load_world()` once available. """
    self.frame_skip = frame_skip; self.grid = grid; self.position_noise = position_noise
    self.width, self.height = self.metadata['render_resolution']
    if self.layout is not None: self.setup_world(self.layout)
    self.load_world() 

  def setup_world(self, layout:np.ndarray):
    """Helper function to generate a simulation from a board-based `layout`"""
    tree = ET.parse(self.base_xml); 
    worldbody = tree.find(".//worldbody"); assert worldbody is not None
    _pos = lambda p: ' '.join(map(str, np.append(self._pos(p), HEIGHT/2))); _size = ' '.join([str(HEIGHT/2)]*3)
    block = lambda p: ET.SubElement(worldbody, "geom", type="box", material="wall", pos=_pos(p), size=_size) 
    [block((i,j)) for i,row in enumerate(layout) for j, cell in enumerate(row) if cell == self.CELLS[self.WALL]]

    # Set initial agent position, velocity and target position, add ground
    self.i_apos, self.i_avel = self._pos(self.getpos(layout, self.AGENT)), np.array([0,0])
    self.i_tpos = self._pos(self.getpos(layout, self.TARGET))
    
    asset = tree.find(".//asset"); assert asset is not None  # Add Grid texture and floor plane
    if self.grid: ET.SubElement(asset, "material", name="floor", texture="grid", specular="0", shininess="0",
                                texrepeat=' '.join([str(s) for s in self.size]), rgba=".4 .4 .4 .8")
    ET.SubElement(worldbody, "geom", name="ground", material="floor", type="plane", 
                  pos="0 0 0", size=' '.join([str(s/2) for s in (*self.size,.1)]))

    with tempfile.TemporaryDirectory() as tmp_dir: self.model_path = path.join(path.dirname(tmp_dir), "world.xml")
    tree.write(self.model_path)  # Save new xml with maze to a temporary file

  def load_world(self):
    """Helper function to load a generated world from `self.model_path` falling back to `self.base_xml`"""
    # Set camera to top-down view for 2D rendering and isometric view for 3D rendering
    default_cam_config = {"azimuth": 90, "elevation": -90,
                          "distance": sum(self.size) / 1.5 } if self.render_mode == '2D' else { 
                          "azimuth": 135, "elevation": -60, "distance": sum(self.size) / 1.125,  
                          "lookat": np.append(np.array(self.size) / np.array([15, -15]), 0) } 
    # Setup model, data, renderer and link movable parts
    self.model = MujocoModel.from_xml_path(self.model_path or self.base_xml); self.data = MujocoData(self.model)
    self.model.vis.global_.offwidth, self.model.vis.global_.offheight = self.width, self.height
    self.init_qpos, self.init_qvel = self.data.qpos.ravel().copy(), self.data.qvel.ravel().copy()
    self.mujoco_renderer = MujocoRenderer(self.model, self.data, default_cam_config=default_cam_config)
    self.target_site_id = MujocoModelNames(self.model).site_name2id["target"]
    if not self.grid: self.metadata["render_fps"] = int(np.round(1.0 / self.model.opt.timestep * self.frame_skip))

  def reset_world(self):
    """Reset simulation and reposition agent and target to respective `i_pos`"""
    if self.layout is None: self.setup_world(self.board); self.load_world()
    assert self.model is not None; mujoco_reset(self.model, self.data)
    self.target = np.append(self._noisy(self.i_tpos), HEIGHT/2)
    self.model.site_pos[self.target_site_id] = self.target
    self.set_state(*self.agent)
  
  def randomize(self, cell:str, board:np.ndarray)->tuple[tuple[int],tuple[int]]:
    """Update model upon randomization"""
    oldpos, newpos = super().randomize(cell, board)
    if cell == self.AGENT: self.i_apos = self._pos(newpos)
    if cell == self.TARGET: self.i_tpos = self._pos(newpos)
    return oldpos, newpos

  ### Gym functionality ### 
  def close(self): self.mujoco_renderer.close()

  def state_vector(self) -> np.ndarray:
    """Return the position and velocity joint states of the model"""
    assert self.data is not None, "No model loaded"
    return np.concatenate([self.data.qpos, self.data.qvel]).ravel()

  def set_state(self, qpos:Optional[np.ndarray]=None, qvel:Optional[np.ndarray]=None):
    """Set the position `qpos` and velocity `qvel` of the agent."""
    assert self.data is not None and self.model is not None, "No model loaded"
    if qpos is not None: assert qpos.shape == (self.model.nq,); self.data.qpos[:]=np.copy(qpos)
    if qvel is not None: assert qvel.shape == (self.model.nq,); self.data.qvel[:]=np.copy(qvel)
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
    a = np.array; swap = a((-1,1)); return (((a(idx) + 0.5) * swap)[::-1] + a(self.size)/2 * swap)
  
  def _noisy(self, pos: np.ndarray) -> np.ndarray:
    """Pass an x,y coordinate and it will return the same coordinate with uniform noise added"""
    return pos + self.np_random.uniform(-self.position_noise, self.position_noise, pos.shape)
