from os import path; import tempfile
from typing import Union; import xml.etree.ElementTree as ET
import numpy as np; from gymnasium import spaces
from hyphi_gym.envs.common.simulation import Simulation
from hyphi_gym.utils.mujoco_utils import MujocoModelNames

from hyphi_gym.envs.common.board import AGENT, CELLS, FIELD, WALL, TARGET, HOLE
SIZE = 1.0; HEIGHT = 1.0; AGENT_SIZE = 0.3; 

class Point(Simulation):
  """ Base class for Continous Control in Board Games 
  Use for 3D simulation of continuous board envs and rendering of grids"""
  step_scale = 10  # Used for calculating max_episode_steps according to grid size
  base_xml = path.join(path.dirname(path.realpath(__file__)), "../../assets/point.xml")

  def __init__(self, grid=False, render_mode=None, frame_skip=1):
    self.grid = grid; self.holes = []
    
    super().__init__(render_mode=render_mode, position_noise=0.2, frame_skip=frame_skip); 
    assert self.model is not None; 
    self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64)
    self.observation_space = spaces.Box(-np.inf, np.inf, shape=(2 * (4 if len(self.holes) else 3),), dtype=np.float64) 
    bounds = self.model.actuator_ctrlrange.copy().astype(np.float32) # Set Action Space 
    self.action_space = spaces.Box(low=bounds.T[0], high=bounds.T[1], dtype=np.float32)
    self.action_space.seed(self._seed)

  def state_vector(self) -> np.ndarray: #-> np.ndarray:
    """Return the position and velocity joint states of the model"""
    assert self.data is not None, "No model loaded"
    velocity = self.data.qvel[:2]
    agent = self.data.qpos[:2]
    target = self.target[:2] - self.data.qpos[:2]
    state = np.concatenate((velocity, agent, target))

    if len(self.holes):
      hole_dist = self.holes - self.data.qpos[:2]; 
      next_hole = hole_dist[np.linalg.norm(hole_dist, axis=1).argmin()]
      hole_norm = np.clip(next_hole/np.linalg.norm(next_hole, ord=1) * 2, -1, 1) * SIZE / 2
      state = np.concatenate((state, next_hole - hole_norm)) #Normalized to compensate delta to the center of the hole

    return state

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

  @property # Set camera to top-down view for 2D rendering and isometric view for 3D rendering
  def default_cam_config(self): return {
    "azimuth": 90, "elevation": -90, 'lookat': np.array([0,0,0]),
    "distance": sum(self.size) / 4*3 } if self.render_mode == '2D' else { 
    "azimuth": 135, "elevation": -60, "distance": sum(self.size)/0.96,# / 1.125,  
    "lookat": (0.5,-0.5,1)} # np.append(np.array(self.size) / np.array([50, -50]), 0) 

  def setup_world(self, layout=None):
    """Helper function to generate a simulation from a board-based `layout`"""
    layout = self.layout if layout is None else layout
    if layout is None: return 
    
    tree = ET.parse(self.base_xml); worldbody = tree.find(".//worldbody"); assert worldbody is not None

    _str = lambda list: ' '.join(map(str,list))
    lookup = { CELLS[WALL]: WALL, CELLS[FIELD]: FIELD,
               CELLS[AGENT]: FIELD, CELLS[TARGET]: FIELD}
    holes = []
    block = lambda p,cell: ET.SubElement(
        worldbody, "geom", type="box", material=lookup[cell], 
        pos=_str([*self._pos(p), (1 if cell==CELLS[WALL] else -1)*HEIGHT/2]), 
        size=_str([SIZE/2,SIZE/2,HEIGHT/(1 if cell==CELLS[WALL] else 2)])
      ) if cell in lookup.keys() else holes.append(self._pos(p))
    [block((i,j), cell) for i,row in enumerate(layout) for j, cell in enumerate(row)]
    self.holes = np.array(holes)
    # Set initial agent position, velocity and target position, add ground
    self.i_apos, self.i_avel = self._pos(self.getpos(layout, AGENT)), np.array([0,0])
    self.i_tpos = self._pos(self.getpos(layout, TARGET))
    
    asset = tree.find(".//asset"); assert asset is not None  # Add Grid texture and floor plane
    if self.grid:
      ET.SubElement(asset, "material", name="floor", texture="grid", rgba=".4 .4 .4 .8",
                    specular="0", shininess="0", texrepeat=' '.join([str(s) for s in self.size[::-1]]))
      agent = tree.find('.//worldbody/body/body'); assert agent is not None
      for part in ['Body', 'Ears', 'Eyes', 'Hat', 'Lamp', 'Mouth', 'White']:
        ET.SubElement(asset, "mesh", file=f'{path.dirname(self.base_xml)}/Agent/{part}.obj')
        ET.SubElement(agent, "geom", mesh=part, material=part, type="mesh")

    with tempfile.TemporaryDirectory() as tmp_dir: self.model_path = path.join(path.dirname(tmp_dir), "world.xml")
    tree.write(self.model_path)  # Save new xml with maze to a temporary file
  
  def load_world(self):
    """Helper function to load a generated world from `self.model_path` falling back to `self.base_xml`"""
    super().load_world()
    if self.grid: self.agent_id = MujocoModelNames(self.model).body_name2id["Agent"]

  def update_world(self, action:int, position: Union[np.ndarray, tuple], cell: str):
    quat = np.array([([.7071068,.7071068,0,0],[-0.5,-0.5,0.5,0.5],[0,0,.7071068,.7071068],[0.5,0.5,0.5,0.5])[action]])
    assert self.model is not None and self.agent_id is not None
    self.model.body_quat[self.agent_id] = quat # Set rotation according to action
    if cell == HOLE: self.set_state(qpos=self._pos((-100,-100)))
    if cell == TARGET: self._toggle_target(False)
    if cell in [FIELD, TARGET]: self.set_state(qpos=self._pos(position))

  def reset_world(self):
    """Reset simulation and reposition agent and target to respective `i_pos`"""
    if self.layout is None: self.setup_world(self.board); self.load_world()
    super().reset_world()
  
  def _update(self, key:str, oldpos, newpos):
    if key[0] == AGENT: self.i_apos = self._pos(newpos)
    if key[0] == TARGET: self.i_tpos = self._pos(newpos)
