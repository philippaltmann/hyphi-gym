""" MuJoCo-Based Fetch (https://fetchrobotics.com) environment insipired by:
- 'Gymnasium Robotics' by Rodrigo de Lazcano, Kallinteris Andreas, Jun Jet Tai, Seungjae Ryan Lee, Jordan Terry (https://github.com/Farama-Foundation/Gymnasium-Robotics)
- 'Meta-World: A Benchmark and Evaluation for Multi-Task and Meta Reinforcement Learning' by Tianhe Yu∗1, Deirdre Quillen, Zhanpeng He, Ryan Julian, Avnish Narayan, Hayden Shively, Adithya Bellathur, Karol Hausman, Chelsea Finn, Sergey Levine. (https://github.com/Farama-Foundation/Metaworld)
- 'D4RL: Datasets for Deep Data-Driven Reinforcement Learning' by Justin Fu, Aviral Kumar, Ofir Nachum, George Tucker, Sergey Levine. (https://github.com/Farama-Foundation/D4RL)
Args:
  agent (numpy array): base position of the agent gripper arm
  block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
  continue_task (bool): whether to spawn a new target or reset the agent ore upon episode completion
  distance_threshold (float): the threshold after which a goal is considered achieved
  has_object (boolean): whether or not the environment has an object
  target (numpy array): base position of the target 
  target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
  target_noise (float): range of a uniform distribution for sampling a target
  frame_skip (int): number of substeps the simulation runs on every call to step (prev: n_substeps)
  position_noise (float): range of a uniform distribution for sampling initial object positions (prev: obj_range)
  render_mode (str)
"""

from typing import Optional; import numpy as np; import gymnasium as gym
from hyphi_gym.envs.common.simulation import Simulation
from hyphi_gym.envs.common import Base

class Robot(Base, Simulation):
  """Continous-control robot base class"""
  step_scale = 10
  metadata = {"render_modes": ["3D"], "render_resolution": (720,720)}   
  default_cam_config = {"distance": 2, "azimuth": 135, "elevation": -16, "lookat": np.array([1, 0.85, 0.85])}

  def __init__(self, agent: Optional[np.ndarray] = np.array([1,1,1]), block_gripper: bool = False,
               continue_task: bool = True, distance_threshold: float = 0.05, has_object: bool = False, 
               target: Optional[np.ndarray] = np.array([1,1,1]), target_in_the_air: bool = True, target_noise: float = 0.25,
               frame_skip: int = 20, position_noise: float = 0.25, render_mode = None, **kwargs):

    self.agent = agent; self.block_gripper = block_gripper; self.continue_task = continue_task
    self.distance_threshold = distance_threshold; self.has_object = has_object; self.target = target
    self.target_in_the_air = target_in_the_air; self.target_noise = target_noise; Base.__init__(self, **kwargs)
    Simulation.__init__(self, render_mode=render_mode, position_noise=position_noise, frame_skip=frame_skip) 
    self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(10+15*self.has_object,), dtype=np.float64)
    self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype="float32"); self.action_space.seed(self._seed)

  def load_world(self):
    """Helper function to load a generated world from `self.model_path` falling back to `self.base_xml`"""
    super().load_world(); initial = {"robot0:slide0": 0.1, "robot0:slide1": 0.73, "robot0:slide2": 0.375}
    if self.has_object: initial = {**initial, "object0:joint": [1.25, 0.53, 0.4, 1, 0, 0, 0],}
    self._set_pos(initial); self._position_mocap() # Move end effector into position
    if self.has_object: self.height_offset = self._get_pos("object0")[0][2]
    self.set_world()

  def _validate(self, layout, error=True, setup=False): return 0

  def _generate(self)->np.ndarray:
    """Random generator function for a layout of self.specs"""
    if self.has_object: assert False, "TOOD"
    else: return self.tpos

  # Helper functions to manage mocap, target and object positioning 
  def _position_mocap(self, pos=None, rot=[1.0, 0.0, 1.0, 0.0]):
    pos = getattr(self, '_agent', pos if pos is not None else self._noisy(self.agent))
    if not self.continue_task and 'Agents' not in self.random: self._agent = pos
    self._set_mocap(pos, rot, "robot0:mocap"); [self.do_simulation() for _ in range(10)]

  @property
  def tpos(self):
    target = getattr(self, '_target', self._noisy(self.target, self.target_noise))
    if not self.continue_task and 'Targets' not in self.random: self._target = target
    self.model.site_pos[self.target_id] = target; self._forward(); return target
      
  def _randomize(self, layout:np.ndarray, key:str):
    """Mutation function to randomize the position of `key` in `layout`"""
    if 'Target' in key: return # Handled via tpos 
    if 'Agent' in key: return self._position_mocap()
    assert False, f'{key} ranomization not supported'

  def state_vector(self) -> np.ndarray: #-> np.ndarray:
    """Return the position and velocity joint states of the model"""
    grip_pos, grip_velp = self._get_pos("robot0:grip")

    robot_qpos, robot_qvel = self._robot_obs
    gripper_state, gripper_vel = robot_qpos[-2:], (robot_qvel[-2:] * self.dt)
    obs = np.concatenate([grip_pos, grip_velp, gripper_state, gripper_vel])

    if self.has_object: # rotations, velocities, and gripper state
      _pos, _velp = self._get_pos("object0")
      _rot, _velr = self._get_rot("object0")
      _rel_pos = (_pos - grip_pos).ravel(); _velp -= grip_velp.ravel()
      obs = np.concatenate([obs, _pos, _rot, _velp, _velr, _rel_pos])
    
    agent = np.squeeze(_pos.copy()) if self.has_object else grip_pos.copy()
    target = self._get_pos("target")[0]
    return {'obs': obs, 'target': target, 'agent': agent}
  

  def execute(self, action:np.ndarray) -> tuple[dict, dict]:
    """Executes the action, returns the new state, info, and distance between agent and target
    Action should be 4d array containing x,y,z, and gripper displacement in [-1,1]"""    
    if np.array(action).shape != self.action_space.shape: raise ValueError("Action dimension mismatch")
    action = np.clip(action.copy(), self.action_space.low, self.action_space.high)
    pos_ctrl, gripper_ctrl = action[:3] * 0.05, np.array([action[-1], action[-1]])
    rot_ctrl = [1.0, 0.0, 1.0, 0.0] # fixed rotation of the end effector, expressed as a quaternion
    if self.block_gripper: gripper_ctrl = np.zeros_like(gripper_ctrl)

    # Apply action to simulation.
    self.do_simulation(np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl]))
    state = self.state_vector(); d = np.linalg.norm(state['target'] - state['agent'], axis=-1)
    info = {'distance': d}
    if (d < self.distance_threshold): info = {**info, 'termination_reason':'GOAL'}
    return state['obs'], info

  # Gym API
  def render(self): return self.mujoco_renderer.render('rgb_array')
  
  def reset(self, **kwargs)->tuple[gym.spaces.Space, dict]:
    """Reset the environment simulation and randomize if needed"""
    if not self.continue_task: self.reset_world()
    super().reset(**kwargs)
    state = self.state_vector(); d = np.linalg.norm(state['target'] - state['agent'], axis=-1)
    info = {'distance': d}
    if (d < self.distance_threshold): info = {**info, 'termination_reason':'GOAL'}
    return state['obs'], info
