import gymnasium as gym; from typing import Optional; import numpy as np
from gymnasium.utils.seeding import np_random
from gymnasium.envs.registration import EnvSpec
import math
# from hyphi_gym.envs.common import Point

# Rewards 
GOAL, STEP, FAIL = 1/2, -1, -1/2
SOLUTION_THRESHOLD = 1.2

class Base(gym.Env):
  """ Base Env Implementing 
  • Trucated Episodes (via `max_episode_steps`)
  • Sparse Rewards (defaults to False)
  • Detailed Rewards (defaults to False)
  • Reward Threshold for Early Stopping 
  • Exploration Mode, where both the target and the reward are removed. 
  • Seeding nondeterministic environemtn configureation
  • Generating a dynamic spec obejct and env name based on the configuration"""
  random:list; _name: str

  def __init__(self, max_episode_steps=100, sparse=False, detailed=False, explore=False, seed:Optional[int]=None):
    max_path = (self.size-1)**2/2-2; max_steps = math.ceil(max_path * self.step_scale * SOLUTION_THRESHOLD / 100) * 100
    self.max_episode_steps, self.dynamic_spec, self._spec = max_steps, ['nondeterministic', 'max_episode_steps'], {}
    self.sparse, self.detailed, self.explore, self.nondeterministic = sparse, detailed, explore, len(self.random) > 0; self.seed(seed)
    if len(self.random): assert self.np_random is not None, "Please provide a seed to use nondeterministic features"

  @property
  def name(self)->str: 
    """Generates the dynamic environent name"""
    return ''.join([self._name, *[n.capitalize() for n in ['explore', 'sparse', 'detailed'] if getattr(self,n)], *self.random])

  @property
  def spec(self)->EnvSpec: 
    """Getter function to generate the dynmaic environment spec"""
    return EnvSpec(**{**self._spec, **{k:getattr(self,k) for k in self.dynamic_spec}})

  @spec.setter
  def spec(self, spec: dict): 
    """Function to mutate the internal environment spec (e.g., for adapting max_episode_steps)"""
    self._spec = {k:v for k,v in spec.__dict__.items() if k not in ['namespace','name','version']}

  def seed(self, seed:Optional[int]=None): self.np_random, self._seed = np_random(seed)

  def reset(self, **kwargs)->tuple[gym.spaces.Space, dict]:
    """Gymnasium compliant function to reset the environment""" 
    self.reward_buffer, self.termination_resaons = [], []; 
    return super().reset(**kwargs)
  
  def _reward_range(self, board:Optional[np.ndarray]=None):
    """Given a `board` layout, calculates the min and max returns"""
    if self.detailed: return (0, self.max_episode_steps)
    optimal_path = self._validate(board) * self.step_scale if board is not None else 0
    min_return = self.max_episode_steps * STEP + ('Holes' in self._name) * FAIL * self.max_episode_steps
    max_return = self.max_episode_steps * GOAL + optimal_path * STEP
    self.reward_threshold = self.max_episode_steps * GOAL + 1.2 * optimal_path * STEP
    return (min_return, max_return)
  
  def execute(self, action:gym.spaces.Space) -> tuple[gym.spaces.Space, dict]: 
    """Overwrite this function to perfom step mutaions on the actual environment."""
    raise(NotImplementedError)
  
  def step(self, action:gym.spaces.Space) -> tuple[gym.spaces.Space, float, bool, bool, dict]:
    """Gymnasium compliant fucntion to step the environment with `action` using the internal `_step`"""
    # Step the environment 
    state, info = self.execute(action)

    # Calculate the reward 
    terminated = 'termination_reason' in info
    if self.explore: reward = 0; terminated = False
    elif self.detailed: 
      assert 'distance' in info, "Target distance information needed for detailed reward calulation"
      reward = np.exp(-info['distance'])
    else: 
      reward = STEP
      if terminated: reward += self.max_episode_steps * (GOAL if info['termination_reason'] == 'GOAL' else FAIL)
    self.reward_buffer.append(reward)
    
    truncated = self.max_episode_steps is not None and len(self.reward_buffer) >= self.max_episode_steps
    if truncated: info = {'termination_reason': 'TIME', **info}
    if self.sparse: reward = 0 if not (terminated or truncated) else sum(self.reward_buffer)
    return state, reward, terminated, truncated, info
