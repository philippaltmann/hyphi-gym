from typing import Any, List, Optional, SupportsFloat
import gymnasium as gym; 
from gymnasium.utils.seeding import np_random
from gymnasium.envs.registration import EnvSpec

class Base(gym.Env):
  """ Base Env Implementing 
  • Trucated Episodes (via `max_episode_steps`)
  • Sparse Rewards (defaults to False)
  • Reward Threshold for Early Stopping 
  • Exploration Mode, where both the target and the reward are removed. 
  • Seeding nondeterministic environemtn configureation
  • Generating a dynamic spec obejct and env name based on the configuration"""
  random: List; termination_resaons:List; _name: str

  def __init__(self, max_episode_steps=100, sparse=False, explore=False, seed:Optional[int]=None):
    self.max_episode_steps, self.reward_range = max_episode_steps, (-max_episode_steps, -max_episode_steps)
    self.dynamic_spec, self._spec = ['reward_threshold', 'nondeterministic', 'max_episode_steps'], {}
    self.sparse, self.explore, self.nondeterministic = sparse, explore, len(self.random) > 0; self.seed(seed)
    if len(self.random): assert self.np_random is not None, "Please provide a seed to use nondeterministic features"

  @property
  def reward_threshold(self)->float: 
    """Gets the current layout's reward threshold"""
    return round(.95 * self.reward_range[1])
  
  @property
  def name(self)->str: 
    """Generates the dynamic environent name"""
    return ''.join([self._name, *[n.capitalize() for n in ['explore', 'sparse'] if getattr(self,n)], *self.random])

  @property
  def spec(self)->EnvSpec: 
    """Getter function to generate the dynmaic environment spec"""
    return EnvSpec(**{**self._spec, **{k:getattr(self,k) for k in self.dynamic_spec}})

  @spec.setter
  def spec(self, spec: dict[str,Any]): 
    """Function to mutate the internal environment spec (e.g., for adapting max_episode_steps)"""
    self._spec = {k:v for k,v in spec.__dict__.items() if k not in ['namespace','name','version']}

  def seed(self, seed:Optional[int]=None): self.np_random, self._seed = np_random(seed)

  def reset(self, **kwargs)->tuple[gym.spaces.Space, dict[str, Any]]:
    """Gymnasium compliant function to reset the environment""" 
    self.reward_buffer, self.termination_resaons = [], []; 
    return super().reset(**kwargs)

  def _step(self, action:gym.spaces.Space) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]: 
    """Overwrite this function to perfom step mutaions on the actual environment"""
    raise(NotImplementedError)
  
  def step(self, action:gym.spaces.Space) -> tuple[gym.spaces.Space, SupportsFloat, bool, bool, dict[str, Any]]:
    """Gymnasium compliant fucntion to step the environment with `action` using the internal `_step`"""
    state, reward, terminated, truncated, info = self._step(action); self.reward_buffer.append(reward)
    if self.explore: reward = 0
    truncated = self.max_episode_steps is not None and len(self.reward_buffer) >= self.max_episode_steps
    if truncated: info = {'termination_reason': 'TIME', **info}; self.termination_resaons.append('TIME')
    if self.sparse: reward = 0 if not (terminated or truncated) else sum(self.reward_buffer)
    return state, reward, terminated, truncated, info
