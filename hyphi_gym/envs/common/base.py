from typing import Any, List, SupportsFloat
import gymnasium as gym 
from gymnasium.envs.registration import EnvSpec
from gymnasium.utils import seeding


STOP_PERCENTAGE = 0.95
TARGET_REWARD, STEP_COST, HOLE_COST = 50, -1, -50
WALL, FIELD, AGENT, TARGET, HOLE = '#', ' ', 'A', 'T', 'H'
CELL_LOOKUP = [WALL, FIELD, AGENT, TARGET, HOLE]
CELL_SIZE = 64
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3; ACTIONS = [UP, RIGHT, DOWN, LEFT]
RAND_KEY = ['Agent', 'Target']; RAND_KEYS = ['Agents', 'Targets']; RAND = ['Layouts', *RAND_KEY, *RAND_KEYS]

class Base(gym.Env):
  """ Base Env Implementing 
  • Sparse Rewards (defaults to False)
  • Early Stopping on reward threshold (defaults to False) TODO: MOVE TO WRAPPER?
  • Trucated Episodes (via `max_episode_steps`)"""
  random: List; termination_resaons:List; _name: str

  def __init__(self, max_episode_steps=100, sparse=False, explore=False, seed=None): #stop=False,
    self.dynamic_spec = ['reward_threshold', 'nondeterministic', 'max_episode_steps']
    self._spec, self.max_episode_steps, self.sparse, self.explore = {}, max_episode_steps, sparse, explore 
    self.nondeterministic = len(self.random) > 0; self._np_random, seed = seeding.np_random(seed)

  @property #if stop else None TODO: handle stopping via training param
  def reward_threshold(self): return round(.95 * self.reward_range[1])
  
  @property
  def name(self): return ''.join([self._name, *[n.capitalize() for n in ['explore', 'sparse'] if getattr(self,n)], *self.random])

  @property
  def spec(self): return EnvSpec(**{**self._spec, **{k:getattr(self,k) for k in self.dynamic_spec}})

  @spec.setter
  def spec(self, spec): self._spec = {k:v for k,v in spec.__dict__.items() if k not in ['namespace','name','version']}

  def reset(self, **kwargs): 
    self.reward_buffer, self.termination_resaons = [], []; 
    return super().reset(**kwargs)

  def _step(self, action:Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
    raise(NotImplementedError)
  
  def step(self, action:Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
    state, reward, terminated, truncated, info = self._step(action); self.reward_buffer.append(reward)
    if self.explore: reward = 0
    truncated = self.max_episode_steps is not None and len(self.reward_buffer) >= self.max_episode_steps
    if truncated: info = {'termination_reason': 'TIME', **info}; self.termination_resaons.append('TIME')
    # if truncated or terminated: info = {'episode': {'l': len(self.rewards), 'r': sum(self.rewards)}, 'history':self._history(), **info}
    if self.sparse: reward = 0 if not (terminated or truncated) else sum(self.reward_buffer)
    return state, reward, terminated, truncated, info
