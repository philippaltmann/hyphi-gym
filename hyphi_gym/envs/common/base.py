import gymnasium as gym 
from gymnasium.envs.registration import EnvSpec

STOP_PERCENTAGE = 0.95

class Base(gym.Env):
  """ Base Env Implementing 
  • Sparse Rewards (defaults to False)
  • Early Stopping on reward threshold (defaults to False) TODO: MOVE TO WRAPPER?
  • Trucated Episodes (via `max_episode_steps`)"""
  def __init__(self, max_episode_steps=100, sparse=False, explore=False, seed=None): #stop=False,
    self.dynamic_spec = ['reward_threshold', 'nondeterministic', 'max_episode_steps']
    self.reward_buffer, self._spec, self.sparse, self.explore = None, {}, sparse, explore 
    self.max_episode_steps = max_episode_steps; self.termination_resaons = None
    self.nondeterministic = len(self.random) > 0; self.seed(seed)

  @property #if stop else None TODO: handle stopping via training param
  def reward_threshold(self): return round(.95 * self.reward_range[1])
  
  @property
  def name(self): return ''.join([self._name, *[n.capitalize() for n in ['explore', 'sparse'] if getattr(self,n)], *self.random])

  @property
  def spec(self): return EnvSpec(**{**self._spec, **{k:getattr(self,k) for k in self.dynamic_spec}})

  @spec.setter
  def spec(self, spec): self._spec = {k:v for k,v in spec.__dict__.items() if k not in ['namespace','name','version']}

  def reset(self, **kwargs): self.reward_buffer, self.termination_resaons = [], []; super().reset(**kwargs)

  def seed(self, seed=None): self._np_random, seed = gym.utils.seeding.np_random(seed) if seed is not None else (None, None)
  
  def step(self, action):
    state, reward, terminated, truncated, info = self._step(action); 
    if self.explore: reward = 0
    self.reward_buffer.append(reward)
    truncated = self.max_episode_steps is not None and len(self.reward_buffer) >= self.max_episode_steps
    if truncated: info = {'termination_reason': 'TIME', **info}; self.termination_resaons.append('TIME')
    if truncated or terminated: info = {'episode': {'l': len(self.reward_buffer), 'r': sum(self.reward_buffer)}, **info}
    if self.sparse: reward = 0 if not (terminated or truncated) else sum(self.reward_buffer)
    return state, reward, terminated, truncated, info
