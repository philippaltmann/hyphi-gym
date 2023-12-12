import gymnasium as gym; from typing import Optional; import numpy as np
from gymnasium.utils.seeding import np_random
from gymnasium.envs.registration import EnvSpec

# Rewards 
GOAL, STEP, FAIL = 1/2, -1, -1/2

# Stochasticity
RAND_KEY = ['Agent', 'Target']; 
RAND_KEYS = ['Agents', 'Targets']; 
RAND = [*RAND_KEY, *RAND_KEYS]

class Base(gym.Env):
  """ Base Env Implementing 
  • Truncated Episodes (via `max_episode_steps`)
  • Sparse Rewards (defaults to False)
  • Detailed Rewards (defaults to False)
  • Reward Threshold for Early Stopping 
  • Exploration Mode without reward / target 
  • Supporting randomization upon init (`Agent` and `Target`), 
    or, upon reset (`Agents`, `Targets` and items in `RADD`)
  • Seeding nondeterministic environments
  • Generating dynamic spec obejct and env name based on the configuration"""

  _name: str; layout: Optional[np.ndarray] = None

  def __init__( 
      self, detailed=False, sparse=False, explore=False, can_fail=False,
      random=[], RADD=[], seed:Optional[int]=None, max_episode_steps=100
    ):
    self.max_episode_steps, self._spec = max_episode_steps, {}
    self.dynamic_spec = ['nondeterministic', 'max_episode_steps', 'reward_threshold']
    min_return = self.max_episode_steps * STEP + can_fail * FAIL * self.max_episode_steps
    max_return = self.max_episode_steps * GOAL #+ optimal_path * STEP * (self.step_scale == 1)
    self.reward_range, self.reward_threshold = (min_return, max_return), 'VARY'
    self.sparse, self.detailed, self.explore = sparse, detailed, explore
    self.random = random; self.random.sort(); self.nondeterministic = len(self.random) > 0; self.seed(seed)
    assert all([r in [*RADD, *RAND] for r in random]), f'Please specify all random elements in {[*RADD, *RAND]}'
    if len(self.random): assert self.np_random is not None, "Please provide a seed to use nondeterministic features"
    rand_resets = len([r for r in random if r in [*RADD, *RAND_KEYS]])
    self.layout = self.randomize(self.layout, RAND_KEY, setup = not rand_resets)

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

  def _validate(self, layout:np.ndarray, error:bool, setup:bool) -> int: 
    """Overwrite this function to validate `layout` return steps to solution"""
    raise(NotImplementedError)

  def _randomize(self, layout:np.ndarray, key:str) -> tuple[np.ndarray]: 
    """Overwrite this function to randomize `layout` w.r.t. `key`
    Return old and new position"""
    raise(NotImplementedError)

  def randomize(self, layout, keys=RAND, setup=False):
    """Helper function to randomize all `keys` in `self.random`. 
    Randomization can be forced via setup"""
    if layout is None and not setup: return None
    if layout is None and setup: layout = self._generate()
    if len(random := [r for r in keys if r in self.random]) or setup: 
      layout = layout.copy(); [self._randomize(layout, r) for r in random]
      self.reward_threshold = self._reward_threshold(layout.copy(), setup)
      if self.reward_threshold is None: return self.randomize(layout, keys, setup)
    return layout
  
  def _generate(self)->np.ndarray:
    """Random generator function for a layout of self.specs"""
    raise(NotImplementedError)

  def reset(self, layout=None, **kwargs)->tuple[gym.spaces.Space, dict]:
    """Gymnasium compliant function to reset the environment""" 
    super().reset(**kwargs); self.reward_buffer, self.termination_resaons = [], []; 
    if 'seed' in kwargs and kwargs['seed'] is not None: # Randomize for new seeds
      self.seed(kwargs['seed']); self.layout = self.randomize(self.layout, RAND_KEY, setup=True) 
    layout = self._generate() if self.layout is None else layout if layout is not None else self.layout.copy()
    layout = self.randomize(layout, RAND_KEYS, setup=self.layout is None) # Setup if generated
    return layout
  
  def _reward_threshold(self, layout:Optional[np.ndarray]=None, setup=False):
    """Given a layout, calculates the min and max returns"""
    # if self.detailed: return (0, self.max_episode_steps)
    if (optimal_path := self._validate(layout, error=False, setup=setup)) > self.max_episode_steps: return None
    return self.max_episode_steps * GOAL + 1.2 * optimal_path * self.step_scale if layout is not None else 0 * STEP
  
  def execute(self, action:gym.spaces.Space) -> tuple[gym.spaces.Space, dict]: 
    """Overwrite this function to perfom step mutaions on the actual environment."""
    raise(NotImplementedError)
  
  def step(self, action:gym.spaces.Space) -> tuple[gym.spaces.Space, float, bool, bool, dict]:
    """Gymnasium compliant fucntion to step the environment with `action` using the internal `_step`"""
    state, info = self.execute(action)  # Step the environment 

    # Calculate the reward 
    terminated = 'termination_reason' in info
    if self.explore: reward = 0; terminated = False
    elif self.detailed: reward = np.exp(-info['distance'])
    # assert 'distance' in info, "Target distance information needed for detailed reward calulation"
    else: 
      reward = STEP
      if terminated: reward += self.max_episode_steps * (GOAL if info['termination_reason'] == 'GOAL' else FAIL)
    self.reward_buffer.append(reward)
    
    truncated = self.max_episode_steps is not None and len(self.reward_buffer) >= self.max_episode_steps
    if truncated: info = {'termination_reason': 'TIME', **info}
    if self.sparse: reward = 0 if not (terminated or truncated) else sum(self.reward_buffer)
    return state, reward, terminated, truncated, info
