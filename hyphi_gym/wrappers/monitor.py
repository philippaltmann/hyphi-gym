import time; import numpy as np; import torch as th
from typing import Any, Dict, List, SupportsFloat, Tuple

import gymnasium as gym
from gymnasium.core import ActType, ObsType
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from hyphi_gym.utils import stdout_redirected

class Monitor(gym.Wrapper[ObsType, ActType, ObsType, ActType]):
  """ A monitor wrapper for Gym environments, it is used to know the episode reward, length, time and other data.
  :param env: The environment """
  def __init__( self, env: gym.Env, record_video=False):
    super().__init__(env=env); self.t_start = time.time(); 
    self.record_video = record_video; self._frame_buffer = []
    self.states: List[Any] = []; self.actions: List[Any] = []; self.rewards: List[float] = []
    self._history = lambda: {key: getattr(self,key).copy() for key in ['states','actions','rewards']}
    self._episode_returns: List[float] = []; self._termination_reasons: List[Any] = []
    self._episode_lengths: List[int] = []; self._episode_times: List[float] = []; 
    self._total_steps = 0; self.needs_reset = True

  def reset(self, **kwargs) -> Tuple[ObsType, Dict[str, Any]]:
    """ Calls the Gym environment reset. 
    :param kwargs: Extra keywords saved for the next episode. only if defined by reset_keywords
    :return: the first observation of the environment """
    self.needs_reset = False
    state, info = self.env.reset(**kwargs)
    self.rewards = []; self.states=[state]; self.actions = []
    if self.record_video: self._frame_buffer.append(self.render())
    return state, info

  def step(self, action: ActType) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
    """ Step the environment with the given action
    :param action: the action
    :return: observation, reward, terminated, truncated, information """
    if self.needs_reset: raise RuntimeError("Tried to step environment that needs reset")
    state, reward, terminated, truncated, info = self.env.step(action)
    self.states.append(state); self.actions.append(action); self.rewards.append(float(reward))
    if self.record_video: self._frame_buffer.append(self.render())
    if terminated or truncated:
      self.needs_reset = True; ep_rew = sum(self.rewards); ep_len = len(self.rewards)
      ep_info = {"r": round(ep_rew, 6), "l": ep_len, "t": round(time.time() - self.t_start, 6), 'history': self._history()}
      self._episode_returns.append(ep_rew); self._termination_reasons.append(info.pop('termination_reason'))
      self._episode_lengths.append(ep_len); self._episode_times.append(time.time() - self.t_start)
      info["episode"] = ep_info
    self._total_steps += 1
    return state, reward, terminated, truncated, info

  def save_video(self, path, reset=True):
    """Saves current videobuffer to file"""
    with stdout_redirected(): ImageSequenceClip(self._frame_buffer, fps=5).write_videofile(path)
    if reset: self._frame_buffer = []

  def write_video(self, writer, label, step):
    """Adds current videobuffer to tensorboard"""
    video = th.tensor(np.array(self._frame_buffer)).unsqueeze(0).swapaxes(3,4).swapaxes(2,3)
    writer.add_video(label,video, global_step=step); self._frame_buffer = []
  
  @property
  def total_steps(self) -> int: return self._total_steps

  @property
  def episode_returns(self) -> List[float]: return self._episode_returns

  @property
  def termination_reasons(self) -> List[Any]: return self._termination_reasons

  @property
  def episode_lengths(self) -> List[int]: return self._episode_lengths

  @property
  def episode_times(self) -> List[float]: return self._episode_times
