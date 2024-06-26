import time; import numpy as np; import warnings

import gymnasium as gym
from gymnasium.core import ActType, ObsType; 
from typing import SupportsFloat
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from hyphi_gym.utils import stdout_redirected
from hyphi_gym.common import Grid

from PIL import Image

class Monitor(gym.Wrapper[ObsType, ActType, ObsType, ActType]):
  """ A monitor wrapper for Gym environments, it is used to know the episode reward, length, time and other data.
  :param env: The environment """
  def __init__( self, env: gym.Env, record_video=False):
    super().__init__(env=env); self.t_start = time.time(); 
    self.discrete = isinstance(env.unwrapped, Grid); self.policy = 'MlpPolicy' if self.discrete else 'MultiInputPolicy'
    self.record_video = record_video; self._frame_buffer = []
    self.states: list = [np.ndarray]; self.actions:list = [np.ndarray]; self.rewards: list[float] = []
    self._history = lambda: {key: np.array(getattr(self,key).copy()) for key in ['states','actions','rewards']}
    self._episode_returns: list[float] = []; self._termination_reasons: list[str] = []
    self._episode_lengths: list[int] = []; self._episode_times: list[float] = []; 
    self._total_steps = 0; self.needs_reset = True

  def reset(self, **kwargs) -> tuple[ObsType, dict]:
    """ Calls the Gym environment reset. 
    :param kwargs: Extra keywords saved for the next episode. only if defined by reset_keywords
    :return: the first observation of the environment """
    self.needs_reset = False
    state, info = self.env.reset(**kwargs)
    self.rewards = []; self.states=[state]; self.actions = []
    if self.record_video: self._frame_buffer.append(self.render())
    return state, info

  def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict]:
    """ Step the environment with the given action
    :param action: the action
    :return: observation, reward, terminated, truncated, information """
    if self.needs_reset: raise RuntimeError("Tried to step environment that needs reset")
    state, reward, terminated, truncated, info = self.env.step(action)
    self.states.append(state); self.actions.append(action); self.rewards.append(float(reward))
    if self.record_video: self._frame_buffer.append(self.render())
    if terminated or truncated:
      self.needs_reset = True; ep_rew = sum(self.rewards); ep_len = len(self.rewards); self.states.pop(-1)
      ep_info = {"r": round(ep_rew, 6), "l": ep_len, "t": round(time.time() - self.t_start, 6), 'history': self._history()}
      ep_info['reward_threshold'] = self.env.unwrapped.reward_threshold
      self._episode_returns.append(ep_rew); 
      self._termination_reasons.append(info.pop('termination_reason'))
      self._episode_lengths.append(ep_len); self._episode_times.append(time.time() - self.t_start)
      info["episode"] = ep_info
    self._total_steps += 1
    return state, reward, terminated, truncated, info
  
  def get_video(self, reset=True):
    frame_buffer = self._frame_buffer.copy()
    if reset: self._frame_buffer = []
    return np.array(frame_buffer)
  

  def save_video(self, path:str, reset=True):
    """Saves current videobuffer to file"""
    with stdout_redirected(): 
      if '.gif' in path:
        from PIL import Image
        if isinstance(self._frame_buffer[0], np.ndarray): imgs = [Image.fromarray(img) for img in self._frame_buffer]
        else: imgs = [img for img in self._frame_buffer]
        imgs[0].save(path, save_all=True, append_images=imgs[1:], optimize=False, duration=1000/self.env.metadata['render_fps'], loop=0)
      else: ImageSequenceClip(self._frame_buffer, fps=self.env.metadata['render_fps']).write_videofile(path)
    if reset: self._frame_buffer = []

  def write_video(self, writer, label, step):
    """Adds current videobuffer to tensorboard"""
    if len(self._frame_buffer) > 100: warnings.warn("Saving videos longer than one episode can be slow.")
    assert False, "TODO: write to tb"
    # video = th.tensor(np.array(self._frame_buffer)).unsqueeze(0).swapaxes(3,4).swapaxes(2,3)
    # writer.add_video(label,video, global_step=step); self._frame_buffer = []

  def save_image(self, path)->None:
    render = self.env.render(); assert render is not None
    if isinstance(render, np.ndarray): render = Image.fromarray(render)
    assert isinstance(render, Image.Image); render.save(path)
  
  @property
  def total_steps(self) -> int: return self._total_steps

  @property
  def episode_returns(self) -> list[float]: return self._episode_returns

  @property
  def termination_reasons(self) -> list[str]: return self._termination_reasons

  @property
  def episode_lengths(self) -> list[int]: return self._episode_lengths

  @property
  def episode_times(self) -> list[float]: return self._episode_times
