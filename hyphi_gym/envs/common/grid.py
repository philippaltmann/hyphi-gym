import gymnasium as gym ; import numpy as np
from typing import Optional; from os import path

from hyphi_gym.envs.common.board import *
from hyphi_gym.envs.common.rendering import Rendering
from hyphi_gym.envs.common.simulation import Simulation

class Grid(Simulation, Rendering): 
  base_xml = path.join(path.dirname(path.realpath(__file__)), "../../assets/grid.xml")

  metadata = {"render_modes": ["2D", "3D", "blender"], "render_fps": 5, "render_resolution": (720,720)} 
  def __init__(self, render_mode:Optional[str]=None, **simargs):
    self.observation_space = gym.spaces.MultiDiscrete(np.full(np.prod(self.size), len(CHARS)))
    self.action_space = gym.spaces.Discrete(4); self.action_space.seed(self._seed)
    assert render_mode is None or render_mode in self.metadata["render_modes"]; self.render_mode = render_mode
    if self.layout is not None: self.reward_range = self._reward_range(self.layout.copy())
    if render_mode is not None: 
      self.renderer = Rendering if render_mode == 'blender' else Simulation
      self.renderer.__init__(self, grid=True, **simargs)
        
  def render(self) -> Optional[np.ndarray]:
    """Return rendering of current state as np array if render_mode set"""
    if self.render_mode not in self.metadata['render_modes']: return 
    return self.renderer.render(self)

  """Gym API functions"""
  def reset(self, **kwargs)-> tuple[np.ndarray,dict]:
    super().reset(**kwargs); self._board(self.layout, update=True)
    if self.layout is None: self.reward_range = self._reward_range(self.board.copy())
    if self.render_mode is not None: self.renderer.reset_world(self)
    return self.board.flatten(), {}
    
  def _reward(self, board:np.ndarray, action:int)->tuple[float, Optional[str], np.ndarray, tuple[int,int], str]:
    """Calculate the reward of `action` for the current `borad`.
    :Return: reward, termination reason, agent positon, new position, new field"""
    reward, termination = STEP_COST, None; position = self.getpos(board)
    target = self.newpos(position, action); field = CHARS[board[target]] 
    if field == TARGET: termination, reward = 'GOAL', TARGET_REWARD + STEP_COST
    if field == HOLE: termination, reward = 'FAIL', HOLE_COST + STEP_COST
    return reward, termination, position, target, field
  
  def _reward_range(self, board:np.ndarray):
    """Given a `board` layout, calculates the min and max returns"""
    maximum_reward = -100 if self.explore else TARGET_REWARD + self._validate(board) * STEP_COST
    self.reward_range = (self.max_episode_steps*STEP_COST, maximum_reward); return self.reward_range

  def _step(self, action:int) -> tuple[np.ndarray, float, bool, bool, dict]:
    """Helper function to step the environment, executing `action`, returning its consequences"""
    reward, termination, position, target, field = self._reward(self.board, action)
    info = {'termination_reason': termination} if termination is not None else {} 
    if field is not WALL: self.board[tuple(position)] = CELLS[FIELD]  # Move Agent 
    if field in [FIELD, TARGET]: self.board[target] = CELLS[AGENT]  # Update Board 
    if self.render_mode is not None: self.renderer.update_world(self, action, target, field)
    return self.board.flatten(), reward, termination is not None, False, info
    
  # Helperfunctions for plotting heatmaps
  def iterate(self, function = lambda e,s,a,r: r, fallback=None):
    """Iterate all possible actions in all env states, apply `funcion(env, state, action, reward)`
    function: `f(env, state, action, reward()) => value` to be applied to all actions in all states default: return envreward
    :returns: ENVxACTIONS shaped function results"""
    empty_board = self._board(self.layout, [AGENT]); fallback = [fallback] * len(ACTIONS)
    def prepare(x,y): state = empty_board.copy(); state[y][x] = CELLS[AGENT]; return state

    # Create empty board for iteration & function for reverting Observation to board 
    process = lambda state: [function(self, state, action, self._reward(state, action)[0]) for action in ACTIONS]
    return [ [ process(prepare(x,y)) if CHARS[cell] == FIELD else fallback for x, cell in enumerate(row) ] 
      for y, row in enumerate(empty_board) ]
