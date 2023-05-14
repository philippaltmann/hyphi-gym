import numpy as np
import gymnasium as gym 
import pygame

from numpy.typing import NDArray
from typing import Any, Dict, List, Tuple

from hyphi_gym.envs.common.base import *
from hyphi_gym.envs.common.env3D import Env3D
from hyphi_gym.envs.common.plotting import heatmap_2D

TARGET_REWARD, STEP_COST, HOLE_COST = 50, -1, -50
WALL, FIELD, AGENT, TARGET, HOLE = '#', ' ', 'A', 'T', 'H'
CELL_LOOKUP = [WALL, FIELD, AGENT, TARGET, HOLE]
CELL_SIZE = 64
CELL_RENDER = { # canvas, position, scale
  WALL: lambda c, p, s=CELL_SIZE: pygame.draw.rect(c, (32, 32, 32, 100), pygame.Rect(s*p,(s,s))),
  FIELD: lambda c, p, s=CELL_SIZE: pygame.draw.rect(c, (192, 192, 192, 100), pygame.Rect(s*p,(s,s))),
  HOLE: lambda c, p, s=CELL_SIZE: pygame.draw.rect(c, (0, 0, 0, 100), pygame.Rect(s*p,(s,s))),
  AGENT: lambda c, p, s=CELL_SIZE: pygame.draw.rect(c, (32, 128, 255), pygame.Rect((p+.1)*s,(s*.8,s*.8)), border_radius=int(s*.2)),
  TARGET: lambda c, p, s=CELL_SIZE: pygame.draw.rect(c, (32, 224, 64), pygame.Rect((p+.25)*s,(s*.5,s*.5)), border_radius=int(s*.25)),
}
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3; ACTIONS = [UP, RIGHT, DOWN, LEFT]
RAND_KEY = ['Agent', 'Target']; RAND_KEYS = ['Agents', 'Targets']; RAND = ['Layouts', *RAND_KEY, *RAND_KEYS]

class Grid(Base, Env3D):
  board: NDArray
  metadata = {"render_modes": ["2D", "3D", "ascii"], "render_fps": 4, "render_resolution": (960,720), 'tmp': '/tmp/env.png'} ## Alternative Resolutions: (320,240),(960,720),(1280,960),(1920,1440),(4096,3072)
  def __init__(self, size, layout, render_mode='2D', random=[], **kwargs):
    assert all([r in RAND for r in random]), f'Please specify all random elements in {RAND}' 
    self.random = random; self.random.sort(); super().__init__(**kwargs)
    if len(random): assert self._np_random is not None, "Please provide a seed to use nondeterministic features"
    self.observation_space = gym.spaces.MultiDiscrete(np.full(np.prod(size), len(CELL_LOOKUP)))
    self.size, self.action_space = size, gym.spaces.Discrete(4)
    self.layout = self._to_grid(layout) if layout else None
    [self.randomize(key[0], self.layout) for key in RAND_KEY if key in random]
    assert render_mode is None or render_mode in self.metadata["render_modes"]
    self.render_mode, self.scene = render_mode, None
    if render_mode == "3D" and layout: self.setup3D(self.layout)
    if render_mode == 'ascii': print('Rendering ASCII:\n'+'\n'*(self.size[0]))

  def ascii(self, grid): [print(f"{l}") for l in self._to_ascii(grid)]
  def _to_ascii(self, grid:NDArray[np.integer[Any]]) -> List[str]:
    """Transform 2D-INT Array to 1D string"""
    return [''.join([CELL_LOOKUP[c] for c in row]) for row in list(grid)]
  
  def _to_grid(self, ascii) -> NDArray[np.integer[Any]]:
    """Transform 1D string to 2D-INT Array"""
    return np.array([[CELL_LOOKUP.index(char) for char in row] for row in ascii])

  def _generate(self):
    inside = tuple(s-2 for s in self.size)
    agent = tuple(np.random.randint((0,0), inside)); target = agent
    while target is agent: target = tuple(np.random.randint((0,0), inside))
    grid = np.full(inside, CELL_LOOKUP.index(' '))
    grid[agent] = CELL_LOOKUP.index('A')
    grid[target] = CELL_LOOKUP.index('T')
    grid = np.pad(grid, 1, constant_values=CELL_LOOKUP.index('#'))
    return grid.astype(self.observation_space.dtype)

  def _validate(self, board):
    def _findPath(b, v, p, t, d, md):
      """Find the path with the shortest distance _d_ in maze _b_ starting from _p_ to _t_, 
      using current min distance _md_ and visited states _v_ """
      if all(a==b for a,b in zip(p,t)): return min(d, md) # Break Condition
      v[p] = True; next = [tuple(n) for n in self.iterate_actions(p, condition=self.action_possible).values()]
      dist = [_findPath(b, v, n, t, d+1, md) for n in next if (b[n]==CELL_LOOKUP.index(FIELD) and not v[n])]
      v[p] = False # backtrack: remove p from the visited matrix
      return min([md, *dist])
    APOS = self.getpos(board=board, cell=AGENT); board[tuple(APOS)]=CELL_LOOKUP.index(FIELD)
    TPOS = self.getpos(board=board, cell=TARGET); board[tuple(TPOS)]=CELL_LOOKUP.index(FIELD)
    visited = np.full_like(board, False)
    D = _findPath(board, visited, tuple(APOS), tuple(TPOS), 0, self.max_episode_steps+1)
    assert D < np.inf, f'Environment not solvable.\n{self._to_ascii(board)}'
    return TARGET_REWARD - D
 
  def renderAscii(self, clr=False, write=False):
    RST = f"\x1B[{self.board.shape[0]+2}A" if clr else "" ; CLR = "\x1B[0K" if clr else ""
    return f'{RST}Step: {len(self.reward_buffer)}{CLR}\n' + f'{CLR}\n'.join(self._to_ascii(self.board))+f'{CLR}\n'

  def render2D(self):
    window_size = tuple(np.array(self.size)[::-1] * CELL_SIZE); canvas = pygame.Surface(window_size); canvas.fill((192, 192, 192))
    [CELL_RENDER[CELL_LOOKUP[int(cell)]](canvas, np.array([x,y])) for y, row in enumerate(self.board) for x, cell in enumerate(row)]
    for x in range(self.size[0] + 1): pygame.draw.line(canvas, 0, (0, CELL_SIZE * x), (window_size[0], CELL_SIZE * x), width=1)
    for x in range(self.size[1] + 1): pygame.draw.line(canvas, 0, (CELL_SIZE * x, 0), (CELL_SIZE * x, window_size[1]), width=1)
    return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))  
    
  def render(self):
    if self.render_mode not in self.metadata['render_modes']: return 
    if self.render_mode == "2D": return self.render2D()
    if self.render_mode == "3D": return self.render3D()
    if self.render_mode == 'ascii': return self.renderAscii(clr=True)   

  def randomize(self, cell, board):
    genpos = lambda: tuple([self.np_random.integers(1,s) for s in self.size])
    newpos = genpos(); oldpos = tuple(self.getpos(board=board,cell=cell))
    while CELL_LOOKUP[board[newpos]] != FIELD: newpos = genpos()
    # print(f"Moving {cell} from {oldpos} to {newpos}")   
    board[oldpos] = CELL_LOOKUP.index(FIELD); board[newpos] = CELL_LOOKUP.index(cell) 
    return board #; print(board)
    
  def _board(self, layout, remove=[]):
    board = layout.copy() if layout is not None else self._generate()
    [self.randomize(key[0], board) for key in RAND_KEYS if key in self.random]
    for rm in remove: board[tuple(self.getpos(board, rm))] = CELL_LOOKUP.index(FIELD)
    maximum_reward = -100 if len(remove) else self._validate(board.copy())
    self.reward_range = (self.max_episode_steps*STEP_COST, maximum_reward)
    return board 
  
  """Gym API functions"""
  def reset(self, **kwargs):
    super().reset(**kwargs); self.board = self._board(self.layout, remove=[TARGET] if self.explore else [])
    if self.render_mode == '3D': self.reset3D()
    if self.render_mode == 'ascii': self.render() 
    return self.board.flatten(), {}
    
  def _reward(self, board, action):
    reward, termination = STEP_COST, None; position = self.getpos(board)
    target = self.newpos(position, action); field = CELL_LOOKUP[board[target]] 
    if field == TARGET: termination, reward = 'GOAL', TARGET_REWARD + STEP_COST
    if field == HOLE: termination, reward = 'FAIL', HOLE_COST + STEP_COST
    return reward, termination, position, target, field

  def _step(self, action:int) -> Tuple[NDArray, int, bool, bool, Dict[str,Any]]:
    reward, termination, position, target, field = self._reward(self.board, action)
    info = {'termination_reason': termination} if termination is not None else {} 
    if field is not WALL: self.board[tuple(position)] = CELL_LOOKUP.index(FIELD)  # Move Agent 
    if field in [FIELD, TARGET]: self.board[target] = CELL_LOOKUP.index(AGENT)  # Update Board 
    if self.render_mode == '3D': self.update3D(action, target, field)
    if self.render_mode == 'ascii': self.render()
    return self.board.flatten(), reward, termination is not None, False, info

  # Helper Functions for stepping  
  def getpos(self, board=None, cell=AGENT):
    if board is None: board = self.board 
    return np.array(tuple(zip(*np.where(board == CELL_LOOKUP.index(cell))))[0])

  def newpos(self, position, action, n=1): return tuple(np.array(position)+[(-n,0),(0,n),(n,0),(0,-n)][action])

  def iterate_actions(self, p, n=1, condition=lambda act,pos,n: True): 
    """Return possible n actions in a bounded box given a position p and their mutated positions"""
    return {a: self.newpos(p,a,n) for a in ACTIONS if condition(a,p,n)}
  
  def action_possible(self, act, pos, n=1): 
    """Return possible n actions in a bounded box given a position p"""
    return [(pos[0]>n), (pos[1]<self.size[1]-n-1), (pos[0]<self.size[0]-n-1), (pos[1]>n)][act]
  
  # Helperfunctions for plotting heatmaps
  def iterate(self, function = lambda e,s,a,r: r, fallback=None):
    """Iterate all possible actions in all env states, apply `funcion(env, state, action)`
    function: `f(env, state, action, reward()) => value` to be applied to all actions in all states 
      default: return envreward
    :returns: ENVxACTIONS shaped function results"""
    empty_board = self._board(self.layout, [AGENT]); fallback = [fallback] * len(ACTIONS)
    def prepare(x,y): state = empty_board.copy(); state[y][x] = CELL_LOOKUP.index(AGENT); return state

    # Create empty board for iteration & function for reverting Observation to board 
    process = lambda state: [function(self, state, action, self._reward(state, action)[0]) for action in ACTIONS]
    return [ [ process(prepare(x,y)) if CELL_LOOKUP[cell] == FIELD else fallback for x, cell in enumerate(row) ] 
      for y, row in enumerate(empty_board) ]

  def heatmap(self, fn, args): 
    return heatmap_2D(self.iterate(fn), *args)

