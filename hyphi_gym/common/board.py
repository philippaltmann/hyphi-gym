import numpy as np; from typing import Optional, Union
from hyphi_gym.common import Base; import gymnasium as gym

# State Types
WALL, FIELD, AGENT, TARGET, HOLE = '#', ' ', 'A', 'T', 'H' 
CELLS = {WALL: 0, FIELD: 1, AGENT: 2, TARGET: 3, HOLE: 4}
CHARS = list(CELLS.keys()); 

# Actions
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3; ACTIONS = [UP, RIGHT, DOWN, LEFT] 

class Board(Base):
  """Base for grid-based games managing 
  • A `layout` of a variable `size`
  • Containing `CELLS` ∈ `[WALL, FIELD, AGENT, TARGET, HOLE]`
  • Navigable with `ACTIONS` ∈ `[UP, RIGHT, DOWN,LEFT]`"""

  board: np.ndarray; size:tuple[int,int]; layout:Optional[np.ndarray]=None

  def __init__(self, size:tuple[int,int], layout:Optional[list[str]], bound=None, **kwargs):
    self.layout = self._grid(layout) if layout is not None else None
    self.size = size; self.bound = bound or sum(size) - 6; Base.__init__(self, **kwargs)

  def ascii(self, grid:Optional[np.ndarray] = None) -> list[str]:
    """Transform 2D-INT Array to list of strings"""
    return [''.join([CHARS[c] for c in row]) for row in list(grid if grid is not None else self.board)]
  
  def _grid(self, ascii:list[str]) -> np.ndarray:
    """Transform 1D string to 2D-INT Array"""
    return np.array([[CELLS[char] for char in row] for row in ascii])
  
  def getpos(self, board:Optional[np.ndarray]=None, cell:str=AGENT) -> np.ndarray:
    """Position helper for finding the vector position of `cell` on the `board` or internal board"""
    return np.array(tuple(zip(*np.where((board if board is not None else self.board) == CELLS[cell])))[0]).astype(int)
  
  def newpos(self, position:Union[np.ndarray, tuple[int,int]], action:int, n=1) -> tuple[int,int]:
    """Action helper mutating a `position` tuple by appying `action` `n`-times"""
    return tuple(np.array(position)+[(-n,0),(0,n),(n,0),(0,-n)][action])

  def iterate_actions(self, p:tuple[int,int], n=1, condition=lambda act,pos,n: True) -> dict[int,tuple[int,int]]: 
    """Return possible n actions in a bounded box given a position p and their mutated positions"""
    return {a: self.newpos(p,a,n) for a in ACTIONS if condition(a,p,n)}
  
  def action_possible(self, act:int, pos:tuple[int,int], n=1)->bool: 
    """Return possible `n` actions `act` in a bounded box given a position `pos`"""
    return [(pos[0]>n), (pos[1]<self.size[1]-n-1), (pos[0]<self.size[0]-n-1), (pos[1]>n)][act]
  
  def _generate(self):
    """Random generator for flat grids, creating a grid of `size`"""
    board = np.pad(np.full(np.subtract(self.size, (2,2)), CELLS[FIELD]), 1, constant_values=CELLS[WALL])
    board[(self.size[0] - 2,1)] = CELLS[AGENT]; board[(1,self.size[0] - 2)] = CELLS[TARGET]
    return board

  def _validate(self, board, error=True, setup=False):
    board = board.copy()
    if 'Flat' in self._name: return sum(self.size)-6 # no-obstacle grids do not need to be validated

    def _findPath(b, v, p, t, d, md):
      """Find the path with the shortest distance _d_ in maze _b_ starting from _p_ to _t_, 
      using current min distance _md_ and visited states _v_ """
      if d > self.bound: return md # Enforce min path length
      if all(a==b for a,b in zip(p,t)): return min(d, md) # Break Condition
      v[p] = True; next = [tuple(n) for n in self.iterate_actions(p, condition=self.action_possible).values()]
      dist = [_findPath(b, v, n, t, d+1, md) for n in next if (b[n]==CELLS[FIELD] and not v[n])]
      v[p] = False; return min([md, *dist])
    b = board.copy(); visited = np.full_like(board, False)
    APOS = self.getpos(board=board, cell=AGENT); board[tuple(APOS)]=CELLS[FIELD]
    TPOS = self.getpos(board=board, cell=TARGET); board[tuple(TPOS)]=CELLS[FIELD]
    D = _findPath(board, visited, tuple(APOS), tuple(TPOS), 0, self.max_episode_steps+1)
    if error: assert D < self.max_episode_steps+1, 'Environment not solvable.\n'+"\n".join(self.ascii(b))
    if setup: self.tpos = TPOS
    return D
  
  def _update(self, key:str, oldpos, newpos):
    super()._update(key, oldpos, newpos)
    if key[0] == TARGET: self.tpos = newpos
  
  def _randomize(self, board:np.ndarray, key:str):
    """Mutation function to randomize the position of `cell` on `board`"""
    genpos = lambda: tuple([self.np_random.integers(1,s) for s in self.size])
    newpos = genpos(); oldpos = tuple(self.getpos(board=board,cell=key[0]))
    while CHARS[board[newpos]] != FIELD: newpos = genpos()
    board[oldpos] = CELLS[FIELD]; board[newpos] = CELLS[key[0]];
    self._update(key, oldpos, newpos)
    return (oldpos, newpos)
    
  def reset(self, **kwargs)->tuple[gym.spaces.Space, dict]:
    """Gymnasium compliant function to reset the environment""" 
    self.board = super().reset(**kwargs); return self.board.flatten(), {}