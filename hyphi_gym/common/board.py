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
    for s in size: assert s % 2 == 1 and 15 >= s >= 7, "Only odd sizes € [7,15] are supported."
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
    DIST = self.max_episode_steps+1 
    b = board.copy(); visited = np.full_like(board, False); m = visited.copy()
    APOS = self.getpos(b, cell=AGENT); b[tuple(APOS)]=CELLS[FIELD]
    TPOS = self.getpos(b, cell=TARGET); b[tuple(TPOS)]=CELLS[FIELD]    
    iter = lambda p: self.iterate_actions(p, condition=self.action_possible).values()
    acts = lambda p, mask: [a for a in iter(p) if b[a]==CELLS[FIELD] and not mask[a]]
    def _mask(p): m[tuple(p)] = True; [_mask(n) for n in acts(p,m)]; return m
    def _findPath(position, target, distance, d):
      """Find the path with the shortest distance on board starting from position to target, 
      using current min distance _d_ and visited states """
      if distance > self.bound: return d # Enforce min path length
      if all(p==t for p,t in zip(position, target)): return min(distance, d) # Break Condition
      visited[position] = True; dist = []
      for pos in acts(position, visited):
        dist.append(_findPath(pos, target, distance+1, d))
        if dist[-1] == self.bound: return self.bound
      visited[position] = False; return min([d, *dist])
    if _mask(APOS)[tuple(TPOS)]: DIST = _findPath(tuple(APOS), tuple(TPOS), 0, DIST)
    if error: assert DIST < self.max_episode_steps+1, 'Environment not solvable.\n'+"\n".join(self.ascii(board))
    if setup: self.tpos = TPOS
    return DIST
  
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