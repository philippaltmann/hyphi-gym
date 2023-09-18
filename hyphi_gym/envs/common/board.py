import numpy as np; from typing import Optional, Union
from hyphi_gym.envs.common import Base; import gymnasium as gym

# State Types
WALL, FIELD, AGENT, TARGET, HOLE = '#', ' ', 'A', 'T', 'H' 
CELLS = {WALL: 0, FIELD: 1, AGENT: 2, TARGET: 3, HOLE: 4}

# Actions
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3; ACTIONS = [UP, RIGHT, DOWN, LEFT] 

# Stochasticity
RAND_KEY = ['Agent', 'Target']; RAND_KEYS = ['Agents', 'Targets']; 
CHARS = list(CELLS.keys()); RAND = [*RAND_KEY, *RAND_KEYS]


class Board(Base):
  """Board: Grid Based Games Base Class managing a `layout`of a variable `size`, extending `Base`.
  Containining `CELLS` ∈ `[WALL, FIELD, AGENT, TARGET, HOLE]`, navigatable with `ACTIONS` ∈ `[UP, RIGHT, DOWN,LEFT]`, 
  supporting the randomization of "Agent" and "Target" position on `__init__`, or "Agents" and "Targets" on `reset`."""

  board: np.ndarray; size:tuple[int,int]; layout:Optional[np.ndarray]=None

  def __init__(self, size:tuple[int,int], layout:Optional[list[str]], random=[], RADD=[], **kwargs):
    self.random = random; self.random.sort(); Base.__init__(self, **kwargs); self.size = size
    assert all([r in [*RADD, *RAND] for r in random]), f'Please specify all random elements in {[*RADD, *RAND]}'
    if layout is not None: 
      self.layout = self._grid(layout); self.reward_range = self._reward_range(self.layout.copy())
      [self.randomize(r[0], self.layout) for r in RAND_KEY if r in random]

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

  def _generate(self)->np.ndarray:
    """Random generator function for a grid of size `self.size`"""
    raise(NotImplementedError)
  
  def _validate(self, board, error=True):
    def _findPath(b, v, p, t, d, md):
      """Find the path with the shortest distance _d_ in maze _b_ starting from _p_ to _t_, 
      using current min distance _md_ and visited states _v_ """
      if all(a==b for a,b in zip(p,t)): return min(d, md) # Break Condition
      v[p] = True; next = [tuple(n) for n in self.iterate_actions(p, condition=self.action_possible).values()]
      dist = [_findPath(b, v, n, t, d+1, md) for n in next if (b[n]==CELLS[FIELD] and not v[n])]
      v[p] = False; return min([md, *dist])
    b = board.copy(); visited = np.full_like(board, False)
    APOS = self.getpos(board=board, cell=AGENT); board[tuple(APOS)]=CELLS[FIELD]
    TPOS = self.getpos(board=board, cell=TARGET); board[tuple(TPOS)]=CELLS[FIELD]
    D = _findPath(board, visited, tuple(APOS), tuple(TPOS), 0, self.max_episode_steps+1)
    if error: assert D < self.max_episode_steps+1, 'Environment not solvable.\n'+"\n".join(self.ascii(b))
    return D
  
  def randomize(self, cell:str, board:np.ndarray) -> tuple[tuple[int], tuple[int]]:
    """Mutation function to randomize the position of `cell` on `board`"""
    genpos = lambda: tuple([self.np_random.integers(1,s) for s in self.size])
    newpos = genpos(); oldpos = tuple(self.getpos(board=board,cell=cell))
    while CHARS[board[newpos]] != FIELD: newpos = genpos()
    board[oldpos] = CELLS[FIELD]; board[newpos] = CELLS[cell];
    return (oldpos, newpos)
    
  def _board(self, layout:Optional[np.ndarray], remove=[], update=False)->np.ndarray:
    """Get the current board according to an optional `layout` and the global random configuration, 
    optionally update globally"""
    board = layout.copy() if layout is not None else self._generate()
    [self.randomize(key[0], board) for key in RAND_KEYS if key in self.random]
    for rm in remove: board[tuple(self.getpos(board, rm))] = CELLS[FIELD]
    if update:
      if self._validate(board.copy(), error=False) > self.max_episode_steps: 
        return self._board(layout,remove,update)
      self.board = board
      self.tpos = self.getpos(board, TARGET)
    return board 

  def reset(self, **kwargs)->tuple[gym.spaces.Space, dict]:
    """Gymnasium compliant function to reset the environment""" 
    super().reset(**kwargs); self._board(self.layout, update=True)
    if self.layout is None: self.reward_range = self._reward_range(self.board.copy())
    return self.board.flatten(), {}