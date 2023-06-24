import numpy as np
from typing import Any, Dict, Tuple, Optional
from hyphi_gym.envs.common import Base

RAND_KEY = ['Agent', 'Target']; RAND_KEYS = ['Agents', 'Targets']; 

class Board(Base):
  """Board: Grid Based Games Base Class managing a `layout`of a variable `size`, extending `Base`.
  Containining `CELLS` ∈ `[WALL, FIELD, AGENT, TARGET, HOLE]`, navigatable with `ACTIONS` ∈ `[UP, RIGHT, DOWN,LEFT]`, 
  supporting the randomization of "Agent" and "Target" position on `__init__`, or "Agents" and "Targets" on `reset`."""

  board: np.ndarray; size:tuple[int,int]; layout:Optional[np.ndarray]=None
  UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3; ACTIONS = [UP, RIGHT, DOWN, LEFT] 
  WALL, FIELD, AGENT, TARGET, HOLE = '#', ' ', 'A', 'T', 'H' 
  CELLS = {WALL: 0, FIELD: 1, AGENT: 2, TARGET: 3, HOLE: 4}
  CHARS = list(CELLS.keys()); RAND = [*RAND_KEY, *RAND_KEYS]

  def __init__(self, size:tuple[int,int], layout:list[str], random=[], RAND=[], **kwargs):
    self.random = random; self.random.sort(); self.RAND=[*RAND,*self.RAND]; super().__init__(**kwargs); self.size = size 
    assert all([r in self.RAND for r in random]), f'Please specify all random elements in {self.RAND}' 
    if layout is not None: self.layout = self._grid(layout); [self.randomize(r[0], self.layout) for r in RAND_KEY if r in random]

  def ascii(self, grid:Optional[np.ndarray]=None) -> list[str]:
    """Transform 2D-INT Array to list of strings"""
    return [''.join([self.CHARS[c] for c in row]) for row in list(grid if grid is not None else self.board)]
  
  def _grid(self, ascii:list[str]) -> np.ndarray:
    """Transform 1D string to 2D-INT Array"""
    return np.array([[self.CELLS[char] for char in row] for row in ascii])
  
  def getpos(self, board:Optional[np.ndarray]=None, cell:str=AGENT) -> np.ndarray:
    """Position helper for finding the vector position of `cell` on the `board` or internal board"""
    return np.array(tuple(zip(*np.where((board if board is not None else self.board) == self.CELLS[cell])))[0]).astype(int)
  
  def newpos(self, position:Tuple[int,int], action:int, n=1) -> Tuple[int,int]:
    """Action helper mutating a `position` tuple by appying `action` `n`-times"""
    return tuple(np.array(position)+[(-n,0),(0,n),(n,0),(0,-n)][action])

  def iterate_actions(self, p:Tuple[int,int], n=1, condition=lambda act,pos,n: True) -> Dict[int,Tuple[int,int]]: 
    """Return possible n actions in a bounded box given a position p and their mutated positions"""
    return {a: self.newpos(p,a,n) for a in self.ACTIONS if condition(a,p,n)}
  
  def action_possible(self, act:int, pos:Tuple[int,int], n=1)->bool: 
    """Return possible `n` actions `act` in a bounded box given a position `pos`"""
    return [(pos[0]>n), (pos[1]<self.size[1]-n-1), (pos[0]<self.size[0]-n-1), (pos[1]>n)][act]

  def _generate(self)->np.ndarray:
    """Random generator function for a grid of size `self.size`"""
    inside = tuple(s-2 for s in self.size)
    agent = tuple(np.random.randint((0,0), inside)); target = agent
    while target is agent: target = tuple(np.random.randint((0,0), inside))
    grid = np.full(inside, self.CELLS[self.FIELD])
    grid[agent], grid[target] = self.CELLS[self.AGENT], self.CELLS[self.TARGET]
    grid = np.pad(grid, 1, constant_values=self.CELLS[self.WALL])
    return grid
  
  def randomize(self, cell:str, board:np.ndarray) -> tuple[tuple[int], tuple[int]]:
    """Mutation function to randomize the position of `cell` on `board`"""
    genpos = lambda: tuple([self.np_random.integers(1,s) for s in self.size])
    newpos = genpos(); oldpos = tuple(self.getpos(board=board,cell=cell))
    while self.CHARS[board[newpos]] != self.FIELD: newpos = genpos()
    board[oldpos] = self.CELLS[self.FIELD]; board[newpos] = self.CELLS[cell];
    return (oldpos, newpos)
    
  def _board(self, layout:Optional[np.ndarray], remove=[])->np.ndarray:
    """Get the current board according to an optional `layout` and the global random configuration"""
    if self.explore: remove = [*remove, self.TARGET]
    board = layout.copy() if layout is not None else self._generate()
    [self.randomize(key[0], board) for key in RAND_KEYS if key in self.random]
    for rm in remove: board[tuple(self.getpos(board, rm))] = self.CELLS[self.FIELD]
    return board 
