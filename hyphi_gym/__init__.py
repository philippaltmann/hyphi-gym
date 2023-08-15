import re
from gymnasium.envs.registration import register
from hyphi_gym.wrappers import Monitor
from functools import reduce

def register_envs():
  register(id="HoleyGrid", entry_point="hyphi_gym.envs.HoleyGrid.holey_grid:HoleyGrid")
  register(id="HoleyPlane", entry_point="hyphi_gym.envs.HoleyPlane.holey_plane:HoleyPlane")
  register(id="GridMaze", entry_point="hyphi_gym.envs.GridMaze.grid_maze:GridMaze") 
  register(id="PointMaze", entry_point="hyphi_gym.envs.PointMaze.point_maze:PointMaze")

def named(name):
  """Enviroment creation helper, trasforms string name to make arguments.
  Usage: `gym.make(hyphi_gym.named(name))`
  Supported Envs: Any Sized Grid Mazes and and Holey Grids
  Supported Options: Sparse, Explore, Random-layout, -target, and -agent placement"""      
  level = {}; random = []
  if 'Maze' in name: 
    if 'Mazes' in name: random.append('Layouts')
    id = 'PointMaze' if 'Point' in name else 'GridMaze'
    size = int(re.findall(r'\d+', name)[0]); level= {'id':id, 'size': size}
    name = reduce(lambda n,r: n.replace(r,''), [str(size),'Points','Point','Mazes','Maze'], name)
  if 'Holey' in name:
    if 'Grids' in name or 'Planes' in name: random.append('Layouts')
    id = 'HoleyGrid' if 'Grid' in name else 'HoleyPlane'
    level = {'id': id, 'level': 'Shift' if 'Shift' in name else 'Train'}
    name = reduce(lambda n,r: n.replace(r,''), ['Holey','Shift','Planes','Plane','Grids','Grid'], name)
  args = {'sparse': 'Sparse' in name, 'explore': 'Explore' in name}
  name = name.replace('Sparse','').replace('Explore','')
  random = [*random, *re.findall('[A-Z][^A-Z]*', name)]
  return {**level, **args, 'random': random, }

#Sparse, Explore, Agent, Target, Agents, Targets
ENVS = [
  'HoleyGrid', 'HoleyGridShift',  'HoleyGrids', 'HoleyPlane', 'HoleyPlaneShift',  'HoleyPlanes', 
  [[[f'{base}Maze{s}', f'{base}Mazes{s}'] for s in [7,9,11,13,15]] for base in ['Grid', 'Point']],
]