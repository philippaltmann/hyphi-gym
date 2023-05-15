import re
from gymnasium.envs.registration import register, WrapperSpec
from hyphi_gym.wrappers import Monitor

def register_envs():
  register(id="HoleyGrid", entry_point="hyphi_gym.envs.HoleyGrid.holey_grid:HoleyGrid")
  register(id="GridMaze", entry_point="hyphi_gym.envs.Maze.maze:Maze") 

def named(name):
  """Enviroment creation helper, trasforms string name to make arguments.
  Usage: `gym.make(hyphi_gym.named(name))`
  Supported Envs: Any Sized Grid Mazes and and Holey Grids
  Supported Options: Sparse, Explore, Random-layout, -target, and -agent placement"""      
  random = [v for k,v in {'Mazes': 'layout', 'Targets': 'target', 'Agents': 'agent'}.items() if k in name]
  level = {}; random = []
  if 'Maze' in name: 
    if 'Mazes' in name: random.append('Layouts')
    size = int(re.findall(r'\d+', name)[0]); level= {'id':'GridMaze', 'size': size}
    name = name.replace(str(size),'').replace('Mazes','').replace('Maze','')
  if 'Holes' in name:
    level = {'id': 'HoleyGrid', 'level': 'Shift' if 'Shift' in name else 'Train'}
    name = name.replace('Holes','').replace('Shift','')
  args = {'sparse': 'Sparse' in name, 'explore': 'Explore' in name}
  name = name.replace('Sparse','').replace('Explore','')
  random = [*random, *re.findall('[A-Z][^A-Z]*', name)]
  return {**level, **args, 'random': random, }

